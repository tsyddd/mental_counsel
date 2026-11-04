import base64
import hashlib
import io
import json
import os
import re
import uuid
from pathlib import Path

import av
import psycopg
import redis
from psycopg.rows import dict_row
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from relation_videoarm.api import APIClient, parse_json_object

DB = os.environ["DATABASE_URL"]
REDIS = os.getenv("REDIS_URL", "redis://redis:6379/0")
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/media")).resolve()
_encoder = None


def conn():
    return psycopg.connect(DB, row_factory=dict_row)


def queue():
    return redis.Redis.from_url(REDIS, decode_responses=True)


def embed(texts):
    global _encoder
    if not texts:
        return []
    if _encoder is None:
        TextEmbedding.add_custom_model(
            model="intfloat/multilingual-e5-small", pooling=PoolingType.MEAN,
            normalization=True, sources=ModelSource(hf="intfloat/multilingual-e5-small"),
            dim=384, model_file="onnx/model.onnx")
        _encoder = TextEmbedding(model_name=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
                                 cache_dir=os.getenv("FASTEMBED_CACHE_PATH", "/root/.cache/fastembed"))
    return [value.tolist() for value in _encoder.embed(texts)]


def vector_literal(values):
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def record_tool_memory(tool_name, input_text, status, summary="", error="", user_id=None, session_id=None):
    """Keep an operational trace and a scoped procedural/failure memory, never raw user text."""
    digest = hashlib.sha256((input_text or "").encode("utf-8")).hexdigest()
    run_id = uuid.uuid4()
    with conn() as db:
        db.execute("""INSERT INTO tool_runs(id,user_id,session_id,tool_name,input_digest,status,summary,error_summary)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id, user_id, session_id, tool_name, digest, status, (summary or "")[:500], (error or "")[:300]))
    if not user_id:
        return
    layer = "procedural" if status == "succeeded" else "failure"
    kind = "tool_success" if status == "succeeded" else "tool_failure"
    content = (f"{tool_name} completed successfully: {summary}" if status == "succeeded"
               else f"{tool_name} failed ({error[:240]}); input fingerprint {digest[:12]}.")
    vector = vector_literal(embed(["passage: " + content])[0])
    with conn() as db:
        previous = db.execute("""SELECT id FROM memories WHERE user_id=%s AND layer=%s AND kind=%s
            AND content=%s AND status='active' LIMIT 1""", (user_id, layer, kind, content)).fetchone()
        if previous:
            db.execute("""UPDATE memories SET salience=LEAST(3.0,salience+0.15),
                reinforcement_count=reinforcement_count+1,updated_at=now(),last_accessed_at=now() WHERE id=%s""",
                (previous["id"],))
        else:
            db.execute("""INSERT INTO memories(id,user_id,kind,layer,scope_session_id,content,source_type,source_id,
                confidence,metadata,embedding) VALUES (%s,%s,%s,%s,%s,%s,'tool_run',%s,0.7,%s,%s::vector)""",
                (uuid.uuid4(), user_id, kind, layer, session_id, content, str(run_id),
                 json.dumps({"tool_name": tool_name, "input_digest": digest}, ensure_ascii=False), vector))


def safe_media_path(raw):
    path = Path(raw).resolve()
    if not path.is_relative_to(MEDIA_ROOT) or not path.is_file():
        raise ValueError("video must be an existing file below MEDIA_ROOT")
    return path


def llm_json(system, payload, purpose, max_tokens=3000):
    client = APIClient()
    prompt = json.dumps(payload, ensure_ascii=False)
    for attempt in range(2):
        result = client.chat([
            {"role": "system", "content": system + "\n只输出有效 JSON 对象。"},
            {"role": "user", "content": prompt},
        ], purpose=purpose, max_tokens=max(4000, max_tokens))
        try:
            return parse_json_object(result.get("content") or "")
        except (ValueError, json.JSONDecodeError):
            prompt += "\n请修复 JSON 格式。"
    raise ValueError("model did not return valid JSON")


def transcribe(video_path):
    from faster_whisper import WhisperModel

    device = os.getenv("ASR_DEVICE", "cpu")
    model = WhisperModel(os.getenv("ASR_MODEL", "small"), device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(str(video_path), language="zh", vad_filter=True,
                                      initial_prompt="这是一次中文师生访谈。")
    rows = []
    for item in segments:
        value = item.text.strip()
        if value:
            rows.append({"time_start_secs": round(item.start, 2),
                         "time_end_secs": round(item.end, 2), "caption": value})
    if not rows:
        raise ValueError("ASR produced no speech segments")
    return rows


def _video_and_graph(video_id):
    with conn() as db:
        row = db.execute("SELECT path,graph,captions FROM videos WHERE id=%s", (video_id,)).fetchone()
    if not row or not row["graph"]:
        raise ValueError("video graph is not ready")
    return row


def search_graph(video_id, question, super_limit=3, macro_limit=5):
    row = _video_and_graph(video_id)
    graph = row["graph"]
    super_limit = min(3, max(1, int(super_limit)))
    macro_limit = min(5, max(1, int(macro_limit)))
    qvec = vector_literal(embed(["query: " + question])[0])
    with conn() as db:
        supers = db.execute("""SELECT id,content,start_sec,end_sec,data
            FROM graph_nodes WHERE video_id=%s AND kind='super'
            ORDER BY embedding <=> %s::vector LIMIT %s""",
            (video_id, qvec, super_limit)).fetchall()
    chosen = {item["id"] for item in supers}
    for edge in graph.get("super_relations", []):
        if len(chosen) >= super_limit:
            break
        if edge.get("type") == "SAME_TOPIC":
            a, b = edge.get("source"), edge.get("target")
            if a in chosen and b not in chosen:
                chosen.add(b)
            elif b in chosen and a not in chosen:
                chosen.add(a)
    super_map = {item["id"]: item for item in graph.get("super_events", [])}
    macros = {item["id"]: item for item in graph.get("macro_events", [])}
    candidates = [macros[mid] for sid in chosen for mid in super_map.get(sid, {}).get("macro_ids", []) if mid in macros]
    if not candidates:
        raise ValueError("no Macro candidates")
    anchor_response = llm_json(
        "根据问题，从候选事件中选最相关的一个 Macro 锚点。只使用给定 ID。返回 {\"anchor_id\":\"M01\"}。",
        {"question": question, "root": graph.get("root"), "candidates":
         [{"id": m["id"], "summary": m["summary"], "start": m["start"], "end": m["end"]} for m in candidates[:40]]},
        "macro_anchor", 1200)
    anchor = str(anchor_response.get("anchor_id", ""))
    if anchor not in {m["id"] for m in candidates}:
        anchor = candidates[0]["id"]
    visited, frontier = [], [anchor]
    while frontier and len(visited) < macro_limit:
        current = frontier.pop(0)
        if current in visited or current not in macros:
            continue
        visited.append(current)
        for edge in graph.get("relations", []):
            if edge.get("type") in ("RESPONSE", "ELABORATE", "RETURN"):
                if edge.get("source") == current:
                    frontier.append(edge.get("target"))
                elif edge.get("target") == current:
                    frontier.append(edge.get("source"))
    block_map = {item["block_id"]: item for item in graph.get("source_blocks", [])}
    evidence = []
    for trace in graph.get("consolidation_trace", []):
        if trace.get("event_id") in visited:
            block = block_map.get(trace.get("block_id"))
            if block:
                evidence.append(block)
    evidence.sort(key=lambda item: item["start"])
    return {"root": graph.get("root"), "supers": [super_map[s] for s in chosen if s in super_map],
            "anchor_id": anchor, "macros": [macros[mid] for mid in visited],
            "evidence": evidence}


def review_frames(video_path, start, end, count=4):
    path = safe_media_path(video_path)
    output = []
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        span = max(1.0, end - start)
        for idx in range(count):
            sec = max(0.0, start + span * (idx + 0.5) / count)
            video.seek(int(sec / float(stream.time_base)), stream=stream, backward=True)
            for frame in video.decode(stream):
                if frame.time is not None and frame.time + 0.02 < sec:
                    continue
                image = frame.to_image()
                image.thumbnail((640, 640))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
                output.append({"time_sec": round(sec, 2), "jpeg_base64":
                               base64.b64encode(buffer.getvalue()).decode("ascii")})
                break
    return output


def answer_video(video_id, question):
    record = _video_and_graph(video_id)
    hits = search_graph(video_id, question)
    macros = hits["macros"]
    if not macros:
        raise ValueError("no evidence window")
    start = max(0.0, min(item["start"] for item in macros) - 3)
    end = max(item["end"] for item in macros) + 3
    frames = review_frames(record["path"], start, min(end, start + 30), count=4)
    message = [{"type": "text", "text": json.dumps({"question": question,
        "evidence": hits["evidence"], "macros": macros}, ensure_ascii=False)}]
    message += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + item["jpeg_base64"]}}
                for item in frames]
    visual_status = "verified"
    try:
        response = APIClient().chat([
            {"role": "system", "content": "依据字幕及回看帧回答问题。区分可见事实与推断；引用证据块 ID 和时间戳。若证据不足，明确说明。"},
            {"role": "user", "content": message}], purpose="video_answer", max_tokens=2200)
        answer = response.get("content") or ""
        if not answer.strip():
            raise ValueError("vision model returned no answer")
    except Exception as exc:
        visual_status = "model_image_input_unavailable"
        response = APIClient().chat([
            {"role": "system", "content": "仅根据字幕证据回答，并明确标注尚未完成模型视觉核验；引用块 ID 和时间戳。"},
            {"role": "user", "content": json.dumps({"question": question, "evidence": hits["evidence"]}, ensure_ascii=False)}],
            purpose="caption_answer", max_tokens=2200)
        answer = response.get("content") or ""
    return {"answer": answer, "visual_status": visual_status,
            "frame_times": [item["time_sec"] for item in frames],
            "retrieval": hits}
