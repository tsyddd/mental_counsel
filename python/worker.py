import json
import os
import re
import time
import traceback
import uuid
from pathlib import Path

from relation_videoarm.api import APIClient
from relation_videoarm.graph import build_graph

from .core import conn, embed, llm_json, queue, safe_media_path, vector_literal, transcribe, answer_video


def progress(job_id, value, status="running", result=None, error=None):
    with conn() as db:
        db.execute("""UPDATE jobs SET progress=%s,status=%s,result=%s,error=%s,updated_at=now()
            WHERE id=%s""", (value, status, json.dumps(result, ensure_ascii=False) if result is not None else None,
                          error, job_id))


def ingest(job):
    video_id = job["video_id"]
    path = safe_media_path(job["path"])
    progress(job["job_id"], 5)
    checkpoint = Path(os.getenv("CHECKPOINT_ROOT", "/checkpoints")) / video_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    caption_path = checkpoint / "captions.json"
    meta_path = checkpoint / "asr.meta.json"
    signature = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                 "model": os.getenv("ASR_MODEL", "small")}
    if caption_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == signature:
        rows = json.loads(caption_path.read_text(encoding="utf-8"))["data"]
    else:
        rows = transcribe(path)
        caption_path.write_text(json.dumps({"data": rows}, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(signature), encoding="utf-8")
    progress(job["job_id"], 25)
    graph = build_graph(APIClient(), caption_path, checkpoint_dir=checkpoint / "graph")
    progress(job["job_id"], 75)
    nodes = [{"id": "ROOT", "kind": "root", "parent": None,
              "content": graph["root"]["summary"], "start": 0.0,
              "end": graph["duration"], "data": graph["root"]}]
    parents = {}
    for item in graph["super_events"]:
        for mid in item["macro_ids"]:
            parents[mid] = item["id"]
        nodes.append({"id": item["id"], "kind": "super", "parent": "ROOT",
                      "content": item["summary"] + " " + " ".join(item.get("topic_tags", [])),
                      "start": item["start"], "end": item["end"], "data": item})
    for item in graph["macro_events"]:
        nodes.append({"id": item["id"], "kind": "macro", "parent": parents.get(item["id"]),
                      "content": item["summary"], "start": item["start"], "end": item["end"], "data": item})
    vectors = embed(["passage: " + item["content"] for item in nodes])
    with conn() as db:
        db.execute("DELETE FROM graph_edges WHERE video_id=%s", (video_id,))
        db.execute("DELETE FROM graph_nodes WHERE video_id=%s", (video_id,))
        db.execute("UPDATE videos SET captions=%s,graph=%s,status='ready' WHERE id=%s",
                   (json.dumps(rows, ensure_ascii=False), json.dumps(graph, ensure_ascii=False), video_id))
        for item, vector in zip(nodes, vectors):
            db.execute("""INSERT INTO graph_nodes(video_id,id,kind,parent_id,content,start_sec,end_sec,data,embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                (video_id, item["id"], item["kind"], item["parent"], item["content"],
                 item["start"], item["end"], json.dumps(item["data"], ensure_ascii=False), vector_literal(vector)))
        for edge in graph.get("relations", []) + graph.get("super_relations", []):
            db.execute("""INSERT INTO graph_edges(video_id,src,dst,relation,evidence) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING""",
                (video_id, edge["source"], edge["target"], edge["type"], edge.get("evidence", "")))
    progress(job["job_id"], 100, "completed", {"caption_segments": len(rows),
             "macro_events": len(graph["macro_events"]), "super_events": len(graph["super_events"])})


def generate_report(job):
    video_id = job["video_id"]
    questions = [str(q) for q in job.get("questions", []) if str(q).strip()][:12]
    if not questions:
        questions = ["学生如何描述近期学习生活？", "学生提到哪些人际支持或困难？", "有哪些需要继续了解的话题？"]
    progress(job["job_id"], 5)
    with conn() as db:
        existing = db.execute("SELECT report FROM reports WHERE video_id=%s", (video_id,)).fetchone()
    cached_answers = (existing or {}).get("report", {}).get("question_answers", [])
    answered = cached_answers if [item.get("question") for item in cached_answers] == questions else []
    for index, question in enumerate(questions[len(answered):], start=len(answered)):
        response = answer_video(video_id, question)
        answered.append({"question": question, "answer": response["answer"],
                         "visual_status": response["visual_status"],
                         "evidence_ids": [item["block_id"] for item in response["retrieval"]["evidence"]],
                         "frame_times": response["frame_times"]})
        progress(job["job_id"], 5 + int(65 * (index + 1) / len(questions)))
    skill = (Path(__file__).parent / "expert_skill.md").read_text(encoding="utf-8")
    with conn() as db:
        source = db.execute("SELECT graph FROM videos WHERE id=%s", (video_id,)).fetchone()
    compact = [{"question": item["question"], "answer": item["answer"][:1800],
                "evidence_ids": item["evidence_ids"]} for item in answered]
    output = llm_json(skill + "\n输出 JSON 对象，顶层只有 sections。sections 包含访谈概况、情绪与应对、学校与家庭、安全待核实、后续访谈建议。",
                      {"video_id": video_id, "root": source["graph"].get("root"), "question_answers": compact},
                      "expert_report", 6500)
    report = output.get("sections") or output.get("report") or output
    profile_output = llm_json(
        "仅从下列访谈问答提取可用于后续个性化交流的明确自述或沟通偏好。"
        "不做诊断，不把访谈者的问题写成学生事实，不推断模糊的安全表述。"
        "返回 JSON：{\"observations\":[{\"text\":\"...\",\"kind\":\"fact|preference|state\","
        "\"evidence_ids\":[\"B01\"],\"confidence\":0.8}],\"communication_suggestions\":[\"...\"]}。"
        "每条 observation 必须有给定 evidence_ids 中的至少一个。",
        {"question_answers": compact}, "profile_extract", 5000)
    profile = profile_output.get("profile") or profile_output
    allowed = {eid for item in answered for eid in item["evidence_ids"]}
    observations = []
    for item in profile.get("observations", []):
        if not isinstance(item, dict):
            continue
        refs = [str(eid) for eid in item.get("evidence_ids", []) if str(eid) in allowed]
        if refs:
            observations.append({"text": str(item.get("text", ""))[:500],
                                 "kind": str(item.get("kind", "observation"))[:30],
                                 "evidence_ids": refs,
                                 "confidence": max(0.0, min(float(item.get("confidence", 0.5)), 1.0))})
    profile["observations"] = observations
    profile["source_video_id"] = video_id
    with conn() as db:
        db.execute("""INSERT INTO reports(video_id,report,profile) VALUES (%s,%s,%s)
            ON CONFLICT(video_id) DO UPDATE SET report=excluded.report,profile=excluded.profile,created_at=now()""",
            (video_id, json.dumps({"sections": report, "question_answers": answered}, ensure_ascii=False),
             json.dumps(profile, ensure_ascii=False)))
    progress(job["job_id"], 100, "completed", {"video_id": video_id,
             "answered_questions": len(answered), "profile_observations": len(observations)})


def seed_profile(job):
    with conn() as db:
        row = db.execute("SELECT profile FROM reports WHERE video_id=%s", (job["video_id"],)).fetchone()
    if not row:
        raise ValueError("report not ready")
    user_id = job["user_id"]
    observations = row["profile"].get("observations", [])
    for item in observations:
        text = item.get("text", "")
        if not text:
            continue
        refs = item.get("evidence_ids", [])
        source_id = job["video_id"] + ":" + ",".join(refs)
        remember(user_id, text, item.get("kind", "observation"), "profile", source_id,
                 float(item.get("confidence", 0.5)), {"evidence_ids": refs, "video_id": job["video_id"]},
                 layer="user")
    progress(job["job_id"], 100, "completed", {"seeded": len(observations)})


def terms(text):
    text = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.update("".join(chinese[i:i+2]) for i in range(max(0, len(chinese)-1)))
    return tokens


def overlap(a, b):
    left, right = terms(a), terms(b)
    return len(left & right) / max(1, len(left | right))


def remember(user_id, text, kind, source_type, source_id, confidence, metadata,
             layer="user", scope_session_id=None, force_new=False):
    vector = vector_literal(embed(["passage: " + text])[0])
    with conn() as db:
        candidates = db.execute("""SELECT id,content,(embedding <=> %s::vector) AS distance
            FROM memories WHERE user_id=%s AND layer=%s AND kind=%s AND status='active'
            AND (scope_session_id IS NULL OR scope_session_id IS NOT DISTINCT FROM %s::uuid)
            AND embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT 6""",
            (vector, user_id, layer, kind, scope_session_id, vector)).fetchall()
        for existing in candidates if not force_new else []:
            similarity = 1.0 - float(existing["distance"])
            lexical = overlap(text, existing["content"])
            # Similar situations are separate episodes. The broad rule below is
            # appropriate for stable preferences, but collapses distinct
            # school events that share a topic or actors.
            same_memory = ((similarity >= 0.97 and lexical >= 0.32) or
                           (similarity >= 0.91 and lexical >= 0.65)) if kind == "person_event" else (
                               similarity >= 0.91 or (similarity >= 0.83 and lexical >= 0.12))
            if same_memory:
                db.execute("""UPDATE memories SET confidence=GREATEST(confidence,%s),
                    salience=LEAST(3.0,salience+0.15),reinforcement_count=reinforcement_count+1,
                    last_accessed_at=now(),updated_at=now(),
                    metadata=metadata || %s::jsonb WHERE id=%s""",
                    (confidence, json.dumps({"last_source_id": source_id,
                        "last_reinforced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                        ensure_ascii=False), existing["id"]))
                return str(existing["id"]), False
        mid = uuid.uuid4()
        db.execute("""INSERT INTO memories(id,user_id,kind,layer,scope_session_id,content,source_type,source_id,
              confidence,metadata,embedding,expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,
              CASE WHEN %s IN ('state','episode_summary') OR %s='session' THEN now()+interval '30 days' ELSE NULL END)""",
            (mid, user_id, kind, layer, scope_session_id, text, source_type, source_id, confidence,
             json.dumps(metadata, ensure_ascii=False), vector, kind, layer))
    return str(mid), True


def fold_concept(user_id, label, memory_id, intent_label, event_text):
    label_vector = vector_literal(embed(["passage: " + label])[0])
    event_terms = sorted(terms(event_text))
    with conn() as db:
        candidates = db.execute("""SELECT id,label,keywords,memory_ids,support_count,
                (embedding <=> %s::vector) AS distance FROM concepts WHERE user_id=%s
                AND embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT 8""",
                (label_vector, user_id, label_vector)).fetchall()
        match = None
        for candidate in candidates:
            similarity = 1.0 - float(candidate["distance"])
            shared = set(candidate["keywords"] or []) & set(event_terms)
            lexical = len(shared) / max(1, len(set(candidate["keywords"] or []) | set(event_terms)))
            if similarity >= 0.90 or (similarity >= 0.78 and lexical >= 0.08):
                match = candidate
                break
        if match:
            row = db.execute("""UPDATE concepts SET
                support_count=support_count+CASE WHEN %s::uuid=ANY(memory_ids) THEN 0 ELSE 1 END,
                memory_ids=CASE WHEN %s::uuid=ANY(memory_ids) THEN memory_ids ELSE array_append(memory_ids,%s::uuid) END,
                keywords=(SELECT ARRAY(SELECT DISTINCT x FROM unnest(keywords || %s::text[]) x)),
                strength=LEAST(3.0,strength+0.2),last_seen_at=now()
                WHERE id=%s RETURNING id,support_count""",
                (memory_id, memory_id, memory_id, event_terms, match["id"])).fetchone()
        else:
            row = db.execute("""INSERT INTO concepts(id,user_id,label,keywords,embedding,memory_ids)
                VALUES (%s,%s,%s,%s,%s::vector,ARRAY[%s::uuid])
                ON CONFLICT(user_id,label) DO UPDATE SET
                  support_count=concepts.support_count+1,
                  memory_ids=array_append(concepts.memory_ids,%s::uuid),
                  strength=LEAST(3.0,concepts.strength+0.2),last_seen_at=now()
                RETURNING id,support_count""",
                (uuid.uuid4(), user_id, label, event_terms, label_vector, memory_id, memory_id)).fetchone()
    if row["support_count"] >= 3 and row["support_count"] % 3 == 0:
        surface_intent(user_id, row["id"], intent_label)


def surface_intent(user_id, concept_id, candidate_hint=""):
    """Judge intent from several grounded episodes after a concept reaches density."""
    with conn() as db:
        concept = db.execute("""SELECT label,memory_ids,support_count FROM concepts
            WHERE user_id=%s AND id=%s""", (user_id, concept_id)).fetchone()
        if not concept or concept["support_count"] < 3:
            return
        existing = db.execute("""SELECT id FROM intents WHERE user_id=%s
            AND %s::uuid=ANY(concept_ids) AND status IN ('tentative','confirmed') LIMIT 1""",
            (user_id, concept_id)).fetchone()
        if existing:
            db.execute("UPDATE intents SET support_count=GREATEST(support_count,%s) WHERE id=%s",
                       (concept["support_count"], existing["id"]))
            return
        events = db.execute("""SELECT id,content FROM memories WHERE user_id=%s
            AND id=ANY(%s::uuid[]) AND layer='episodic' AND status='active'
            ORDER BY created_at,id LIMIT 30""", (user_id, concept["memory_ids"])).fetchall()
    if len(events) < 3:
        return
    evidence = [{"id":str(item["id"]),"text":item["content"]} for item in events]
    try:
        proposal = llm_json(
            "你负责从同一个人的多条情景证据中识别可供咨询师询问的近期跟进意图。"
            "仅当至少两条不同经历共同显示尚未解决的需求、正在尝试改变的方向或明确目标时，"
            "才提出一个简短、具体、非诊断性的 tentative_intent。"
            "已经解决、明确拒绝、普通重复日常、第三方经历或证据不足时返回 null。"
            "候选提示只是线索，不是事实。返回 JSON："
            "{tentative_intent:string|null,support_ids:[支持该判断的事件ID]}。",
            {"concept":concept["label"],"events":evidence,"candidate_hint":candidate_hint},
            "intent_emergence", 500)
    except Exception as exc:
        print(f"intent emergence skipped: {type(exc).__name__}", flush=True)
        return
    intent = str(proposal.get("tentative_intent") or "").strip()
    if intent.lower() in {"none", "null", "n/a", "无", "暂无"}:
        return
    valid_ids = {item["id"] for item in evidence}
    support_ids = proposal.get("support_ids") or []
    if not intent or not isinstance(support_ids,list) or len(valid_ids & set(map(str,support_ids))) < 2:
        return
    with conn() as db:
        db.execute("""INSERT INTO intents(id,user_id,label,concept_ids,support_count)
            VALUES (%s,%s,%s,ARRAY[%s::uuid],%s)
            ON CONFLICT(user_id,label) DO UPDATE SET
            concept_ids=CASE WHEN %s::uuid=ANY(intents.concept_ids) THEN intents.concept_ids
              ELSE array_append(intents.concept_ids,%s::uuid) END,
            support_count=GREATEST(intents.support_count,excluded.support_count),updated_at=now()""",
            (uuid.uuid4(), user_id, intent, concept_id, concept["support_count"],
             concept_id, concept_id))


def extract_memory(job):
    first_id = job["first_message_id"]
    last_id = job["last_message_id"]
    with conn() as db:
        turns = db.execute("""SELECT id,role,content FROM messages WHERE session_id=%s
            AND created_at >= (SELECT created_at FROM messages WHERE id=%s)
            AND created_at <= (SELECT created_at FROM messages WHERE id=%s)
            ORDER BY created_at,id LIMIT 14""",
            (job["session_id"], first_id, last_id)).fetchall()
    if not turns:
        raise ValueError("memory batch has no messages")
    user_ids = {str(turn["id"]) for turn in turns if turn["role"] == "user"}
    last_user_id = next(str(turn["id"]) for turn in reversed(turns)
                        if turn["role"] == "user")
    text = "；".join(turn["content"] for turn in turns if turn["role"] == "user")
    query_vector = vector_literal(embed(["query: " + text])[0])
    video_profile = {}
    with conn() as db:
        nearby = db.execute("""SELECT id,content,kind FROM memories WHERE user_id=%s
            AND status='active' ORDER BY embedding <=> %s::vector LIMIT 5""",
            (job["user_id"], query_vector)).fetchall()
        pending_intents = db.execute("""SELECT id,label FROM intents WHERE user_id=%s
            AND status='tentative' ORDER BY updated_at DESC LIMIT 5""",
            (job["user_id"],)).fetchall()
        session = db.execute("SELECT video_id FROM sessions WHERE id=%s", (job["session_id"],)).fetchone()
        if session and session["video_id"]:
            report = db.execute("SELECT profile FROM reports WHERE video_id=%s", (session["video_id"],)).fetchone()
            if report:
                video_profile = report["profile"] or {}
    candidates = [{"id": str(item["id"]), "content": item["content"], "kind": item["kind"]}
                  for item in nearby]
    pending = [{"id": str(item["id"]), "label": item["label"]}
               for item in pending_intents]
    turns = [{"id": str(turn["id"]), "role": turn["role"], "content": turn["content"]}
             for turn in turns]
    payload = llm_json(
        "视频报告中的人物画像是此人的第一版长期记忆。后续对话事件是在画像背景下新增的子事件；"
        "先用画像理解事件与人物的关系，但不能把画像原文或稳定特征重复抽成新事件。"
        "你是每五个用户回合运行一次的记忆写入路由器。先判断这批话语是否真的产生新的事件、偏好、目标或更正；"
        "没有新信息则所有数组为空。只识别这批用户话语新提供或确认的信息。请区分两类："
        "person_events 是与这个人有关的具体情景记忆，必须包含情境/行为/结果中的至少两项；"
        "如果当前消息明确讲述了已发生的具体经历，优先将该经历写入 person_events；"
        "即使同一消息也表达了未来目标，也不要因此遗漏这条情景事件，目标可另存为 stable_memories。"
        "偏好、目标、稳定事实直接放 stable_memories，不要硬改写成事件。"
        "交流方式偏好使用 kind=communication_preference；其他偏好用 preference；明确目标用 goal；明确事实用 fact。"
        "不要把模型猜测当事实。"
        "同一旧事件被重复提及时不要重复生成；明确信息纠正时提供 supersedes_memory_id。"
        "结合人物画像下多轮对话中的子事件归纳主题；仅当多个不同情景事件呈现重复主题时给出 tentative_intent，描述可供用户确认的近期需求/跟进方向，"
        "不能是诊断、人格标签或自动行动。"
        "只有最近一条咨询师消息明确询问了某个待确认方向，且当前用户明确确认或否认时，才输出 intent_feedback；"
        "不要从普通回答或沉默推断确认。输出 JSON："
        "{person_events:[{summary,concept,tentative_intent,confidence,source_message_id}],"
        "stable_memories:[{text,kind,confidence,supersedes_memory_id,source_message_id}],"
        "intent_feedback:[{intent_id,status}]}。status 只能是 confirmed 或 rejected。没有新增信息时数组为空。",
        {"video_profile_initial_long_term_memory": video_profile,
         "batch_turns": turns,
         "related_existing_memories": candidates,
         "pending_intents": pending}, "memory_extract", 1800)
    ids = []
    for event in payload.get("person_events", [])[:5]:
        if not isinstance(event, dict):
            continue
        content = str(event.get("summary", "")).strip()
        confidence = max(0.0, min(float(event.get("confidence", 0.5)), 1.0))
        if not content or confidence < 0.65:
            continue
        source_id = str(event.get("source_message_id", ""))
        if source_id not in user_ids:
            continue
        mid, created = remember(job["user_id"], content, "person_event", "chat",
            source_id, confidence, {"session_id": job["session_id"],
                                    "batch_no": job["batch_no"]}, layer="episodic")
        ids.append(mid)
        concept = str(event.get("concept") or "").strip()
        if concept.lower() in {"none", "null", "n/a", "无", "暂无"}:
            concept = ""
        if created and concept:
            intent_label = str(event.get("tentative_intent") or "").strip()
            if intent_label.lower() in {"none", "null", "n/a", "无", "暂无"}:
                intent_label = ""
            fold_concept(job["user_id"], concept, mid,
                         intent_label, content)
    for fact in payload.get("stable_memories", [])[:5]:
        if not isinstance(fact, dict):
            continue
        content = str(fact.get("text", "")).strip()
        if not content:
            continue
        confidence = max(0.0, min(float(fact.get("confidence", 0.5)), 1.0))
        if confidence < 0.65:
            continue
        kind = str(fact.get("kind", "fact"))
        if kind not in {"communication_preference", "preference", "goal", "fact"}:
            kind = "fact"
        superseded = str(fact.get("supersedes_memory_id", ""))
        source_id = str(fact.get("source_message_id", ""))
        if source_id not in user_ids:
            continue
        mid, _ = remember(job["user_id"], content, kind, "chat", source_id, confidence,
            {"session_id": job["session_id"], "batch_no": job["batch_no"]},
            layer="user", force_new=bool(superseded))
        ids.append(mid)
        if superseded in {item["id"] for item in candidates} and superseded != mid:
            with conn() as db:
                db.execute("""UPDATE memories SET status='superseded',updated_at=now(),
                    metadata=metadata || jsonb_build_object('superseded_by',%s)
                    WHERE id=%s AND user_id=%s""", (mid, superseded, job["user_id"]))
    pending_by_id = {item["id"]: item["label"] for item in pending}
    for feedback in payload.get("intent_feedback", [])[:2]:
        if not isinstance(feedback, dict):
            continue
        intent_id = str(feedback.get("intent_id", ""))
        status = str(feedback.get("status", ""))
        if intent_id not in pending_by_id or status not in {"confirmed", "rejected"}:
            continue
        with conn() as db:
            db.execute("""UPDATE intents SET status=%s,updated_at=now()
                WHERE id=%s AND user_id=%s AND status='tentative'""",
                (status, intent_id, job["user_id"]))
        if status == "confirmed":
            goal_id, _ = remember(job["user_id"], pending_by_id[intent_id], "goal",
                "chat", last_user_id, 0.9,
                {"session_id": job["session_id"], "confirmed_intent_id": intent_id},
                layer="user")
            ids.append(goal_id)
    progress(job["job_id"], 100, "completed", {"memory_ids": ids})


def record_job_failure(job, exc):
    user_ids = set()
    if job.get("user_id"):
        user_ids.add(job["user_id"])
    elif job.get("video_id"):
        with conn() as db:
            rows = db.execute("SELECT DISTINCT user_id FROM sessions WHERE video_id=%s", (job["video_id"],)).fetchall()
        user_ids.update(row["user_id"] for row in rows)
    for user_id in user_ids:
        text = f"{job.get('kind', 'background task')} failed: {str(exc)[:240]}"
        try:
            remember(user_id, text, "tool_failure", "job", job.get("job_id", ""), 0.75,
                     {"task_kind": job.get("kind"), "video_id": job.get("video_id")}, layer="failure")
        except Exception:
            traceback.print_exc()


HANDLERS = {"ingest": ingest, "report": generate_report,
            "seed_profile": seed_profile, "extract_memory": extract_memory}


def consolidate_idle_episodes():
    with conn() as db:
        episodes = db.execute("""SELECT e.session_id,e.episode_no,s.user_id
            FROM episodes e JOIN sessions s ON s.id=e.session_id
            WHERE e.consolidated_at IS NULL AND
              (e.status='closed' OR s.last_message_at<now()-interval '30 minutes')
            ORDER BY e.started_at LIMIT 5""").fetchall()
    for episode in episodes:
        with conn() as db:
            messages = db.execute("""SELECT role,content FROM messages
                WHERE session_id=%s AND episode_no=%s ORDER BY created_at LIMIT 80""",
                (episode["session_id"], episode["episode_no"])).fetchall()
        if messages:
            result = llm_json("从一段已暂停的对话提炼简短的阶段摘要，保留用户明确的目标、偏好和待续话题，"
                              "不要把推断当事实。返回 {summary:string}。",
                              {"messages": messages}, "episode_consolidation", 1600)
            summary = str(result.get("summary", "")).strip()
            if summary:
                remember(episode["user_id"], summary, "episode_summary", "episode",
                         f"{episode['session_id']}:{episode['episode_no']}", 0.75,
                         {"session_id": str(episode["session_id"]), "episode_no": episode["episode_no"]},
                         layer="session", scope_session_id=str(episode["session_id"]))
        with conn() as db:
            db.execute("""UPDATE episodes SET consolidated_at=now() WHERE session_id=%s AND episode_no=%s""",
                       (episode["session_id"], episode["episode_no"]))


def main():
    client = queue()
    last_consolidation = 0.0
    while True:
        from .mindtrace import process_next
        if process_next():
            continue
        if os.getenv('LEGACY_MEMORY_WORKER','0') == '1' and time.time() - last_consolidation > 60:
            try:
                consolidate_idle_episodes()
            except Exception:
                traceback.print_exc()
            last_consolidation = time.time()
        item = client.brpop("interview:jobs", timeout=1)
        if item is None:
            continue
        job = None
        try:
            job = json.loads(item[1])
            handler = HANDLERS[job["kind"]]
            handler(job)
        except Exception as exc:
            if isinstance(job, dict) and job.get("job_id"):
                progress(job["job_id"], 0, "failed", error=str(exc)[:1500])
                record_job_failure(job, exc)
            traceback.print_exc()


if __name__ == "__main__":
    main()
