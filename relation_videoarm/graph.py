from __future__ import annotations

import json
import math
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from .api import APIClient, parse_json_object


ALLOWED_RELATIONS = {"RESPONSE", "ELABORATE", "RETURN"}
ALLOWED_SUPER_RELATIONS = {"SAME_TOPIC"}
ALLOWED_DECISIONS = {"MERGE", "EXTEND", "TRANSITION", "NEW", "RESUME", "CONFLICT"}


def load_caption_blocks(database_path: str | Path, group_seconds: float = 10.0) -> tuple[list[dict[str, Any]], float]:
    payload = json.loads(Path(database_path).read_text(encoding="utf-8"))
    rows = sorted(payload.get("data") or [], key=lambda x: float(x.get("time_start_secs", 0)))
    duration = 0.0
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in rows:
        start = float(row.get("time_start_secs", 0))
        end = float(row.get("time_end_secs", start + 5))
        caption = str(row.get("caption") or "").strip()
        duration = max(duration, end)
        if not caption:
            continue
        caption = caption.split("\n", 1)[-1].strip()
        bucket = math.floor(start / group_seconds)
        if current is None or current["bucket"] != bucket:
            if current:
                blocks.append(current)
            current = {"bucket": bucket, "start": start, "end": end, "descriptions": [caption]}
        else:
            current["end"] = end
            current["descriptions"].append(caption)
    if current:
        blocks.append(current)
    compact = []
    for index, block in enumerate(blocks):
        text = " ".join(block["descriptions"])
        compact.append({
            "block_id": f"B{index:02d}",
            "start": round(float(block["start"]), 2),
            "end": round(float(block["end"]), 2),
            "description": text[:2400],
        })
    return compact, duration


GRAPH_SYSTEM = """You build timestamped Macro Events from chronological video description blocks.
The input is a sequence of 10-second-or-shorter descriptions with absolute timestamps. No question or answer is available.
Process the blocks chronologically. Consolidate repeated observations into coherent bounded events; do not create one event
per block. Preserve direct observations, state changes, exact time spans and uncertainty. Participants can be coarse roles.
Use RESPONSE when a later event responds to an earlier one, ELABORATE when it adds detail, and RETURN when the video goes
back to an earlier topic. Emit only grounded, useful directed links. Do not infer causes or intentions without evidence.

Return JSON only:
{"consolidation_trace":[{"block_id":"B00","decision":"MERGE|EXTEND|TRANSITION|NEW|RESUME|CONFLICT","event_id":"M01","reason":"short"}],
 "macro_events":[{"id":"M01","start":0.0,"end":30.0,"label":"event","summary":"grounded summary","key_steps":["step"],"state_before":"visible/unknown","state_after":"visible/unknown","evidence_times":[5.0],"confidence":0.0}],
 "relations":[{"source":"M01","target":"M02","type":"RESPONSE|ELABORATE|RETURN","evidence":"grounded reason"}],
 "conflicts":[{"claims":["claim A","claim B"],"interval":[0.0,10.0],"status":"unresolved"}]}
Every input block must appear exactly once in consolidation_trace. Relations must reference declared Macro IDs."""

SUPER_SYSTEM = """You build the middle level of a hierarchical video memory from already validated timestamped Macro Events.
Group Macro Events into coherent topic/interaction Super Events. Each Macro Event ID must occur in exactly one Super Event.
Super Event summaries should capture the phase/topic shared by their members while keeping the source time span.
Connect two Super Events only when they concern the same topic, using a SAME_TOPIC edge with a short evidence phrase.
Do not create other Super Event relation types and do not invent event content.

Return JSON only: {"super_events":[{"id":"S01","start":0.0,"end":60.0,"label":"topic","summary":"compact report","topic_tags":["topic"],"macro_ids":["M01"]}],
"super_relations":[{"source":"S01","target":"S02","type":"SAME_TOPIC","evidence":"shared topic evidence"}]}"""

ROOT_SYSTEM = """Write a compact Root report for a long video from its Super Event reports and their SAME_TOPIC links.
Summarize the overall activity, ordered phases, recurring patterns and unresolved ambiguity. Preserve distinctions between
observed events and interpretation. Do not add facts absent from the supplied memory.

Return JSON only: {"root":{"summary":"whole-video summary","overall_flow":["phase in order"],"repeated_patterns":["pattern"],"uncertainties":["ambiguity"]}}"""


def _validate(graph: dict[str, Any], blocks: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    macros = graph.get("macro_events") or []
    if not isinstance(macros, list) or not macros:
        raise ValueError("graph has no macro_events")
    cleaned = []
    seen: set[str] = set()
    for index, raw in enumerate(macros, 1):
        mid = str(raw.get("id") or f"M{index:02d}")
        if mid in seen:
            mid = f"M{index:02d}"
        seen.add(mid)
        start = max(0.0, min(float(raw.get("start", 0)), duration))
        end = max(start, min(float(raw.get("end", duration)), duration))
        cleaned.append({
            "id": mid,
            "start": round(start, 2),
            "end": round(end, 2),
            "label": str(raw.get("label") or "event").strip(),
            "summary": str(raw.get("summary") or raw.get("label") or "event").strip(),
            "key_steps": [str(x).strip() for x in raw.get("key_steps", []) if str(x).strip()][:8],
            "state_before": str(raw.get("state_before") or "unknown").strip(),
            "state_after": str(raw.get("state_after") or "unknown").strip(),
            "evidence_times": [round(max(start, min(float(x), end)), 2) for x in raw.get("evidence_times", [])[:8]],
            "confidence": max(0.0, min(float(raw.get("confidence", 0.5)), 1.0)),
        })
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    graph["macro_events"] = cleaned
    ids = {x["id"] for x in cleaned}
    relations = []
    for edge in graph.get("relations") or []:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        kind = str(edge.get("type", "")).upper()
        if source in ids and target in ids and kind in ALLOWED_RELATIONS:
            relations.append({"source": source, "target": target, "type": kind, "evidence": str(edge.get("evidence") or "")})
    graph["relations"] = relations
    trace = []
    known_blocks = {x["block_id"] for x in blocks}
    for item in graph.get("consolidation_trace") or []:
        bid, decision, event_id = str(item.get("block_id", "")), str(item.get("decision", "")).upper(), str(item.get("event_id", ""))
        if bid in known_blocks and decision in ALLOWED_DECISIONS and event_id in ids:
            trace.append({"block_id": bid, "decision": decision, "event_id": event_id, "reason": str(item.get("reason") or "")})
    if len({x["block_id"] for x in trace}) != len(blocks):
        raise ValueError("consolidation_trace does not cover every input block")
    graph["consolidation_trace"] = trace
    raw_supers = graph.get("super_events") or []
    supers = []
    for index, raw in enumerate(raw_supers, 1):
        sid = str(raw.get("id") or f"S{index:02d}")
        macro_ids = [str(mid) for mid in raw.get("macro_ids", []) if str(mid) in ids]
        if not macro_ids:
            continue
        members = [event for event in cleaned if event["id"] in macro_ids]
        supers.append({
            "id": sid,
            "start": max(0.0, min(float(raw.get("start", members[0]["start"])), duration)),
            "end": max(0.0, min(max(float(raw.get("end", members[-1]["end"])), float(raw.get("start", 0))), duration)),
            "label": str(raw.get("label") or sid).strip(),
            "summary": str(raw.get("summary") or raw.get("label") or sid).strip(),
            "topic_tags": [str(x).strip() for x in raw.get("topic_tags", []) if str(x).strip()][:8],
            "macro_ids": macro_ids,
        })
    if not supers and cleaned:
        supers = [{"id": "S01", "start": cleaned[0]["start"], "end": cleaned[-1]["end"],
                   "label": "全片事件", "summary": str((graph.get("root") or {}).get("summary", "全片事件")),
                   "topic_tags": [], "macro_ids": [event["id"] for event in cleaned]}]
    claimed = set()
    for item in supers:
        unique_ids = []
        for macro_id in item["macro_ids"]:
            if macro_id not in claimed:
                unique_ids.append(macro_id)
                claimed.add(macro_id)
        item["macro_ids"] = unique_ids
    supers = [item for item in supers if item["macro_ids"]]
    missing = [event for event in cleaned if event["id"] not in claimed]
    if missing:
        next_id = max([int(item["id"][1:]) for item in supers if item["id"].startswith("S") and item["id"][1:].isdigit()] or [0]) + 1
        supers.append({"id": f"S{next_id:02d}", "start": missing[0]["start"], "end": missing[-1]["end"],
                      "label": "待归类事件", "summary": "；".join(event["summary"] for event in missing),
                      "topic_tags": [], "macro_ids": [event["id"] for event in missing]})
    super_ids = {item["id"] for item in supers}
    super_relations = []
    for edge in graph.get("super_relations") or []:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        kind = str(edge.get("type", "")).upper()
        if source in super_ids and target in super_ids and source != target and kind in ALLOWED_SUPER_RELATIONS:
            super_relations.append({"source": source, "target": target, "type": kind,
                                    "evidence": str(edge.get("evidence") or "")})
    graph["super_events"] = supers
    graph["super_relations"] = super_relations
    graph.setdefault("root", {"summary": "", "overall_flow": [], "repeated_patterns": [], "uncertainties": []})
    graph.setdefault("conflicts", [])
    graph["duration"] = round(duration, 2)
    graph["schema_version"] = "relation-event-v2"
    return graph


def _call_json(client: APIClient, system: str, payload: Any, purpose: str, max_tokens: int) -> dict[str, Any]:
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    last: Exception | None = None
    for attempt in range(3):
        message = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            purpose=purpose,
            max_tokens=max_tokens,
        )
        try:
            return parse_json_object(message.get("content") or "")
        except Exception as exc:
            last = exc
            user += f"\nThe previous response was invalid JSON ({type(exc).__name__}: {exc}). Return one corrected complete JSON object."
    raise RuntimeError(f"could not build {purpose}: {last}")


def _cached_json(client: APIClient, system: str, payload: Any, purpose: str, max_tokens: int,
                checkpoint_dir: str | Path | None, stage: str) -> dict[str, Any]:
    if checkpoint_dir is None:
        return _call_json(client, system, payload, purpose, max_tokens)
    material = json.dumps({
        "stage": stage, "system": system, "payload": payload,
        "model": client.model, "provider": client.base_url, "purpose": purpose,
        "max_tokens": max_tokens, "schema": "relation-event-v2",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / f"{stage}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint and isinstance(cached.get("result"), dict):
            return cached["result"]
    except (OSError, ValueError, AttributeError):
        pass
    result = _call_json(client, system, payload, purpose, max_tokens)
    temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"fingerprint": fingerprint, "result": result},
                                    ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, cache_path)
    return result


def build_graph(client: APIClient, database_path: str | Path,
                checkpoint_dir: str | Path | None = None,
                progress_callback: Callable[[str, int], None] | None = None) -> dict[str, Any]:
    blocks, duration = load_caption_blocks(database_path)
    # Bound the model's output size. Each chunk is checkpointed separately so an
    # interrupted video resumes from the first incomplete time window.
    macro_graph: dict[str, Any] = {"macro_events": [], "relations": [],
                                   "consolidation_trace": [], "conflicts": []}
    previous_context: list[dict[str, Any]] = []
    for chunk_index, offset in enumerate(range(0, len(blocks), 12)):
        chunk = blocks[offset:offset + 12]
        partial = _cached_json(client, GRAPH_SYSTEM, {
            "description_blocks": chunk,
            "preceding_events_for_context_only": previous_context,
            "instruction": "Output events and trace only for description_blocks; preceding events are context only.",
        }, "macro_memory_build", 7000, checkpoint_dir, f"macro_{chunk_index:03d}")
        local_ids = {str(item.get("id")): f"M{len(macro_graph['macro_events']) + i:03d}"
                     for i, item in enumerate(partial.get("macro_events") or [], 1)}
        if not local_ids:
            raise ValueError(f"macro chunk {chunk_index} has no events")
        for item in partial.get("macro_events") or []:
            copied = dict(item)
            copied["id"] = local_ids[str(item.get("id"))]
            macro_graph["macro_events"].append(copied)
        for item in partial.get("consolidation_trace") or []:
            if str(item.get("event_id")) in local_ids:
                copied = dict(item)
                copied["event_id"] = local_ids[str(item.get("event_id"))]
                macro_graph["consolidation_trace"].append(copied)
        for item in partial.get("relations") or []:
            source, target = str(item.get("source")), str(item.get("target"))
            if source in local_ids and target in local_ids:
                copied = dict(item)
                copied["source"], copied["target"] = local_ids[source], local_ids[target]
                macro_graph["relations"].append(copied)
        macro_graph["conflicts"].extend(partial.get("conflicts") or [])
        previous_context = [{"id": item["id"], "summary": item.get("summary", "")}
                            for item in macro_graph["macro_events"][-2:]]
    if progress_callback:
        progress_callback("macro", 30)
    macro_events = macro_graph.get("macro_events") or []
    super_graph = _cached_json(client, SUPER_SYSTEM, {"macro_events": macro_events},
                               "super_memory_build", 5000, checkpoint_dir, "super")
    if progress_callback:
        progress_callback("super", 60)
    root_graph = _cached_json(client, ROOT_SYSTEM, {
        "super_events": super_graph.get("super_events") or [],
        "super_relations": super_graph.get("super_relations") or [],
    }, "root_memory_build", 2500, checkpoint_dir, "root")
    if progress_callback:
        progress_callback("root", 80)
    graph = {**macro_graph, **super_graph, **root_graph}
    try:
        graph = _validate(graph, blocks, duration)
    except Exception:
        # Do not keep syntactically valid but schema-invalid model output as a reusable checkpoint.
        if checkpoint_dir is not None:
            for stage in ("super", "root"):
                try:
                    (Path(checkpoint_dir) / f"{stage}.json").unlink()
                except FileNotFoundError:
                    pass
        raise
    graph["source_blocks"] = blocks
    graph["build_usage"] = client.usage.as_dict()
    graph["build_pipeline"] = "chunked_description_blocks_to_macros_to_supers_to_root"
    return graph
