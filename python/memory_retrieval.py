"""Turn-scoped memory routing, hybrid retrieval, and evidence expansion."""

import math
import re
import uuid
from datetime import datetime, timezone

from .core import conn, embed, llm_json, vector_literal


def _terms(value):
    value = str(value or "").lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", value))
    chinese = re.findall(r"[\u4e00-\u9fff]", value)
    words.update("".join(chinese[i:i + 2]) for i in range(len(chinese) - 1))
    return words


def _lexical(a, b):
    left, right = _terms(a), _terms(b)
    return len(left & right) / max(1, len(left | right))


def route_turn(text, recent):
    text = text.strip()
    if re.fullmatch(r"(你好|您好|谢谢|好的|嗯|哦|再见|晚安|早安|hi|hello|thanks)[!！。,.，\s]*", text, re.I):
        return {"route": "none", "queries": []}
    # A direct question about a previously stated preference must search even
    # when it does not contain words such as "remember" or "previously".
    asks_preference = (re.search(r"(?:你知道|你记得|还记得|告诉我).{0,12}(?:我|本人).{0,8}(?:喜欢|偏好|不喜欢)", text)
                       or re.search(r"我(?:喜欢|偏好|不喜欢)(?:什么|哪门|哪种|哪些)", text))
    asks_history = bool(re.search(r"反复|经历|困扰|变化|发生过|那件事|谈到的事", text))
    if asks_preference and asks_history:
        queries = [part.strip() for part in re.split(r"[？?。；;]", text) if part.strip()]
        return {"route": "event_preference", "queries": queries[:2]}
    if asks_preference:
        return {"route": "preference", "queries": [text]}
    # Classify every substantive turn from its meaning and the recent dialogue.
    # A historical reference is a useful signal, never a prerequisite for recall.
    fallback = "goal" if any(token in text for token in ("目标", "进展", "计划", "改善得", "之前希望")) else "event"
    if any(token in text for token in ("回答方式", "怎么称呼", "我的偏好", "你还记得我喜欢")):
        fallback = "preference"
    try:
        result = llm_json(
            "你是记忆检索路由器，不回答用户问题。判断当前回复能否受益于这个用户的历史信息。"
            "仅从 none、event、goal、preference、event_goal、event_preference 中选择 route。"
            "简单寒暄、纯事实问答或本轮信息已完整且无需个性化时选 none；"
            "涉及用户经历、情绪起因、同类情况或含糊指代时选 event；"
            "涉及用户持续目标或进展时选 goal；涉及喜欢的事物、交流方式或选择倾向时选 preference；"
            "需要同时核对经历和目标时选 event_goal；同时询问偏好与经历时选 event_preference。"
            "即使没有‘之前’之类的词，只要历史信息能改善回答，也应检索。"
            "将省略指代结合最近对话改写为最多两个独立检索短句，不添加对话中没有的事实。"
            "返回 {route:string,queries:[string]}。",
            {"current_message": text, "recent_turns": recent[-4:]},
            "memory_route", 500)
        route = result.get("route", "none")
        if route not in {"none", "event", "goal", "preference", "event_goal", "event_preference"}:
            route = fallback
        queries = [str(item).strip()[:200] for item in result.get("queries", [])
                   if isinstance(item, str) and item.strip()][:2]
        return {"route": route, "queries": queries or ([text] if route != "none" else [])}
    except Exception:
        previous = next((item.get("content", "") for item in reversed(recent[:-1])
                         if item.get("role") == "user"), "")
        rewritten = previous + "；" + text if len(text) < 30 and previous else text
        return {"route": fallback, "queries": [rewritten[:200]]}


def _candidates(db, user_id, query, kinds):
    vector = vector_literal(embed(["query: " + query])[0])
    patterns = ["%" + term.replace("%", "\\%").replace("_", "\\_") + "%"
                for term in sorted(_terms(query), key=len, reverse=True)[:8]]
    if not patterns:
        patterns = ["%" + query[:30] + "%"]
    base = "user_id=%s AND status='active' AND layer=ANY(%s) AND (expires_at IS NULL OR expires_at>now())"
    sql = f"""WITH sem AS (
        SELECT id FROM memories WHERE {base} AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector LIMIT 40
    ), lex AS (
        SELECT id FROM memories WHERE {base} AND content ILIKE ANY(%s::text[])
        ORDER BY updated_at DESC LIMIT 30
    ), fresh AS (
        SELECT id FROM memories WHERE {base} ORDER BY updated_at DESC LIMIT 20
    ) SELECT id,kind,layer,content,source_type,source_id,metadata,confidence,
       salience,reinforcement_count,retrieval_count,updated_at,
       1-(embedding <=> %s::vector) AS similarity
       FROM memories WHERE id IN (SELECT id FROM sem UNION SELECT id FROM lex UNION SELECT id FROM fresh)
       AND kind=ANY(%s)"""
    args = (user_id, ["user", "episodic"], vector,
            user_id, ["user", "episodic"], patterns,
            user_id, ["user", "episodic"], vector, kinds)
    rows = db.execute(sql, args).fetchall()
    now = datetime.now(timezone.utc)
    for row in rows:
        age = max(0, (now - row["updated_at"]).total_seconds() / 86400)
        row["score"] = (0.60 * max(0, float(row["similarity"] or 0)) +
                        0.19 * _lexical(query, row["content"]) +
                        0.09 * math.exp(-age / 60) +
                        0.06 * min(1, float(row["salience"]) / 3) +
                        0.03 * min(1, math.log1p(row["reinforcement_count"]) / math.log(8)) +
                        0.03 * min(1, math.log1p(row["retrieval_count"]) / math.log(10)))
    return rows


def _source_window(db, row):
    if row["source_type"] != "chat" or not row["source_id"]:
        return []
    try:
        message_id = uuid.UUID(row["source_id"])
    except ValueError:
        return []
    anchor = db.execute("SELECT session_id,created_at FROM messages WHERE id=%s", (message_id,)).fetchone()
    if not anchor:
        return []
    before = db.execute("""SELECT role,content FROM messages WHERE session_id=%s
        AND created_at<%s ORDER BY created_at DESC,id DESC LIMIT 2""",
        (anchor["session_id"], anchor["created_at"])).fetchall()
    after = db.execute("""SELECT role,content FROM messages WHERE session_id=%s
        AND created_at>=%s ORDER BY created_at,id LIMIT 3""",
        (anchor["session_id"], anchor["created_at"])).fetchall()
    return [{"role": item["role"], "content": item["content"][:400]}
            for item in reversed(before)] + [
                {"role": item["role"], "content": item["content"][:400]} for item in after]


def retrieve_context(user_id, text, recent):
    decision = route_turn(text, recent)
    route = decision["route"]
    if route == "none":
        return {"route": route, "queries": [], "preferences": [], "events": [],
                "goals": [], "linked_intents": []}
    queries = decision["queries"]
    kinds = ({"event": ["person_event"], "goal": ["goal"],
              "preference": ["communication_preference", "preference"],
              "event_goal": ["person_event", "goal"],
              "event_preference": ["person_event", "communication_preference", "preference"]})[route]
    with conn() as db:
        ranked = {}
        for query in queries:
            for row in _candidates(db, user_id, query, kinds):
                key = row["id"]
                if key not in ranked or row["score"] > ranked[key]["score"]:
                    ranked[key] = row
        chosen = sorted(ranked.values(), key=lambda item: item["score"], reverse=True)
        # Relative and absolute gates prevent irrelevant top-k results entering the prompt.
        peak = chosen[0]["score"] if chosen else 0.0
        if route == "event_preference":
            # Preserve both requested memory types; one strong preference must
            # not crowd all event evidence out of a combined question.
            chosen = [row for row in chosen if row["score"] >= 0.40]
        else:
            chosen = [row for row in chosen if row["score"] >= max(0.48, peak - 0.10)][:5]
        event_rows = [row for row in chosen if row["kind"] == "person_event"][:3]
        goal_rows = [row for row in chosen if row["kind"] == "goal"][:2]
        preference_rows = [row for row in chosen if row["kind"] in
                           {"communication_preference", "preference"}][:2]
        selected_ids = [row["id"] for row in event_rows + goal_rows + preference_rows]
        concept_rows = []
        if selected_ids:
            concept_rows = db.execute("""SELECT id,label,memory_ids FROM concepts
                WHERE user_id=%s AND memory_ids && %s::uuid[] LIMIT 12""",
                (user_id, selected_ids)).fetchall()
        confirmed_ids = []
        for row in goal_rows:
            candidate = (row["metadata"] or {}).get("confirmed_intent_id")
            if candidate:
                try:
                    confirmed_ids.append(uuid.UUID(candidate))
                except ValueError:
                    pass
        concept_ids = [row["id"] for row in concept_rows]
        intent_rows = []
        if concept_ids or confirmed_ids:
            intent_rows = db.execute("""SELECT id,label,status,concept_ids,support_count,
                retrieval_count FROM intents WHERE user_id=%s
                AND (concept_ids && %s::uuid[] OR id=ANY(%s::uuid[]))
                AND status IN ('confirmed','tentative')
                ORDER BY support_count DESC,retrieval_count DESC LIMIT 8""",
                (user_id, concept_ids, confirmed_ids)).fetchall()
        missing_concepts = [concept_id for intent in intent_rows
                            for concept_id in intent["concept_ids"]
                            if concept_id not in concept_ids]
        if missing_concepts:
            concept_rows += db.execute("""SELECT id,label,memory_ids FROM concepts
                WHERE user_id=%s AND id=ANY(%s::uuid[]) LIMIT 12""",
                (user_id, missing_concepts)).fetchall()
        linked = []
        for intent in intent_rows:
            support = [concept for concept in concept_rows if
                       concept["id"] in intent["concept_ids"]]
            if intent["status"] == "tentative" and not support:
                continue
            linked.append({"id": str(intent["id"]), "label": intent["label"],
                           "status": intent["status"], "support_count": intent["support_count"],
                           "concepts": [item["label"] for item in support]})
        linked = linked[:2]
        related_ids = [memory_id for concept in concept_rows
                       for memory_id in concept["memory_ids"] if memory_id not in selected_ids]
        related_rows = []
        if related_ids and (event_rows or goal_rows):
            related_rows = db.execute("""SELECT id,content,source_type,source_id
                FROM memories WHERE user_id=%s AND id=ANY(%s::uuid[])
                AND kind='person_event' AND status='active'
                ORDER BY updated_at DESC LIMIT 2""", (user_id, related_ids)).fetchall()
        selected_ids += [row["id"] for row in related_rows]
        if selected_ids:
            db.execute("""UPDATE memories SET retrieval_count=retrieval_count+1,
                last_accessed_at=now() WHERE id=ANY(%s::uuid[])""", (selected_ids,))
        if linked:
            db.execute("""UPDATE intents SET retrieval_count=retrieval_count+1
                WHERE id=ANY(%s::uuid[])""", ([uuid.UUID(item["id"]) for item in linked],))
        events = [{"id": str(row["id"]), "content": row["content"],
                   "source_type": row["source_type"], "source_id": row["source_id"],
                   "source_window": _source_window(db, row), "score": round(row["score"], 3)}
                  for row in event_rows]
        goals = [{"id": str(row["id"]), "content": row["content"],
                  "score": round(row["score"], 3)} for row in goal_rows]
        preferences = [{"id": str(row["id"]), "content": row["content"]}
                       for row in preference_rows]
        related_events = [{"id": str(row["id"]), "content": row["content"],
                           "source_type": row["source_type"], "source_id": row["source_id"]}
                          for row in related_rows]
    return {"route": route, "queries": queries, "events": events,
            "goals": goals, "preferences": preferences, "linked_intents": linked,
            "related_events": related_events}
