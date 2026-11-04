"""Pure, independently testable policies for the two-layer MindTrace memory."""
import math
import re


def normalize(text):
    return re.sub(r"[\W_]+", "", str(text).lower())


def terms(text):
    value = str(text).lower()
    chinese = re.findall(r"[\u4e00-\u9fff]", value)
    return set(re.findall(r"[a-z0-9]+", value)) | set(zip(chinese, chinese[1:]))


def lexical(a, b):
    x, y = terms(a), terms(b)
    return len(x & y) / max(1, len(x | y))


def cosine(a, b):
    if len(a) != len(b):
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / max(1e-12,
        math.sqrt(sum(x*x for x in a) * sum(y*y for y in b)))


def rank_events(events, query, vector, limit=12):
    return sorted(events, key=lambda e: .75*cosine(vector, e['embedding']) +
                  .25*lexical(query, e['content']), reverse=True)[:limit]


def bounded_window(intents, events, max_chars=2400):
    """Whole entries only. A fixed character budget, not an unverified token budget."""
    lines, selected_events, selected_intents = [], [], []
    def append(line):
        if len('\n'.join(lines + [line])) <= max_chars:
            lines.append(line)
            return True
        return False
    for i in intents:
        line = (f"[意图 {i['id']} | 会话 {i['source_session']} 第{i['extracted_turn']}轮形成"
                f" | {i['status']}] {i['label']}")
        if append(line):
            selected_intents.append(i['id'])
    for e in events:
        line = (f"[事件 {e['id']} | 会话 {e['source_session']} 第{e['start_turn']}-{e['end_turn']}轮"
                f" | 发生时间 {e.get('occurred_at') or '未确定'}] {e['content']}")
        if append(line):
            selected_events.append(e['id'])
    return {'content': '\n'.join(lines), 'event_ids': selected_events,
            'intent_ids': selected_intents}


def validate_events(payload, messages):
    users = {m['id']: m for m in messages if m['role'] == 'user'}
    result = []
    for e in payload.get('events', [])[:12]:
        ids = list(dict.fromkeys(e.get('source_ids') or []))
        if not ids or any(x not in users for x in ids):
            continue
        content = str(e.get('summary') or '').strip()[:500]
        if not content or e.get('subject') != 'user':
            continue
        result.append({'content': content, 'subject': 'user',
            'occurred_at': str(e.get('occurred_at') or '')[:100], 'source_ids': ids,
            'start_turn': min(users[x]['turn_no'] for x in ids),
            'end_turn': max(users[x]['turn_no'] for x in ids)})
    return result


def valid_intent(operation, event_ids, existing_ids):
    if operation.get('action') not in {'create', 'update', 'link'}:
        return False
    if operation['action'] != 'create' and operation.get('intent_id') not in existing_ids:
        return False
    support = operation.get('support_event_ids') or []
    return (bool(str(operation.get('label') or '').strip()) and bool(support)
            and all(x in event_ids for x in support)
            and operation.get('status', 'tentative') in {'tentative', 'deferred', 'resolved'})
