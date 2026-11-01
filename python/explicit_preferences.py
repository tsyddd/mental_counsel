"""Capture short, explicit self-reported preferences on the current turn."""

import re

CLAUSE_END = re.compile(r"[，,。；;！？!?\n]|(?:但是|不过|可是|然后|另外|我最近)")
COMMUNICATION = re.compile(r"我(?:更)?(?:希望|喜欢)你(?P<value>[^，,。；;！？!?\n]{1,40})")
COMMUNICATION_CUES = ("问", "答", "回复", "说", "称呼", "解释", "时间", "节奏", "语气", "简短", "慢", "安静", "暂停", "别急", "温柔")
LIKES = (
    re.compile(r"我最喜欢的(?:课程|科目|学科)是(?P<value>[^，,。；;！？!?\n]{1,28})"),
    re.compile(r"我喜欢的(?:课程|科目|学科)是(?P<value>[^，,。；;！？!?\n]{1,28})"),
    re.compile(r"我(?:最|很|非常|特别|比较|更)?喜欢(?P<value>[^，,。；;！？!?\n]{1,28})"),
)
DISLIKES = re.compile(r"我(?:很|非常|特别)?不喜欢(?P<value>[^，,。；;！？!?\n]{1,28})")


def extract_explicit_preferences(text):
    """Return grounded statements; do not infer a preference from context."""
    results = []
    covered = []
    for pattern, prefix in [(COMMUNICATION, ""), (DISLIKES, "我不喜欢"),
                            *((item, "我喜欢") for item in LIKES)]:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in covered):
                continue
            value = CLAUSE_END.split(match.group("value"), 1)[0].strip(" 、：: ")
            if (not value or len(value) > 18 or
                    value.startswith(("什么", "哪门", "哪种", "哪些")) or
                    value.endswith(("吗", "呢", "么"))):
                continue
            if pattern is COMMUNICATION and not any(cue in value for cue in COMMUNICATION_CUES):
                continue
            if prefix == "我喜欢" and text[max(0, match.start()-2):match.start()].endswith("不"):
                continue
            statement = (match.group(0)[:-len(match.group("value"))] + value
                         if pattern is COMMUNICATION else prefix + value)
            if statement not in results:
                results.append(statement)
                covered.append(match.span())
    return results[:3]


def store_explicit_preferences(user_id, session_id, message_id, text):
    from .worker import remember

    statements = extract_explicit_preferences(text)
    saved = []
    for statement in statements:
        kind = ("communication_preference" if COMMUNICATION.match(statement)
                else "preference")
        memory_id, created = remember(
            user_id, statement, kind, "chat", message_id, 0.98,
            {"session_id": session_id, "explicit": True}, layer="user")
        saved.append({"id": memory_id, "content": statement, "created": created})
    return saved
