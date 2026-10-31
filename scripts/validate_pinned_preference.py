"""Check that a confirmed speaking preference is available without search."""

import json
from platform_app.core import conn
from platform_app.memory_retrieval import route_turn

record = json.load(open("/checkpoints/long-dialogue-eval-v2.json", encoding="utf-8"))
user_id = record["user_id"]
with conn() as db:
    rows = db.execute("""SELECT content FROM memories WHERE user_id=%s
        AND kind='communication_preference' AND layer='user' AND status='active'
        AND (expires_at IS NULL OR expires_at>now()) ORDER BY updated_at DESC LIMIT 3""",
        (user_id,)).fetchall()
route = route_turn("你好。", [])
print(json.dumps({"greeting_route": route["route"],
                  "persistent_preferences": [row["content"] for row in rows]},
                 ensure_ascii=False))
