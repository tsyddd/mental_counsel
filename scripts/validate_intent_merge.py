"""Exercise concept-level intent consolidation on the synthetic roleplay user."""

import json
from platform_app.core import conn
from platform_app.worker import fold_concept

record = json.load(open("/checkpoints/long-dialogue-eval-v2.json", encoding="utf-8"))
user_id = record["user_id"]
with conn() as db:
    row = db.execute("""SELECT id,label,memory_ids FROM concepts
        WHERE user_id=%s AND support_count>=3 ORDER BY support_count DESC LIMIT 1""",
        (user_id,)).fetchone()
    assert row and row["memory_ids"]
    before = db.execute("""SELECT status,count(*) AS n FROM intents
        WHERE user_id=%s AND %s::uuid=ANY(concept_ids) GROUP BY status""",
        (user_id, row["id"])).fetchall()

fold_concept(user_id, row["label"], str(row["memory_ids"][0]),
             "继续关注低压力情境中的表达体验", row["label"])

with conn() as db:
    after = db.execute("""SELECT status,count(*) AS n FROM intents
        WHERE user_id=%s AND %s::uuid=ANY(concept_ids) GROUP BY status""",
        (user_id, row["id"])).fetchall()
print(json.dumps({"before": [dict(item) for item in before],
                  "after": [dict(item) for item in after]}, ensure_ascii=False))
