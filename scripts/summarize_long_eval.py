"""Compact diagnostics for the synthetic long-dialogue run."""

import json
import os

import psycopg
from psycopg.rows import dict_row

record = json.load(open(os.getenv("ROLEPLAY_OUTPUT", "/checkpoints/long-dialogue-eval.json"), encoding="utf-8"))
print("user:", record["user_id"])
for number, session in enumerate(record["sessions"], 1):
    print("session:", number, session["session_id"])
    for turn in session["turns"]:
        print("turn:", turn["turn"], "route:", turn["route"],
              "visitor:", turn["visitor"][:95],
              "reply:", turn["counselor"][:150],
              "batch:", turn.get("memory_job_status"),
              "written:", turn.get("memory_job_result"))
with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
    rows = db.execute("""SELECT id,layer,kind,content,source_type,source_id,
        reinforcement_count,retrieval_count,metadata FROM memories WHERE user_id=%s
        AND status='active' ORDER BY created_at""", (record["user_id"],)).fetchall()
    for row in rows:
        print("memory:", str(row["id"]), row["layer"], row["kind"],
              row["content"][:180], "source:", row["source_type"], row["source_id"],
              "reinforcement:", row["reinforcement_count"],
              "retrieval:", row["retrieval_count"],
              "meta:", row["metadata"])
    intents = db.execute("""SELECT label,status,support_count,retrieval_count FROM intents
        WHERE user_id=%s""", (record["user_id"],)).fetchall()
    for item in intents:
        print("intent:", item)
