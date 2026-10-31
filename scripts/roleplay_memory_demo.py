"""Run a clearly synthetic, two-agent memory demo against the local Compose stack.

Run inside the worker container so both model roles use the configured DeepSeek
endpoint without printing or copying credentials. No real student utterances are
used; the new user ID keeps this roleplay separate from the existing demo user.
"""

from __future__ import annotations

import json
import os
import time

import psycopg
import requests
from psycopg.rows import dict_row

from relation_videoarm.api import APIClient


GATEWAY = os.getenv("ROLEPLAY_GATEWAY_URL", "http://gateway:8080")
MEDIA_API = os.getenv("ROLEPLAY_MEDIA_API_URL", "http://media-api:8000")
USER_ID = os.getenv("ROLEPLAY_USER_ID", "student-mock-roleplay-final-20261021")
VIDEO_ID = os.getenv("ROLEPLAY_VIDEO_ID", "interview-demo")
DB = os.environ["DATABASE_URL"]

SCENARIOS = [
    "上周数学测验没考好，我担心爸妈失望，就把成绩藏了两天，晚上自己熬夜补题，第二天上课很困。后来我还是告诉了妈妈，她陪我挑了几道错题一起看，我轻松了一些。请用自然的第一人称说出来。",
    "这周小组展示前，我怕讲错被同学笑，所以一直拖着不练。后来我告诉组员自己很紧张，对方陪我彩排了一遍，我感觉没那么害怕了。说成一个新的具体经历，不重复上一个事件。",
    "今天物理作业有一道题我不懂，我担心问老师会显得很笨，就一直拖到下课后才去问。老师解释以后我松了口气。说成一个新的具体经历，不重复前两个事件。",
]

CLIENT_SYSTEM = (
    "你正在参与一个明确标记为合成的心理访谈产品演示。请扮演虚构的中学生来访者，"
    "只根据每次给你的情景提示，用第一人称说 2 到 3 句完整自然中文，至少 60 个汉字，必须以完整句号结束。"
    "人物设定是：遇到学业压力时有时会迟疑求助，"
    "但在得到支持后会感到轻松。不要声称是真实学生，不要引用或编造真实身份信息；"
    "不做诊断，不提自伤、虐待、创伤或危险行为。每轮说一个具体情景，包含情境、做法和结果/感受。"
)


def db_conn():
    return psycopg.connect(DB, row_factory=dict_row)


def wait_for_profile(timeout: int = 150) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with db_conn() as db:
            seed_job = db.execute(
                "SELECT status,error FROM jobs WHERE kind='seed_profile' "
                "AND payload->>'user_id'=%s ORDER BY created_at DESC LIMIT 1",
                (USER_ID,),
            ).fetchone()
            count = db.execute(
                "SELECT count(*) AS n FROM memories WHERE user_id=%s AND layer='user' AND status='active'",
                (USER_ID,),
            ).fetchone()["n"]
        if seed_job and seed_job["status"] == "completed":
            return count
        if seed_job and seed_job["status"] == "failed":
            raise RuntimeError(f"profile seeding failed: {seed_job.get('error', 'unknown error')}")
        time.sleep(2)
    raise TimeoutError("video profile was not seeded in time")


def wait_for_job(job_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{MEDIA_API}/jobs/{job_id}", timeout=15)
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(1)
    raise TimeoutError(f"memory job {job_id} did not finish in time")


def counselor_reply(session_id: str, user_text: str) -> tuple[str, str]:
    response = requests.post(
        f"{GATEWAY}/api/sessions/{session_id}/messages",
        json={"text": "【合成角色扮演·来访者】" + user_text},
        stream=True,
        timeout=(15, 300),
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    current_event = ""
    answer_parts: list[str] = []
    memory_job_id = ""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            current_event = ""
            continue
        if raw.startswith("event: "):
            current_event = raw[7:]
        elif raw.startswith("data: "):
            data = raw[6:]
            if current_event == "token":
                try:
                    answer_parts.append(json.loads(data).get("text", ""))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid token SSE payload at line {exc.lineno}, column {exc.colno}: {data[:120]!r}"
                    ) from exc
            elif current_event == "done":
                memory_job_id = json.loads(data).get("memory_job_id", "")
    if not memory_job_id:
        raise RuntimeError("chat stream ended without a memory job ID")
    return "".join(answer_parts), memory_job_id


def main() -> None:
    session_response = requests.post(
        f"{GATEWAY}/api/sessions",
        json={"user_id": USER_ID, "video_id": VIDEO_ID, "mode": "counselor"},
        timeout=30,
    )
    session_response.raise_for_status()
    session_id = session_response.json()["session_id"]
    profile_count = wait_for_profile()

    client = APIClient()
    client_history = [{"role": "system", "content": CLIENT_SYSTEM}]
    rounds = []
    for scenario in SCENARIOS:
        client_history.append({"role": "user", "content": scenario})
        generated = {}
        user_text = ""
        for attempt in range(3):
            if attempt:
                client_history.append({
                    "role": "user",
                    "content": "请完整重述这段来访者发言：至少 60 个汉字，写完情境、做法和结果/感受，最后用句号收尾。"
                    "只输出第一人称普通文本，不要输出 JSON 或思维过程。",
                })
            generated = client.chat(client_history, purpose="synthetic_client_roleplay", max_tokens=500)
            content = generated.get("content")
            if isinstance(content, str):
                user_text = content.strip()
            elif isinstance(content, list):
                user_text = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
            if user_text.endswith(("。", "！", "？")) and len(user_text) >= 50:
                break
            user_text = ""
        if not user_text:
            raise RuntimeError(f"synthetic client returned no complete turn; fields={sorted(generated.keys())}")
        client_history.append({"role": "assistant", "content": user_text})

        answer, job_id = counselor_reply(session_id, user_text)
        job = wait_for_job(job_id)
        if job["status"] != "completed":
            raise RuntimeError(f"memory extraction failed: {job.get('error', 'unknown error')}")
        client_history.append({"role": "user", "content": "咨询师刚才的回应是：" + answer})
        rounds.append({"client": user_text, "counselor": answer, "memory_job": job["status"]})

    with db_conn() as db:
        layers = db.execute(
            "SELECT layer,count(*) AS n FROM memories WHERE user_id=%s AND status='active' "
            "GROUP BY layer ORDER BY layer",
            (USER_ID,),
        ).fetchall()
        concepts = db.execute(
            "SELECT count(*) AS n FROM concepts WHERE user_id=%s", (USER_ID,)
        ).fetchone()["n"]
        intents = db.execute(
            "SELECT label,status,support_count FROM intents WHERE user_id=%s ORDER BY updated_at DESC",
            (USER_ID,),
        ).fetchall()
        episodic = db.execute(
            "SELECT count(*) AS n FROM memories WHERE user_id=%s AND layer='episodic' AND status='active'",
            (USER_ID,),
        ).fetchone()["n"]

    print(json.dumps({
        "synthetic": True,
        "model": client.model,
        "user_id": USER_ID,
        "session_id": session_id,
        "video_profile_seeded": profile_count,
        "rounds": rounds,
        "active_memory_counts_by_layer": {row["layer"]: row["n"] for row in layers},
        "episodic_memories": episodic,
        "concepts": concepts,
        "intents": intents,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
