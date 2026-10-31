"""Synthetic DeepSeek visitor/counselor evaluation over two sessions, one user."""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from relation_videoarm.api import APIClient


GATEWAY = os.getenv("ROLEPLAY_GATEWAY_URL", "http://gateway:8080")
MEDIA = os.getenv("ROLEPLAY_MEDIA_API_URL", "http://media-api:8000")
USER_ID = os.getenv("ROLEPLAY_USER_ID", "student-synthetic-long-" + uuid.uuid4().hex[:10])
VIDEO_ID = os.getenv("ROLEPLAY_VIDEO_ID", "interview-demo")
OUT = Path(os.getenv("ROLEPLAY_OUTPUT", "/checkpoints/long-dialogue-eval.json"))

SCENARIOS = [
    [
        "上周数学测验没考好，担心妈妈失望而把试卷藏了两天；后来告诉妈妈，她陪我看错题。讲一个具体经历。",
        "今天一道物理题不会，我担心问老师显得笨，拖到下课才问；得到解释后松了口气。讲具体经过。",
        "向咨询师明确提出：我希望你一次只问我一个问题，先听我说完。这是交流方式偏好。",
        "小组讨论时想发言却怕说错，最后没有开口，回家有点遗憾。讲一次具体经历。",
        "明确说出我希望逐渐敢于向同学和老师提问，把这作为接下来想尝试的目标。",
        "昨天和朋友聊了一会儿周末作业，我们一起列了计划，心情轻松些。讲一个新经历。",
        "今天在课堂上还是不敢举手，但课后把问题写在纸上交给了老师。讲新进展。",
        "询问咨询师：你记得我上次小组讨论没发言的事吗？那件事和现在有什么关联？",
        "询问咨询师：我之前想逐渐敢于提问，这个目标现在有什么进展？",
        "简单说一句谢谢，不提出历史问题。",
    ],
    [
        "简单向咨询师打个招呼，说你好。",
        "跨会话追问：你记得上次小组讨论我没有发言，后来给老师递纸条的事吗？",
        "跨会话询问：你还记得我希望你一次只问一个问题吗？",
        "这周我主动问了一个同学作业怎么做，对方解释后我觉得没有想象中尴尬。讲具体新经历。",
        "请咨询师结合以前的目标和今天主动问同学这件事，谈谈我有哪些变化。",
    ],
]


def wait_job(job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{MEDIA}/jobs/{job_id}", timeout=15)
        response.raise_for_status()
        row = response.json()
        if row["status"] in {"completed", "failed"}:
            return row
        time.sleep(2)
    return {"status": "timeout", "job_id": job_id}


def create_session():
    response = requests.post(f"{GATEWAY}/api/sessions", json={
        "user_id": USER_ID, "video_id": VIDEO_ID, "mode": "counselor"}, timeout=30)
    response.raise_for_status()
    return response.json()["session_id"]


def wait_profile(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = requests.get(f"{MEDIA}/users/{USER_ID}/memory-graph", timeout=30)
        response.raise_for_status()
        if any(item["layer"] == "user" for item in response.json()["memories"]):
            return
        time.sleep(3)
    raise TimeoutError("initial video profile did not appear")


def visitor_turn(client, scenario, history):
    if "简单" in scenario and "招呼" in scenario:
        return "你好。"
    if "简单说一句谢谢" in scenario:
        return "谢谢。"
    response = client.chat([
        {"role": "system", "content": "你扮演合成演示中的一名虚构中学生来访者。只把给定情景改写成自然的第一人称发言。"
         "保持情景中的事实、事件、偏好或问题，不添加诊断或危险经历。只输出一句或两句中文发言。"},
        {"role": "user", "content": json.dumps({"scenario": scenario,
            "recent_roleplay_turns": history[-2:]}, ensure_ascii=False)},
    ], purpose="synthetic_visitor_long_dialogue", max_tokens=500)
    value = response.get("content") or ""
    if isinstance(value, list):
        value = "".join(item.get("text", "") for item in value if isinstance(item, dict))
    return str(value).strip()


def chat(session_id, text):
    response = requests.post(f"{GATEWAY}/api/sessions/{session_id}/messages",
        json={"text": text}, stream=True,
        timeout=(20, 420))
    response.raise_for_status()
    response.encoding = "utf-8"
    event = ""
    chunks = []
    done = {}
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            event = ""
            continue
        if raw.startswith("event: "):
            event = raw[7:]
        elif raw.startswith("data: "):
            data = json.loads(raw[6:])
            if event == "token":
                chunks.append(data.get("text", ""))
            elif event == "done":
                done = data
    return "".join(chunks), done


def main():
    requests.get(f"{MEDIA}/health", timeout=20).raise_for_status()
    client = APIClient()
    record = {"synthetic": True, "user_id": USER_ID, "video_id": VIDEO_ID,
              "model": client.model, "started_at": datetime.now(timezone.utc).isoformat(),
              "sessions": [], "findings": []}
    for index, scenarios in enumerate(SCENARIOS, 1):
        session_id = create_session()
        if index == 1:
            wait_profile()
        session_record = {"session_id": session_id, "turns": []}
        record["sessions"].append(session_record)
        history = []
        for number, scenario in enumerate(scenarios, 1):
            text = visitor_turn(client, scenario, history)
            if not text:
                raise RuntimeError(f"visitor generation empty at session {index} turn {number}")
            reply, done = chat(session_id, text)
            if not reply or not done:
                raise RuntimeError(f"counselor/SSE incomplete at session {index} turn {number}")
            row = {"turn": number, "scenario": scenario, "visitor": text,
                   "counselor": reply, "route": done.get("memory_route"),
                   "rewritten_queries": done.get("queries", []),
                   "memory_job_id": done.get("memory_job_id", "")}
            if done.get("memory_job_id"):
                job = wait_job(done["memory_job_id"])
                row["memory_job_status"] = job["status"]
                row["memory_job_result"] = job.get("result")
                if job["status"] != "completed":
                    record["findings"].append(f"session {index} turn {number}: batch {job['status']}")
            session_record["turns"].append(row)
            history.append({"visitor": text, "counselor": reply})
            OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"session": index, "turn": number, "route": row["route"],
                              "batch": row.get("memory_job_status")}, ensure_ascii=False), flush=True)
    graph = requests.get(f"{MEDIA}/users/{USER_ID}/memory-graph", timeout=30)
    graph.raise_for_status()
    memory = graph.json()
    record["summary"] = {
        "sessions": len(record["sessions"]),
        "turns": sum(len(item["turns"]) for item in record["sessions"]),
        "memory_layers": {layer: sum(1 for item in memory["memories"] if item["layer"] == layer)
                          for layer in ["user", "episodic", "session", "procedural", "failure"]},
        "concepts": len(memory["concepts"]), "intents": memory["intents"],
        "frontend_ok": "查看历史会话" in requests.get(GATEWAY, timeout=20).text,
        "backend_sessions": len(requests.get(
            f"{GATEWAY}/api/users/{USER_ID}/sessions", timeout=20).json()["sessions"]),
    }
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": record["summary"]},
                     ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
