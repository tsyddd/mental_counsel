"""Run two synthetic 30-turn counselor dialogues through the live gateway."""

import http.cookiejar
import json
import os
import secrets
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


BASE = os.getenv("INTERVIEW_BASE_URL", "http://127.0.0.1:18081")
OUT = Path(__file__).resolve().parents[1] / "data" / "live_30turn_cases.json"
OUT.parent.mkdir(exist_ok=True)

CASES = {
    "sports_and_research": [
        "我叫小明，我喜欢体育课，最近科研进展有点慢。",
        "我每周三都会去打篮球，那段时间心情会好一些。",
        "这周实验结果没达到预期，我有点沮丧。",
        "导师问起进度时，我不知道怎么解释。",
        "我其实已经排查过数据预处理，但还没找到原因。",
        "今天又试了一组参数，结果仍然不稳定。",
        "我担心别人觉得我不够努力。",
        "上周有个同学建议我把问题拆小一点。",
        "我想先确认是不是数据标注出了问题。",
        "你觉得我可以从哪一步开始整理？",
        "我试着列出三个最可能的原因。",
        "其中一个原因和训练集分布有关。",
        "我和同学讨论后发现验证集也需要检查。",
        "今天我只完成了一个小检查，但至少有进展。",
        "我希望明天能把排查记录发给导师。",
        "发消息之前我还是会紧张。",
        "我怕自己写得太乱，让老师看不懂。",
        "如果按问题、证据、下一步来写，可能会清楚一点。",
        "我准备先写一段简短的说明。",
        "写完后我感觉比一直拖着轻松些。",
        "这几天运动少了，晚上也不太容易放松。",
        "我想把每周三打篮球的时间保留下来。",
        "今天和室友一起去球场走了走。",
        "运动后我能更平静地看实验日志。",
        "我发现有两次失败其实是同一类错误。",
        "我想把这类错误单独做一个记录。",
        "现在最重要的是确认数据问题，而不是再盲目改模型。",
        "我明天会先核对标注，再决定下一步。",
        "和你聊到这里，我觉得思路比刚开始清楚了。",
        "帮我总结一下这段对话里我已经确认的下一步。",
    ],
    "drawing_and_class_pressure": [
        "我叫小雨，我喜欢画画，但最近课堂发言让我紧张。",
        "上周老师问我一个问题，我知道答案却没有举手。",
        "我担心说错以后同学会笑我。",
        "下课后朋友说她其实也会紧张。",
        "知道不是我一个人这样，我稍微放松了一点。",
        "我在画画的时候比较能专注。",
        "昨天我画了一张操场的速写。",
        "同桌看到了，说颜色搭配很好。",
        "我有点开心，但又觉得那只是客气话。",
        "我不太习惯别人夸奖我。",
        "这周课堂上又有一次发言机会。",
        "我本来准备举手，最后还是犹豫了。",
        "我觉得自己可能需要先从简单的问题试起。",
        "我愿意先在小组里讲一遍想法。",
        "今天小组讨论时我说了两句话。",
        "虽然声音有点小，但大家认真听了。",
        "我回来以后把这件事记在本子上。",
        "我希望下次能说得再完整一点。",
        "我喜欢你一次只问我一个问题，这样比较容易想清楚。",
        "如果我沉默一会儿，希望你给我一点时间。",
        "上次小组发言时我最担心的仍然是说错。",
        "不过这次我发现同学并没有笑我。",
        "老师还追问了一句，我试着回答了。",
        "我觉得进步很小，但它确实发生了。",
        "我想下周再尝试一次。",
        "画画的时候我会给自己留一个安静的角落。",
        "有时我会把当天发生的事画成小图。",
        "这样回头看，我能发现自己没那么糟。",
        "我想把小组发言和画画都当成自己的练习。",
        "你能帮我回顾一下我这段时间有什么变化吗？",
    ],
}


def new_client():
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def request(opener, path, body=None):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=raw,
                                 headers={"Content-Type": "application/json"} if raw else {})
    with opener.open(req, timeout=240) as response:
        return response.read().decode("utf-8")


def chat(opener, session, message):
    stream = request(opener, f"/api/sessions/{session}/messages", {"text": message})
    answer, done = "", {}
    for block in stream.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" not in lines:
            continue
        payload = json.loads(lines["data"])
        if lines.get("event") == "token":
            answer += payload.get("text", "")
        elif lines.get("event") == "done":
            done = payload
    return {"user": message, "assistant": answer, "route": done.get("memory_route"),
            "memory_job_id": done.get("memory_job_id")}


result = {"cases": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
for case_name, turns in CASES.items():
    opener = new_client()
    account = json.loads(request(opener, "/api/auth/register", {
        "email": f"live30-{uuid.uuid4().hex[:12]}@example.test",
        "password": secrets.token_urlsafe(20), "display_name": "合成对话案例"}))
    session = json.loads(request(opener, "/api/sessions", {"mode": "counselor"}))["session_id"]
    record = {"user_id": account["user_id"], "session_id": session, "turns": [], "errors": []}
    result["cases"][case_name] = record
    for index, message in enumerate(turns, 1):
        try:
            record["turns"].append(chat(opener, session, message))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            record["errors"].append({"turn": index, "error": str(exc)})
            break
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{case_name}: {index}/30 route={record['turns'][-1]['route']}", flush=True)
    if len(record["turns"]) == 30:
        # A fresh session probes whether the prior dialogue became durable memory.
        next_session = json.loads(request(opener, "/api/sessions", {"mode": "counselor"}))["session_id"]
        question = "你知道我喜欢什么吗？我们这段时间反复谈到的事情是什么？"
        record["cross_session_probe"] = chat(opener, next_session, question)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
