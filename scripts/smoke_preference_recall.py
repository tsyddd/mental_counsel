"""Exercise a cross-session explicit-preference recall with a fresh account."""

import http.cookiejar
import json
import secrets
import urllib.request
import uuid


BASE = "http://127.0.0.1:18081"
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def request(path, body=None):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=raw,
                                 headers={"Content-Type": "application/json"} if raw else {})
    with opener.open(req, timeout=180) as response:
        return response.read().decode("utf-8")


def chat(session, text):
    stream = request(f"/api/sessions/{session}/messages", {"text": text})
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
    return answer, done


email = f"preference-smoke-{uuid.uuid4().hex[:12]}@example.test"
account = json.loads(request("/api/auth/register", {
    "email": email, "password": secrets.token_urlsafe(20), "display_name": "记忆回归案例"}))
first = json.loads(request("/api/sessions", {"mode": "counselor"}))["session_id"]
reply1, route1 = chat(first, "我叫小明，我喜欢体育课，我最近为科研的事情烦心。")
second = json.loads(request("/api/sessions", {"mode": "counselor"}))["session_id"]
reply2, route2 = chat(second, "你知道我喜欢什么课吗？")
graph = json.loads(request(f"/api/users/{account['user_id']}/memory-graph"))
preferences = [item["content"] for item in graph.get("memories", [])
               if item.get("kind") == "preference"]
print(json.dumps({"first_route": route1.get("memory_route"),
                  "second_route": route2.get("memory_route"),
                  "stored_preferences": preferences,
                  "second_reply": reply2,
                  "recalled": "体育课" in reply2}, ensure_ascii=False))
