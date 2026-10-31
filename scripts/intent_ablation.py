"""CogEval-inspired, synthetic 50-case intent ablation for this platform.

Gold follow-up needs are defined before GPT paraphrasing. Both arms use
gpt-6-sol. The full arm invokes the deployed five-turn memory extractor and
event->concept->intent fold; the ablation sees only the last six user turns,
matching the gateway's 12-message recent-history window.
This is a project-specific diagnostic set, not the official CogEval-Bench.
"""
import argparse
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from relation_videoarm.api import APIClient, parse_json_object
from platform_app.core import conn
from platform_app.worker import extract_memory

OUTPUT = Path(os.getenv("INTENT_EVAL_OUTPUT", "/checkpoints/intent-ablation-50.json"))
LABELS = {
    "ask_help": "逐步练习向老师或同学主动求助",
    "parent_talk": "尝试向家长表达学习压力并寻求支持",
    "sleep_routine": "建立更稳定的睡前与起床安排",
    "peer_join": "循序参与同伴活动和小组交流",
    "exam_plan": "把考前复习拆成可执行的小步骤",
}

# Each positive schema contains three distinct episodes supporting one need.
# The seven remaining turns are distractors, including cross-topic details.
POSITIVE = {
    "ask_help": [
        "周一数学题卡住，我想问老师却一直没开口，回家才自己查资料。",
        "小组作业有一步看不懂，我怕拖累同学，先装作明白。",
        "后来我把一道题写在纸上交给老师，发现老师愿意解释。",
    ],
    "parent_talk": [
        "上次测验没考好，我把试卷放进书包，不敢和妈妈说。",
        "周末又遇到作业压力，我想告诉家里，却只说了学校挺好。",
        "昨天我试着说最近学习有点吃力，妈妈听完愿意一起看错题。",
    ],
    "sleep_routine": [
        "上周写完作业还刷手机，第二天早上上课很困。",
        "隔了几天又睡得很晚，早读时注意力很难集中。",
        "昨晚我把手机提前放远一点，今天起床比前几天轻松。",
    ],
    "peer_join": [
        "班上讨论活动我有想法，但怕说错，一直没加入。",
        "社团分组时我又站在旁边，后来有点遗憾没试着参与。",
        "今天我主动问同桌能不能一起做海报，她说可以。",
    ],
    "exam_plan": [
        "上次复习我一直不知道从哪一章开始，最后只翻了目录。",
        "这周又面对一堆练习，我想一次做完，结果半途停下。",
        "昨天我先列出三道最容易错的题，做完后感觉更能继续。",
    ],
}
NEGATIVE = {
    "unrelated": [
        "今天路上看到了新的花店。", "周末家里吃了面条。", "学校操场换了一面旗。"],
    "resolved": [
        "上周有道题不会，我已经问过老师。", "老师讲完后我能独立做同类题。", "今天复查也做对了，目前没有需要再跟进的地方。"],
    "declined": [
        "我之前试过参加社团活动。", "活动结束后我觉得体验已经足够。", "现在我明确不想继续讨论参加社团的事。"],
    "third_party": [
        "同桌说她最近不敢问老师问题。", "她后来把题写在纸上。", "她说老师给她讲明白了；这些是她的经历，不是我的。"],
    "routine": [
        "周一我照常坐公交上学。", "周三仍然坐同一班公交。", "今天也按原来的时间出门，一切顺利。"],
}
DISTRACTORS = [
    "今天午饭有番茄炒蛋。", "我在图书馆借了一本科普书。",
    "放学路上看到新开的文具店。", "体育课练了跳绳。",
    "周末和家人一起整理了书架。", "学校通知下周有运动会。",
    "我最近在看一本关于星空的书。",
]
POSITIONS = (0, 3, 7)


def seed_cases():
    cases = []
    for label, episodes in list(POSITIVE.items()) + list(NEGATIVE.items()):
        for variant in range(5):
            turns = []
            for position in range(10):
                if position in POSITIONS:
                    text = episodes[POSITIONS.index(position)]
                else:
                    text = DISTRACTORS[(position + variant * 2) % len(DISTRACTORS)]
                turns.append(text)
            cases.append({"id": f"{label}-{variant+1}", "gold": label if label in POSITIVE else None,
                          "gold_evidence_turns": [x+1 for x in POSITIONS] if label in POSITIVE else [],
                          "draft_turns": turns})
    return cases


def model_text(message):
    value = message.get("content", "")
    if isinstance(value, list):
        return "".join(item.get("text", "") for item in value if isinstance(item, dict))
    return str(value)


def paraphrase(client, case):
    prompt = ("将下列10条虚构中学生第一人称话语改写成自然中文，保持每条事实、事件主体、"
              "已解决/拒绝状态与顺序；不要新增目标、心理诊断或意图。"
              "返回JSON对象 {turns:[恰好10个字符串]}。\n"
              + json.dumps(case["draft_turns"], ensure_ascii=False))
    try:
        data = parse_json_object(model_text(client.chat([
            {"role":"system","content":"你只做忠实改写，严格输出JSON。"},
            {"role":"user","content":prompt}], purpose="intent_case_generation", max_tokens=1700)))
        turns = data.get("turns")
        if isinstance(turns, list) and len(turns) == 10 and all(isinstance(x,str) and x.strip() for x in turns):
            return turns, True
    except Exception:
        pass
    return case["draft_turns"], False


def store_batch(user, turns, batch_no):
    sid, first, last = uuid.uuid4(), None, None
    start = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=batch_no * 10)
    with conn() as db:
        db.execute("INSERT INTO sessions(id,user_id,agent_mode) VALUES (%s,%s,'counselor')", (sid,user))
        for i, turn in enumerate(turns):
            mid = uuid.uuid4()
            if first is None: first = mid
            db.execute("""INSERT INTO messages(id,session_id,role,content,created_at)
                VALUES (%s,%s,'user',%s,%s)""", (mid,sid,turn,start+timedelta(minutes=i*2)))
            last = uuid.uuid4()
            db.execute("""INSERT INTO messages(id,session_id,role,content,created_at)
                VALUES (%s,%s,'assistant','我在听，请继续。',%s)""",
                (last,sid,start+timedelta(minutes=i*2+1)))
    extract_memory({"job_id":str(uuid.uuid4()),"user_id":user,"session_id":str(sid),"batch_no":batch_no,
                    "first_message_id":str(first),"last_message_id":str(last)})


def graph_prediction(user):
    with conn() as db:
        intents = db.execute("""SELECT label,support_count FROM intents
            WHERE user_id=%s AND status='tentative' ORDER BY support_count DESC,updated_at DESC""",
            (user,)).fetchall()
        concepts = db.execute("SELECT count(*) AS n FROM concepts WHERE user_id=%s", (user,)).fetchone()["n"]
        events = db.execute("""SELECT count(*) AS n FROM memories WHERE user_id=%s
            AND layer='episodic' AND status='active'""", (user,)).fetchone()["n"]
    if not intents:
        return {"has_intent":False,"label":None,"raw_intents":[],"events":events,"concepts":concepts}
    raw = [dict(x) for x in intents]
    return {"has_intent":True,"label":None,"raw_intents":raw,"events":events,"concepts":concepts}


def classify_label(client, text):
    if not text: return None
    prompt=("将下列待确认意图映射到最相近的一个预定义方向；若没有对应项，返回null。"
            "只依据意图文字，不要添加事实。返回JSON {label:string|null}。\n"
            "方向: "+json.dumps(LABELS,ensure_ascii=False)+"\n意图: "+text)
    data=parse_json_object(model_text(client.chat([
        {"role":"system","content":"你是标签映射器，只输出JSON。"},
        {"role":"user","content":prompt}],purpose="intent_label_mapping",max_tokens=150)))
    return data.get("label") if data.get("label") in LABELS else None


def baseline_prediction(client, turns):
    prompt = ("你只能看最近6条用户发言，判断是否有由多个不同情景支持、尚待确认的跟进意图。"
              "一次性提及、已解决、明确拒绝和第三方经历都不算。证据不足返回false。"
              "若有，从选项中选一个；不要猜测诊断。返回JSON {has_intent:boolean,label:string|null}。"
              "选项: " + json.dumps(LABELS, ensure_ascii=False) + "\n最近发言: "
              + json.dumps(turns[-6:], ensure_ascii=False))
    data = parse_json_object(model_text(client.chat([
        {"role":"system","content":"你是严格的意图识别器，只输出JSON。"},
        {"role":"user","content":prompt}], purpose="intent_recent_history_baseline", max_tokens=250)))
    label = data.get("label") if data.get("label") in LABELS else None
    has_intent = data.get("has_intent") is True and label is not None
    return {"has_intent":has_intent,"label":label}


def metrics(cases, key):
    rows = [x for x in cases if key in x]
    tp = sum(x["gold"] is not None and x[key]["has_intent"] for x in rows)
    tn = sum(x["gold"] is None and not x[key]["has_intent"] for x in rows)
    fp = sum(x["gold"] is None and x[key]["has_intent"] for x in rows)
    fn = sum(x["gold"] is not None and not x[key]["has_intent"] for x in rows)
    exact = sum((x["gold"] == x[key]["label"]) for x in rows)
    precision = tp / (tp+fp) if tp+fp else 0
    recall = tp / (tp+fn) if tp+fn else 0
    return {"n":len(rows),"binary_accuracy":(tp+tn)/len(rows) if rows else 0,
            "label_accuracy":exact/len(rows) if rows else 0,"tp":tp,"tn":tn,"fp":fp,"fn":fn,
            "precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    client = APIClient()
    if client.model != "gpt-6-sol": raise RuntimeError("gpt-6-sol is required for both arms")
    cases = seed_cases()
    if OUTPUT.exists():
        saved=json.loads(OUTPUT.read_text(encoding="utf-8"))
        previous={item["id"]:item for item in saved.get("cases",[])}
        cases=[previous.get(item["id"],item) for item in cases]
    for index,case in enumerate(cases[args.start:args.start+args.limit],args.start+1):
        if "full" in case and "baseline" in case: continue
        if "turns" not in case:
            case["turns"],case["gpt_paraphrase_valid"]=paraphrase(client,case)
        user="intent-eval-"+case["id"]+"-"+uuid.uuid4().hex[:8]
        store_batch(user,case["turns"][:5],1)
        store_batch(user,case["turns"][5:],2)
        full=graph_prediction(user)
        if full["has_intent"]:
            full["label"]=classify_label(client,full["raw_intents"][0]["label"])
        case["full"]=full
        case["baseline"]=baseline_prediction(client,case["turns"])
        result={"source":"CogEval-inspired synthetic counseling set; not official CogEval-Bench",
                "generator_model":"gpt-6-sol","evaluation_model":"gpt-6-sol",
                "n_planned":50,"n_completed":sum("full" in x and "baseline" in x for x in cases),
                "full":metrics(cases,"full"),"baseline":metrics(cases,"baseline"),"cases":cases}
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
        print(json.dumps({"case":index,"id":case["id"],"gold":case["gold"],
                          "full":full["label"],"baseline":case["baseline"]["label"],
                          "done":result["n_completed"]},ensure_ascii=False),flush=True)
    print(json.dumps({"full":metrics(cases,"full"),"baseline":metrics(cases,"baseline")},ensure_ascii=False))


if __name__ == "__main__": main()

