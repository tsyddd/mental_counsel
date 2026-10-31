"""Generate synthetic visitors with gpt-6-sol and exercise the deployed platform."""
import json
import os
import time
import uuid
from pathlib import Path
import requests
from relation_videoarm.api import APIClient

GATEWAY = os.getenv("EVAL_GATEWAY_URL","http://gateway:8080")
MEDIA = os.getenv("EVAL_MEDIA_URL","http://media-api:8000")
OUTPUT = Path(os.getenv("EVAL_OUTPUT","/checkpoints/gpt6-sol-cross-session-eval.json"))
SCENARIOS = [
    [
        "作为虚构中学生，说上周数学测验没考好，怕妈妈失望，藏试卷两天，后来告诉妈妈一起看错题。",
        "明确告诉咨询师：希望你一次只问一个问题，先听我把话说完。",
        "说小组讨论时想发言又怕说错，最终沉默，事后有点遗憾。",
        "说今天课堂上没有举手，但课后把问题写在纸上交给老师。",
        "明确说自己想逐渐练习向同学和老师提问，从压力较小的情景开始。",
    ],
    [
        "只说你好，不提旧事。",
        "问咨询师还记不记得小组讨论没发言，以及后来给老师写纸条这两件事。",
        "问咨询师还记不记得自己希望一次只问一个问题。",
        "说这周主动问同学一道作业题，对方耐心解释，没有想象中尴尬。",
        "问咨询师结合以前的目标和今天的新经历，自己有什么变化。",
    ],
]

def model_text(client, prompt, purpose):
    answer=client.chat([{"role":"system","content":"你扮演虚构的中学生来访者。只把指定情景改写为自然的第一人称中文发言，保留全部事实，不添加其他经历或诊断。仅输出一两句。"},
                        {"role":"user","content":prompt}],purpose=purpose,max_tokens=280)
    value=answer.get("content","")
    if isinstance(value,list):value="".join(x.get("text","") for x in value if isinstance(x,dict))
    return str(value).strip()

def chat(http, session, text):
    r=http.post(f"{GATEWAY}/api/sessions/{session}/messages",json={"text":text},stream=True,timeout=(20,240))
    r.raise_for_status()
    parts=[];done={};buffer=b""
    for chunk in r.iter_content(chunk_size=4096):
        buffer+=chunk.replace(b"\r\n",b"\n")
        while b"\n\n" in buffer:
            frame,buffer=buffer.split(b"\n\n",1)
            fields={}
            for line in frame.split(b"\n"):
                if b": " in line:
                    key,value=line.split(b": ",1)
                    fields[key]=value
            if b"data" not in fields:continue
            data=json.loads(fields[b"data"].decode("utf-8"))
            if fields.get(b"event")==b"token":parts.append(data.get("text",""))
            elif fields.get(b"event")==b"done":done=data
    return "".join(parts),done

def wait_job(job_id):
    for _ in range(90):
        r=requests.get(f"{MEDIA}/jobs/{job_id}",timeout=15);r.raise_for_status()
        if r.json()["status"] in {"completed","failed"}:return r.json()
        time.sleep(2)
    return {"status":"timeout"}

def main():
    client=APIClient()
    if client.model!="gpt-6-sol":raise RuntimeError("evaluation requires gpt-6-sol")
    http=requests.Session()
    suffix=uuid.uuid4().hex[:10]
    r=http.post(f"{GATEWAY}/api/auth/register",json={"email":f"eval-{suffix}@example.test",
        "display_name":"合成案例学生","password":uuid.uuid4().hex},timeout=30)
    r.raise_for_status()
    user=r.json()["user_id"]
    record={"synthetic":True,"model":client.model,"user_id":user,"sessions":[]}
    for ix,scenarios in enumerate(SCENARIOS,1):
        r=http.post(f"{GATEWAY}/api/sessions",json={"mode":"counselor"},timeout=30);r.raise_for_status()
        sid=r.json()["session_id"]
        session={"session_id":sid,"turns":[]}
        record["sessions"].append(session)
        for number,scenario in enumerate(scenarios,1):
            visitor="你好。" if ix==2 and number==1 else model_text(client,scenario,"eval_synthetic_visitor")
            reply,done=chat(http,sid,visitor)
            row={"turn":number,"scenario":scenario,"visitor":visitor,"counselor":reply,
                 "route":done.get("memory_route"),"queries":done.get("queries",[])}
            if not reply:row["error"]="empty_reply"
            if done.get("memory_job_id"):
                row["memory_job"]=wait_job(done["memory_job_id"])["status"]
            session["turns"].append(row)
            OUTPUT.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps({"session":ix,"turn":number,"route":row["route"],"memory_job":row.get("memory_job")},ensure_ascii=False),flush=True)
    graph=http.get(f"{GATEWAY}/api/users/{user}/memory-graph",timeout=30);graph.raise_for_status()
    trace=http.get(f"{GATEWAY}/api/sessions/{record['sessions'][-1]['session_id']}/trace",timeout=30);trace.raise_for_status()
    prefs=http.get(f"{GATEWAY}/api/preferences",timeout=30);prefs.raise_for_status()
    answers=[record["sessions"][1]["turns"][i]["counselor"] for i in (1,2,4)]
    judge=client.chat([{"role":"system","content":"你是严格评估者。基于给出的完整对话，只判定以下三个跨会话追问的回复是否正确引用了先前事实：1 小组讨论未发言和课后递纸条；2 一次只问一个问题；3 练习向同学老师提问的目标与新进展。返回 JSON {checks:[{name,pass,reason}],issues:[string]}，不得根据常识补证据。"},
        {"role":"user","content":json.dumps({"sessions":record["sessions"],"probe_answers":answers},ensure_ascii=False)}],
        purpose="cross_session_judge",max_tokens=1100)
    record["judge"]=judge.get("content")
    record["summary"]={"sessions":2,"turns":10,"events":sum(x["layer"]=="episodic" for x in graph.json()["memories"]),
        "concepts":len(graph.json()["concepts"]),"intents":len([x for x in graph.json()["intents"] if x["status"]!="superseded"]),
        "trace_events":len(trace.json()["events"]),"preferences_markdown":prefs.json()["markdown"]}
    OUTPUT.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(OUTPUT),"summary":record["summary"],"judge":record["judge"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

