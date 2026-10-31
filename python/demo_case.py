"""A clearly synthetic, repeatable two-session memory-graph showcase."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from .core import conn, embed, vector_literal

SCENES = [
    ("上周数学测验没考好，我怕妈妈失望，把试卷藏了两天。后来我告诉她，她陪我一起看错题。",
     "你担心让妈妈失望，但最后还是告诉了她，也得到了支持。说出来之前，最难的是哪一步？"),
    ("我希望你一次只问我一个问题，先听我说完。",
     "好，我会一次只问一个问题，也会留出时间让你说完。"),
    ("小组讨论时我想说自己的想法，又怕说错，最后什么都没说，回家觉得有点遗憾。",
     "你已经有想说的内容，却被担心拦住了。那个想法现在还记得吗？"),
    ("今天课堂上我还是没敢举手，不过下课后把问题写在纸上交给了老师。",
     "这是一次新的尝试。把问题递给老师时，你更担心什么？"),
    ("我想慢慢练习向同学和老师提问，从比较容易的情景开始。",
     "我们可以从压力较小的一次提问开始。你想先试哪一种情景？"),
    ("你好。上次你提到我在小组讨论里没发言，你还记得吗？",
     "记得。那次你想表达却担心说错，后来你还试过把问题写给老师。今天想从哪一段聊起？"),
    ("这周我主动问了同学一道作业题，对方解释得很耐心，没有我想象中尴尬。",
     "这和之前的担心相比，是一次重要的新经验。你觉得是什么让你迈出了这一步？"),
]

EVENTS = [
    (0, "数学测验失利后担心妈妈失望，藏起试卷两天；最终主动告诉妈妈并一起看错题。", "求助与表达"),
    (2, "小组讨论中想发言却担心说错，最终沉默，之后感到遗憾。", "求助与表达"),
    (3, "课堂上没有举手，但课后将问题写在纸上交给老师。", "求助与表达"),
    (6, "主动向同学请教作业题，对方耐心解释，体验比预想轻松。", "求助与表达"),
]

def load_demo(user_id):
    with conn() as db:
        old = db.execute("SELECT latest_session_id FROM demo_imports WHERE user_id=%s",
                         (user_id,)).fetchone()
        if old:
            return {"session_id": str(old["latest_session_id"]), "imported": False}
    texts = [item for scene in SCENES for item in scene]
    descriptions = [item[1] for item in EVENTS]
    descriptions += ["求助与表达", "希望对话时一次只问一个问题，先听自己说完。"]
    vectors = [vector_literal(item) for item in embed(["passage: " + text for text in descriptions])]
    session_ids = [uuid.uuid4(), uuid.uuid4()]
    message_ids = []
    now = datetime.now(timezone.utc)
    with conn() as db:
        for index, session_id in enumerate(session_ids):
            stamp = now - timedelta(days=2-index)
            db.execute("""INSERT INTO sessions(id,user_id,agent_mode,created_at,last_message_at)
                VALUES (%s,%s,'counselor',%s,%s)""", (session_id,user_id,stamp,stamp))
            db.execute("INSERT INTO episodes(session_id,episode_no) VALUES (%s,1)", (session_id,))
        for i,(visitor,counselor) in enumerate(SCENES):
            sid = session_ids[0 if i<5 else 1]
            at = now - timedelta(days=2 if i<5 else 1) + timedelta(minutes=i*6)
            user_message_id = uuid.uuid4()
            message_ids.append(user_message_id)
            db.execute("""INSERT INTO messages(id,session_id,role,content,created_at)
                VALUES (%s,%s,'user',%s,%s)""", (user_message_id,sid,visitor,at))
            db.execute("""INSERT INTO messages(id,session_id,role,content,created_at)
                VALUES (%s,%s,'assistant',%s,%s)""", (uuid.uuid4(),sid,counselor,at+timedelta(seconds=30)))
            trace = [
                ("input","接收来访者消息",{"characters":len(visitor)}),
                ("retrieval","按需检索跨会话记忆",{"route":"event" if i==5 else "none","events":2 if i==5 else 0}),
                ("context","组装最近对话与长期偏好",{"recent_turns":min(i+1,6),"pinned_preferences":1 if i>1 else 0}),
                ("response","展示案例回复",{"model":"curated-fixture","characters":len(counselor),"synthetic_fixture":True}),
            ]
            for offset,(kind,title,detail) in enumerate(trace):
                db.execute("""INSERT INTO trace_events(id,user_id,session_id,kind,title,detail,created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (uuid.uuid4(),user_id,sid,kind,title,json.dumps(detail,ensure_ascii=False),
                     at+timedelta(seconds=offset*8)))
        concept_id = uuid.uuid4()
        event_ids = []
        for index,(scene_idx,content,concept) in enumerate(EVENTS):
            mid = uuid.uuid4()
            event_ids.append(mid)
            db.execute("""INSERT INTO memories(id,user_id,kind,layer,content,source_type,source_id,
                confidence,embedding,metadata) VALUES (%s,%s,'person_event','episodic',%s,'chat',%s,
                .92,%s::vector,%s)""",
                (mid,user_id,content,str(message_ids[scene_idx]),vectors[index],
                 json.dumps({"demo":True,"session_id":str(session_ids[0 if scene_idx<5 else 1])})))
        db.execute("""INSERT INTO concepts(id,user_id,label,embedding,memory_ids,support_count,keywords)
            VALUES (%s,%s,%s,%s::vector,%s,4,%s)""",
            (concept_id,user_id,"求助与表达",vectors[4],event_ids,["提问","表达","担心"]))
        db.execute("""INSERT INTO intents(id,user_id,label,concept_ids,support_count,status)
            VALUES (%s,%s,%s,%s,4,'tentative')""",
            (uuid.uuid4(),user_id,"是否愿意继续在低压力情境中尝试表达与求助？",[concept_id]))
        db.execute("""INSERT INTO memories(id,user_id,kind,layer,content,source_type,source_id,
            confidence,embedding,metadata) VALUES (%s,%s,'communication_preference','user',%s,
            'chat',%s,.99,%s::vector,%s)""",
            (uuid.uuid4(),user_id,descriptions[5],str(message_ids[1]),vectors[5],
             json.dumps({"demo":True})))
        db.execute("""INSERT INTO user_preference_docs(user_id,notes) VALUES (%s,%s)
            ON CONFLICT(user_id) DO NOTHING""",
            (user_id,"- 遇到困难时先确认我的具体经历，再一起讨论下一步。"))
        db.execute("""INSERT INTO demo_imports(user_id,latest_session_id)
            VALUES (%s,%s)""", (user_id,session_ids[1]))
    return {"session_id":str(session_ids[1]),"imported":True,
            "sessions":[str(item) for item in session_ids],"events":len(EVENTS)}
