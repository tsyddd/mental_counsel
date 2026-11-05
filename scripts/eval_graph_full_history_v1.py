"""Main comparison: persisted production intent graph vs full timestamped history.

Sequential cases; sole production worker; no TopK reader masquerading as graph.
Checkpointed replay state supports restart without resampling completed results.
"""
import hashlib
import json
import random
import time
import uuid
from datetime import datetime,timedelta
from pathlib import Path
from platform_app import mindtrace as mt
from relation_videoarm.api import APIClient,parse_json_object
from repair_recall_judge_braces import append_root_brace

ROOT=Path('/checkpoints')
OUTPUT=ROOT/'graph-full-v1-results.json'
CALLS=ROOT/'graph-full-v1-calls.jsonl'


def save(value):
    temp=OUTPUT.with_suffix('.json.tmp')
    temp.write_bytes(json.dumps(value,ensure_ascii=False,indent=2,default=str).encode())
    temp.replace(OUTPUT)


def ask(system,payload,purpose):
    started=time.monotonic();record={'purpose':purpose,'system':system,'payload':payload}
    try:
        answer=APIClient(timeout=90,attempts=2).chat([
          {'role':'system','content':system+'\n只返回有效JSON对象。'},
          {'role':'user','content':mt.dumps(payload)}],purpose=purpose,max_tokens=2600)
        raw=str(answer.get('content') or '');record['raw_response']=raw
        try:value=parse_json_object(raw)
        except json.JSONDecodeError:
            value,_=append_root_brace(raw.strip());record['repair']='append one missing final root brace only'
        record['response']=value;return value
    except Exception as exc:
        record['error']=type(exc).__name__+': '+str(exc)[:600];raise
    finally:
        record['seconds']=round(time.monotonic()-started,3)
        with CALLS.open('a',encoding='utf-8') as handle:handle.write(mt.dumps(record)+'\n')


def full_history(messages):
    answer=ask('根据全部带时间戳、来源ID及会话边界的双方对话，识别用户当前有证据支持且尚未解决的候选意图。'
      '可有多个或没有意图，主题不等于目标；第三方、已解决、正常满意及明确拒绝不能当作当前个人意图。'
      '保留主体、交互对象和期望改变，同义不重复，不同目标不合并；候选不等于用户已确认。'
      '历史证据不足不要补编，不把助手建议当用户目标。只引用本次当前提问之前的用户消息。'
      '返回{intents:[{id,subject,counterpart,desired_change,source_message_ids:[string]}]}。',
      {'messages':messages},'full_history_main_intent_baseline')
    intents=answer.get('intents');assert isinstance(intents,list),'invalid baseline intents'
    allowed={m['id'] for m in messages[:-1] if m['role']=='user'}
    for j,intent in enumerate(intents):
        assert isinstance(intent,dict) and isinstance(intent.get('desired_change'),str)
        ids=intent.get('source_message_ids')
        assert isinstance(ids,list) and ids and set(ids)<=allowed,'invalid baseline evidence'
        intent['id']='p'+str(j+1)
    return intents


def wait_job(jid):
    if not jid:return
    started=time.monotonic()
    while True:
        job=mt.all_rows('SELECT status,error FROM mt_jobs WHERE id=%s',(jid,))[0]
        if job['status']=='completed':return
        if job['status']=='failed':raise RuntimeError('production memory job failed: '+str(job['error']))
        if time.monotonic()-started>1200:raise TimeoutError('production memory worker wait exceeded 1200s')
        time.sleep(1)


def snapshot_graph(user):
    state=mt.state(user)
    active=[i for i in mt.load_intents(user) if i['status'] in ('tentative','confirmed')]
    preds=[]
    for j,intent in enumerate(active):
        preds.append({'id':'p'+str(j+1),'subject':'user',
                      'description':intent['label'],'source_event_ids':intent['support_event_ids'],
                      'counter_event_ids':intent['counter_event_ids'],'status':intent['status']})
    return preds,active,state


def score(row,node):
    methods=['graph','full_history'];random.Random(row['id']).shuffle(methods)
    groups={}
    for j,method in enumerate(methods):
        groups[f'group_{j}']={'intents':row['predictions'][method],
                            'evidence':row['graph_events'] if method=='graph' else row['full_history_input'][:-1]}
    response=ask('对匿名两组候选意图做一对一语义匹配。根据主体、对象与期望改变判定，不要求类别或措辞相同。'
       '预测描述须表达对应目标，过于笼统而丢失必要对象不能匹配；每条预测及参考最多匹配一次。'
       '每组只能使用该组预测实际引用的历史证据；另一组证据不能补足本组缺失。当前提问只用于消歧。'
       '事件的eligible_history_support为false时表示只来自当前提问，不能充当之前的历史支持。'
       '不要因某组证据为事件摘要或原消息而偏向任一组。第三方、已解决、拒绝不算当前目标。'
       '返回{groups:{group_0:{matches:[{prediction_id,gold_id,reason}]},group_1:{matches:[{prediction_id,gold_id,reason}]}}}。',
       {'current_user_message':node['query'],
        'gold_intents':[{k:v for k,v in g.items() if k in ['intent_id','subject','counterpart','desired_change']} for g in node['gold_intents']],
        'groups':groups},'blind_graph_vs_full_intent_judge')
    scores={}
    for j,method in enumerate(methods):
        matches=response['groups'][f'group_{j}']['matches']
        ps={p['id'] for p in row['predictions'][method]};gs={g['intent_id'] for g in node['gold_intents']}
        assert isinstance(matches,list)
        assert len({m['prediction_id'] for m in matches})==len(matches)
        assert len({m['gold_id'] for m in matches})==len(matches)
        assert all(m['prediction_id'] in ps and m['gold_id'] in gs for m in matches)
        tp=len(matches);fp=len(ps)-tp;fn=len(gs)-tp
        scores[method]={'tp':tp,'fp':fp,'fn':fn,'matches':matches,'exact':not fp and not fn}
    return scores


def summary(result,dataset):
    summaries={}
    for mode in ['all','long','cross']:
        expected=[n for c in dataset['cases'] if mode=='all' or c['mode']==mode for n in c['evaluation_nodes']]
        rows=[r for r in result['results'] if (mode=='all' or r['mode']==mode) and 'scores' in r]
        out={'expected_nodes':len(expected),'scored_nodes':len(rows),'arms':{}}
        for method in ['graph','full_history']:
            tp=sum(r['scores'][method]['tp'] for r in rows);fp=sum(r['scores'][method]['fp'] for r in rows);fn=sum(r['scores'][method]['fn'] for r in rows)
            gold=sum(len(n['gold_intents']) for n in expected)
            out['arms'][method]={'tp':tp,'fp':fp,'fn':fn,'expected_gold':gold,
                 'micro_recall_scored':tp/(tp+fn) if tp+fn else None,
                 'micro_precision_scored':tp/(tp+fp) if tp+fp else None,
                 'micro_f1_scored':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                 'recall_all_expected_lower_bound':tp/gold if gold else None}
        summaries[mode]=out
    return summaries


if __name__=='__main__':
    raw=Path('/tmp/recall_dataset_v1.json').read_bytes();dataset=json.loads(raw)
    model=APIClient().model
    result=json.loads(OUTPUT.read_bytes()) if OUTPUT.exists() else {
       'dataset_sha256':hashlib.sha256(raw).hexdigest(),'model':model,'phase':'replay',
       'scope':'actual persisted event-to-intent graph vs all timestamped bilateral history; not a TopK reader',
       'graph_implementation':'current deployed two-layer memory; scene/activity retrieval changes not yet integrated into production',
       'concurrency':1,'results':[],'failures':[],'replay_state':{},
       'limitations':dataset['limitations']+['same-model assisted semantic judging, not independent human review',
                                           'same model, different total processing budgets; no clinical efficacy claim']}
    assert result['dataset_sha256']==hashlib.sha256(raw).hexdigest() and result['model']==model
    save(result)
    existing={r['id']:r for r in result['results']}
    for case in dataset['cases']:
        if case['id'] not in result['replay_state']:
            user=str(uuid.uuid4())
            with mt.conn() as db:
                db.execute("INSERT INTO app_users(id,email,display_name,password_hash) VALUES (%s,%s,'graph-full-eval','disabled')",(user,user+'@example.test'))
            result['replay_state'][case['id']]={'user_id':user,'sessions':{},'messages':[],'inserted_through':0}
            save(result)
        replay=result['replay_state'][case['id']];user=replay['user_id']
        checkpoints={n['at_global_turn']:n for n in case['evaluation_nodes']}
        try:
            for turn in case['turns']:
                n=turn['global_turn'];session_number=str(turn['session'])
                if session_number not in replay['sessions']:
                    sid=str(uuid.uuid4())
                    with mt.conn() as db:db.execute("INSERT INTO sessions(id,user_id,agent_mode) VALUES (%s,%s,'counselor')",(sid,user))
                    replay['sessions'][session_number]=sid;save(result)
                sid=replay['sessions'][session_number]
                if n>replay['inserted_through']:
                    for role in ['user','assistant']:
                        mid=str(uuid.uuid4());stamp=datetime.fromisoformat(turn['sent_at'])+timedelta(seconds=0 if role=='user' else 20)
                        with mt.conn() as db:
                            db.execute('INSERT INTO messages(id,session_id,role,content,created_at) VALUES (%s,%s,%s,%s,%s)',(mid,sid,role,turn[role],stamp))
                        replay['messages'].append({'id':mid,'session_id':sid,'session_number':turn['session'],
                           'turn':turn['turn'],'global_turn':n,'sent_at':stamp.isoformat(),'role':role,'content':turn[role]})
                    replay['inserted_through']=n;save(result)
                elif n<replay['inserted_through']:
                    if n in checkpoints:assert checkpoints[n]['id'] in existing,'lost past graph snapshot; do not score future state'
                    continue
                ends_session=n==60 or (case['mode']=='cross' and n==30)
                if turn['turn']%6==0 or ends_session:
                    job=mt.enqueue_batch(user,sid,turn['turn'],consolidate=ends_session)
                    wait_job(job)
                    print(json.dumps({'case':case['id'],'through_global_turn':n,'job':job}),flush=True)
                if n not in checkpoints:continue
                node=checkpoints[n];row=existing.get(node['id'])
                if row is None:
                    preds,intents,state=snapshot_graph(user)
                    # Full history ends at the current user utterance, not its scripted reply.
                    messages=[m for m in replay['messages'] if m['global_turn']<n or (m['global_turn']==n and m['role']=='user')]
                    source_turns={m['id']:m['global_turn'] for m in replay['messages']}
                    for event in state['events']:
                        event['source_global_turns']=[source_turns[mid] for mid in event['source_ids']]
                        event['eligible_history_support']=any(t<n for t in event['source_global_turns'])
                    row={'id':node['id'],'case_id':case['id'],'scenario_id':case['scenario_id'],'mode':case['mode'],
                         'at_global_turn':n,'user_id':user,'gold_intents':node['gold_intents'],
                         'graph_intents':intents,'graph_events':state['events'],'graph_snapshot':state['snapshot'],
                         'full_history_input':messages,'predictions':{'graph':preds}}
                    result['results'].append(row);existing[node['id']]=row;save(result)
                if 'full_history' not in row['predictions']:
                    row['predictions']['full_history']=full_history(row['full_history_input']);save(result)
                print(json.dumps({'checkpoint':node['id'],'prediction_pairs':sum('full_history' in r['predictions'] for r in result['results'])}),flush=True)
        except Exception as exc:
            result['failures'].append({'case':case['id'],'stage':'replay_or_prediction','error':type(exc).__name__+': '+str(exc)[:700]})
            save(result);print(json.dumps(result['failures'][-1]),flush=True)
    result['phase']='judging';save(result)
    nodes={n['id']:n for c in dataset['cases'] for n in c['evaluation_nodes']}
    for row in result['results']:
        if 'scores' in row or 'judge_error' in row:continue
        if 'full_history' not in row['predictions']:
            row['judge_error']='baseline prediction unavailable';save(result);continue
        try:row['scores']=score(row,nodes[row['id']])
        except Exception as exc:row['judge_error']=type(exc).__name__+': '+str(exc)[:700]
        result['summary']=summary(result,dataset);save(result)
        print(json.dumps({'judged':row['id'],'error':row.get('judge_error')}),flush=True)
    result['phase']='completed';result['summary']=summary(result,dataset);save(result)
    print(json.dumps({'phase':'completed','nodes':len(result['results']),'failures':len(result['failures'])}),flush=True)
