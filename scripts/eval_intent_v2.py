"""Replay frozen timestamps through REAL extraction/dedup/intent/window code.
Graph arm reads existing Intent rows; baseline gets the full timestamped dialogue.
Does not use the baseline as a judge. Report binary detection separately from labels.
"""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from platform_app import mindtrace as mt


def predict_baseline(messages):
 return mt.ask('根据全部带时间的双方对话，识别当前有证据支持、尚未解决的候选跟进意图。'
  '重复主题不等于意图；正常活动、第三方、已经解决和明确拒绝不得误报；'
  '不把咨询师问题或建议当成用户目标。不要求固定事件数。'
  '返回{intents:[{label,category,support_message_ids:[string]}]}，无意图返回空数组。',
  {'messages':messages},'full_timestamped_history_baseline')


def run_case(case):
 user=str(uuid.uuid4())
 sessions={}
 with mt.conn() as db:
  db.execute("INSERT INTO app_users(id,email,display_name,password_hash) VALUES (%s,%s,'synthetic-eval','disabled')",(user,user+'@example.test'))
 all_messages=[]
 durations=[]
 for number in sorted({t['session'] for t in case['turns']}):
  session=str(uuid.uuid4());sessions[number]=session
  turns=[t for t in case['turns'] if t['session']==number]
  with mt.conn() as db:
   db.execute("INSERT INTO sessions(id,user_id,agent_mode) VALUES (%s,%s,'counselor')",(session,user))
  for turn in turns:
   for role in ['user','assistant']:
    mid=str(uuid.uuid4())
    from datetime import datetime,timedelta
    stamp=datetime.fromisoformat(turn['sent_at'])+timedelta(seconds=0 if role=='user' else 20)
    with mt.conn() as db:
     db.execute('INSERT INTO messages(id,session_id,role,content,created_at) VALUES (%s,%s,%s,%s,%s)',(mid,session,role,turn[role],stamp))
    all_messages.append({'id':mid,'session_id':session,'turn':turn['turn'],'sent_at':stamp.isoformat(),'role':role,'content':turn[role]})
   if turn['turn']%6==0 or turn is turns[-1]:
    jid=mt.enqueue_batch(user,session,turn['turn'],consolidate=turn is turns[-1])
    if jid:
     started=time.monotonic()
     while True:
      job=mt.all_rows('SELECT status,error FROM mt_jobs WHERE id=%s',(jid,))[0]
      if job['status']=='completed':break
      if job['status']=='failed':raise RuntimeError(job['error'])
      if time.monotonic()-started>300:raise TimeoutError('memory batch did not finish')
      time.sleep(.5)
     durations.append(round(time.monotonic()-started,3))
  with mt.conn() as db:db.execute('UPDATE mt_session_state SET closed=true WHERE session_id=%s',(session,))
 intents=[i for i in mt.load_intents(user) if i['status'] in ('tentative','confirmed')]
 baseline=predict_baseline(all_messages)
 state=mt.state(user)
 return {'id':case['id'],'scenario_id':case['scenario_id'],'mode':case['mode'],
  'gold_has_intent':case['gold_has_intent'],'gold_category':case['gold_category'],
  'graph_prediction':bool(intents),'baseline_prediction':bool(baseline.get('intents')),
  'graph_intents':intents,'baseline_intents':baseline.get('intents',[]),
  'graph_correct':bool(intents)==case['gold_has_intent'],
  'baseline_correct':bool(baseline.get('intents'))==case['gold_has_intent'],
  'event_count':len(state['events']),'job_seconds':durations,'user_id':user,'sessions':sessions,
  'events':state['events'],'messages':all_messages,'snapshot':state['snapshot'],
  'window_chars':len(state['snapshot'][0]['content']) if state['snapshot'] else 0,
  'all_sources_have_turns':all(e['start_turn']>=1 and e['end_turn']>=e['start_turn'] for e in state['events'])}


def metrics(rows,key):
 tp=sum(r['gold_has_intent'] and r[key] for r in rows)
 tn=sum(not r['gold_has_intent'] and not r[key] for r in rows)
 fp=sum(not r['gold_has_intent'] and r[key] for r in rows)
 fn=sum(r['gold_has_intent'] and not r[key] for r in rows)
 return {'n':len(rows),'accuracy':(tp+tn)/len(rows) if rows else None,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
  'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,
  'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=8);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
 raw=Path(a.dataset).read_bytes();dataset=json.loads(raw);out=Path(a.output)
 result={'dataset_sha256':hashlib.sha256(raw).hexdigest(),'model':mt.APIClient().model,
  'synthetic':True,'paired_variants_not_independent':True,'metric':'binary existence only; raw labels retained for review',
  'results':[],'failures':[]}
 from concurrent.futures import ThreadPoolExecutor,as_completed
 executor=ThreadPoolExecutor(max_workers=max(1,min(2,a.workers)))
 futures={executor.submit(run_case,case):case for case in dataset['cases'][a.start:a.start+a.limit]}
 for future in as_completed(futures):
  case=futures[future]
  try:
   row=future.result();result['results'].append(row)
   print(mt.dumps({'case':row['id'],'graph_correct':row['graph_correct'],'baseline_correct':row['baseline_correct']}),flush=True)
  except Exception as exc:
   result['failures'].append({'case':case['id'],'error':type(exc).__name__+': '+str(exc)[:500]})
   print(mt.dumps(result['failures'][-1]),flush=True)
  result['summary']={mode:{'graph':metrics([r for r in result['results'] if mode=='all' or r['mode']==mode],'graph_prediction'),
   'full_history':metrics([r for r in result['results'] if mode=='all' or r['mode']==mode],'baseline_prediction')} for mode in ['all','long','cross']}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(mt.dumps(result),encoding='utf-8')
