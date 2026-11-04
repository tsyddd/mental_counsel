"""Real HTTP/SSE + model + durable worker test against the running deployment."""
import json
import os
import time
import uuid
from pathlib import Path
import requests

BASE=os.getenv('GATEWAY_URL','http://gateway:8080')


def main():
 h=requests.Session();tag=uuid.uuid4().hex[:10]
 r=h.post(BASE+'/api/auth/register',json={'email':tag+'@example.test','display_name':'合成测试','password':uuid.uuid4().hex},timeout=20)
 r.raise_for_status();user=r.json()['user_id']
 result={'user_id':user,'synthetic':True,'checks':{},'turns':[]}
 r=h.put(BASE+'/api/auth/me',json={'display_name':'合成回归用户'},timeout=20);r.raise_for_status()
 result['checks']['account_update']=r.json()['display_name']=='合成回归用户'
 r=h.post(BASE+'/api/sessions',json={'mode':'counselor'},timeout=20);r.raise_for_status();sid=r.json()['session_id']
 result['session_id']=sid
 prompts=[
  '我希望你一次只问一个问题，先听我把话说完。这是我长期的交流偏好。',
  '这周有三天写作业到凌晨一点，第二天早读困得撑不住。',
  '科学课做了个小实验，我把步骤记在本子上，感觉挺有意思。',
  '每天放学后先去上补习班，回到家九点多，还要完成学校作业。',
  '原本周日下午有半天休息，后来又被排了一门课，连续两周没空下来。',
  '妈妈觉得成绩还没起色，下周还准备增加一门；我担心晚上又要更晚睡。',
  '你能结合我前面说过的经历，讲讲现在有哪些事情值得先讨论吗？'
 ]
 for index,text in enumerate(prompts,1):
  r=h.post(BASE+f'/api/sessions/{sid}/messages',json={'text':text},stream=True,timeout=(15,290));r.raise_for_status()
  answer='';done={};waited=False;kind=''
  for line in r.iter_lines(decode_unicode=True):
   if line.startswith('event: '):kind=line[7:]
   if line.startswith('data: '):
    data=json.loads(line[6:])
    if kind=='status':waited=True
    elif kind=='error':raise RuntimeError(data.get('error'))
    elif kind=='token':answer+=data.get('text','')
    elif kind=='done':done=data
  if not answer or not done:raise RuntimeError('missing response/done')
  result['turns'].append({'turn':index,'input':text,'answer':answer,'waited':waited,'done':done})
  print(json.dumps({'turn':index,'waited':waited,'job':done.get('memory_job_id')},ensure_ascii=False),flush=True)
 result['checks']['six_round_job']=bool(result['turns'][5]['done'].get('memory_job_id'))
 result['checks']['no_early_job']=all(not t['done'].get('memory_job_id') for t in result['turns'][:5])
 result['checks']['seventh_turn_waited']=result['turns'][6]['waited']
 state=h.get(BASE+'/api/memory',timeout=30);state.raise_for_status();memory=state.json()
 result['checks']['no_concept_layer']='concepts' not in memory
 result['checks']['bounded_window']=all(len(x['content'])<=2400 for x in memory['snapshot'])
 result['checks']['source_rounds']=bool(memory['events']) and all(e['start_turn']>=1 for e in memory['events'])
 prefs=h.get(BASE+'/api/preferences',timeout=20);prefs.raise_for_status()
 result['checks']['user_md_exists']='一次' in prefs.json()['markdown']
 proposals=[x for x in memory['confirmations'] if x['kind']=='intent' and x['status']=='pending']
 result['checks']['intent_requires_confirmation']=bool(proposals) and all(i['status']=='tentative' for i in memory['intents'])
 if proposals:
  r=h.post(BASE+'/api/confirmations/'+proposals[0]['id'],json={'accept':True},timeout=30);r.raise_for_status()
  result['checks']['explicit_confirmation']=any(i['status']=='confirmed' for i in h.get(BASE+'/api/memory',timeout=20).json()['intents'])
 r=h.post(BASE+f'/api/sessions/{sid}/close',json={},timeout=30);r.raise_for_status();result['tail_job']=r.json().get('job_id')
 for _ in range(300):
  memory=h.get(BASE+'/api/memory',timeout=20).json()
  jobs=[j for j in memory['jobs'] if j['id']==result['tail_job']]
  if jobs and jobs[0]['status']=='failed':raise RuntimeError(jobs[0]['error'])
  if jobs and jobs[0]['status']=='completed':break
  time.sleep(1)
 else:raise TimeoutError('tail consolidation')
 result['checks']['tail_flushed']=any(j['end_turn']==7 and j['consolidate'] and j['status']=='completed' for j in memory['jobs'])
 result['checks']['unauthorized_internal_route']=h.get(BASE+'/api/v2/users/'+user+'/state',timeout=20).status_code==404
 r=h.post(BASE+'/api/sessions',json={'mode':'counselor'},timeout=20);r.raise_for_status()
 result['new_session_id']=r.json()['session_id']
 result['checks']['cross_session_memory_retained']=len(h.get(BASE+'/api/memory',timeout=20).json()['events'])>=len(memory['events'])
 result['memory']=memory
 target=Path('/checkpoints/mindtrace-v2-smoke.json');target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result['checks'],ensure_ascii=False),flush=True)
 if not all(result['checks'].values()):raise SystemExit(1)


if __name__=='__main__':main()
