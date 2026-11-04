"""Attach persisted source messages/events to frozen prediction records for audit."""
import argparse,json
from pathlib import Path
from platform_app import mindtrace as mt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset');a=p.parse_args()
result=json.loads(Path(a.input).read_text('utf-8'))
expected={}
if a.dataset:
 import hashlib
 raw=Path(a.dataset).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==result['dataset_sha256'],'audit dataset mismatch'
 expected={c['id']:len(c['turns'])*2 for c in json.loads(raw)['cases']}
for row in result['results']:
 state=mt.state(row['user_id'])
 row['events']=state['events'];row['snapshot']=state['snapshot']
 row['messages']=[]
 for number,sid in sorted(row['sessions'].items(),key=lambda x:int(x[0])):
  for m in mt.messages_for(sid):
   row['messages'].append({**m,'session_id':sid,'session_number':int(number)})
 assert len(row['messages'])==expected.get(row['id'],48),('incomplete dialogue',row['id'])
 ids={e['id'] for e in row['events']}
 assert all(set(i['support_event_ids'])<=ids for i in row['graph_intents'])
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print('Auditable cases:',len(result['results']))
