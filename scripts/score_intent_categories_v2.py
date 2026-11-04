"""Blind post-hoc label mapping of frozen predictions; does not regenerate intents.
The mapper sees descriptions and the same category definitions for both arms,
not the gold label or method name. Its limitations are reported explicitly.
"""
import argparse,json,random
from pathlib import Path
from platform_app import mindtrace as mt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset');a=p.parse_args()
result=json.loads(Path(a.input).read_text('utf-8'))
definitions={'study_rest_balance':'调整学习负担、课程安排与休息的平衡，包括为此与家长协商',
 'peer_participation':'参与同伴活动、融入小组或改善对同伴互动的回避',
 'parent_communication':'其他家庭沟通需求，不包括已经归入学习休息平衡的课程协商',
 'sleep_routine':'独立的睡眠作息调整，不包括学习课程挤占休息的安排问题',
 'other':'上述类别之外的意图'}
if a.dataset:
 import hashlib
 raw=Path(a.dataset).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==result['dataset_sha256'],'mapping dataset mismatch'
 definitions=json.loads(raw)['category_definitions']
for row in result['results']:
 methods=['graph','baseline'];random.Random(row['id']).shuffle(methods)
 groups={f'group_{i}':[str(x.get('label') or x.get('description') or '') for x in row[m+'_intents']] for i,m in enumerate(methods)}
 if any(groups.values()):
  mapping=mt.ask('仅按给定意图描述映射到类别，不推测遗漏意图，不读取对话。'
   '语义相同只计一次；每组独立，无描述返回空数组。返回{groups:{group_0:[类别代码],group_1:[类别代码]}}。',
   {'definitions':definitions,'groups':groups},'blind_intent_label_mapping').get('groups',{})
 else:mapping={k:[] for k in groups}
 for i,m in enumerate(methods):
  predicted=sorted(set(x for x in mapping.get(f'group_{i}',[]) if x in definitions))
  gold=[row['gold_category']] if row['gold_category'] else []
  row[m+'_mapped_categories']=predicted
  row[m+'_category_correct']=(predicted==gold and bool(row[m+'_prediction'])==row['gold_has_intent'])
 result['category_mapping_note']='same-model blind description mapping; fixed taxonomy; no independent human validation'
 result['category_summary']={mode:{m:{'n':len(selected),'correct':sum(x[m+'_category_correct'] for x in selected),
  'accuracy':sum(x[m+'_category_correct'] for x in selected)/len(selected) if selected else None}
  for m in ['graph','baseline']} for mode in ['all','long','cross']
  for selected in [[x for x in result['results'] if (mode=='all' or x['mode']==mode)
                   and 'graph_category_correct' in x and 'baseline_category_correct' in x]]}
 Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'case':row['id'],'graph':row['graph_mapped_categories'],'baseline':row['baseline_mapped_categories']}),flush=True)
