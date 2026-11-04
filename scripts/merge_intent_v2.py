import argparse,json
from pathlib import Path
from eval_intent_v2 import metrics

p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
parts=[json.loads(Path(x).read_text('utf-8')) for x in a.inputs]
assert len({x['dataset_sha256'] for x in parts})==1,'dataset mismatch'
assert len({x['model'] for x in parts})==1,'model mismatch'
rows=[r for x in parts for r in x['results']]
assert len({r['id'] for r in rows})==len(rows),'duplicate case'
result={**parts[0],'results':rows,'failures':[r for x in parts for r in x['failures']]}
result['summary']={mode:{'graph':metrics([r for r in rows if mode=='all' or r['mode']==mode],'graph_prediction'),
 'full_history':metrics([r for r in rows if mode=='all' or r['mode']==mode],'baseline_prediction')} for mode in ['all','long','cross']}
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result['summary']))
