"""Retry only failed JSON judgments, without rerunning or selecting predictions."""
import copy
import hashlib
import json
import time
from pathlib import Path
import eval_recall_v1 as evaluation
from platform_app import mindtrace as mt
from relation_videoarm.api import APIClient,parse_json_object

ROOT=Path('/checkpoints')
SOURCE=ROOT/'recall-v1-results.json'
OUTPUT=ROOT/'recall-v1-rejudged.json'
CALLS=ROOT/'recall-v1-model-calls.jsonl'


def predictions_hash(result):
    frozen=[{k:r[k] for k in ['id','ranked','retrieval','contexts','predictions','gold_intents']} for r in result['results']]
    return hashlib.sha256(json.dumps(frozen,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def raw_ask(system,payload,purpose):
    started=time.monotonic()
    record={'purpose':purpose,'system':system,'payload':payload,'json_retry_only':True}
    try:
        reply=APIClient(timeout=75,attempts=2).chat([
          {'role':'system','content':system+'\n只返回有效JSON对象。'},
          {'role':'user','content':mt.dumps(payload)}],purpose=purpose,max_tokens=2600)
        record['raw_response']=str(reply.get('content') or '')
        value=parse_json_object(record['raw_response']);record['response']=value
        return value
    except Exception as exc:
        record['error']=type(exc).__name__+': '+str(exc)[:800]
        raise
    finally:
        record['seconds']=round(time.monotonic()-started,3)
        with CALLS.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')


if __name__=='__main__':
    raw=SOURCE.read_bytes();original=json.loads(raw)
    result=json.loads(OUTPUT.read_bytes()) if OUTPUT.exists() else copy.deepcopy(original)
    assert predictions_hash(result)==predictions_hash(original),'frozen predictions changed'
    result['original_result_sha256']=hashlib.sha256(raw).hexdigest()
    result['frozen_predictions_sha256']=predictions_hash(original)
    result['rejudge_note']='Only prior JSON parsing failures retried; identical predictions, evidence, prompt and model; successful initial judgments retained; all attempts logged.'
    result['phase']='judging';evaluation.save(OUTPUT,result)
    nodes={n['id']:n for c in json.loads(Path('/tmp/recall_dataset_v1.json').read_bytes())['cases'] for n in c['evaluation_nodes']}
    mt.ask=raw_ask
    for row in result['results']:
        if 'judge_error' not in row:continue
        history=row.setdefault('judge_retry_history',[])
        if not history:history.append({'initial_error':row['judge_error']})
        for attempt in range(1,4):
            try:
                scores=evaluation.judge(row,nodes[row['id']])
                row['intent_scores']=scores;row.pop('judge_error',None)
                history.append({'retry':attempt,'status':'accepted_first_valid_response'})
                break
            except Exception as exc:
                row['judge_error']=type(exc).__name__+': '+str(exc)[:500]
                history.append({'retry':attempt,'error':row['judge_error']})
            finally:evaluation.save(OUTPUT,result)
        print(json.dumps({'node':row['id'],'remaining_error':row.get('judge_error')}),flush=True)
    assert predictions_hash(result)==predictions_hash(original)
    for old,new in zip(original['results'],result['results']):
        if 'intent_scores' in old:assert old['intent_scores']==new['intent_scores'],'successful initial judgment changed'
    result['summary']=evaluation.summarize(result);result['phase']='completed';evaluation.save(OUTPUT,result)
    print(json.dumps({'phase':'completed','judged':sum('intent_scores' in r for r in result['results']),
                      'judge_errors':sum('judge_error' in r for r in result['results'])}),flush=True)
