"""Recover only an omitted final root brace in recorded judge JSON.

No model calls. No edits to strings, matches, IDs, predictions, ranks or gold.
Uses the first recorded raw judge attempt matching the exact payload.
"""
import copy
import hashlib
import json
import random
from pathlib import Path


def hash_json(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def frozen_predictions(result):
    return [{k:r[k] for k in ['id','ranked','retrieval','contexts','predictions','gold_intents']} for r in result['results']]


def payload_for(row,node):
    methods=['two_route','three_route'];random.Random(row['id']).shuffle(methods)
    groups={f'group_{i}':row['predictions'][method] for i,method in enumerate(methods)}
    evidence={e['id']:e for arm in row['contexts'].values() for e in arm}
    return methods,{'current_user_message':node['query'],
        'gold_intents':[{k:v for k,v in g.items() if k in ['intent_id','subject','counterpart','desired_change']} for g in node['gold_intents']],
        'historical_events':list(evidence.values()),'groups':groups}


def append_root_brace(raw):
    # This repair is intentionally narrower than general permissive JSON parsing.
    assert raw.startswith('{') and raw==raw.strip(),'unexpected response wrapper'
    stack=[];inside=False;escaped=False
    for char in raw:
        if inside:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':inside=False
        elif char=='"':inside=True
        elif char in '{[':stack.append(char)
        elif char in '}]':
            assert stack and stack.pop()==('{' if char=='}' else '['),'mismatched JSON'
    assert not inside and not escaped and stack==['{'],'not exactly one missing root brace'
    repaired=raw+'}'
    return json.loads(repaired),repaired


def scores_from_response(response,row,node,methods):
    assert set(response['groups'])=={'group_0','group_1'}
    scores={}
    for j,method in enumerate(methods):
        matches=response['groups'][f'group_{j}']['matches']
        assert isinstance(matches,list)
        pids={i['id'] for i in row['predictions'][method]}
        gids={g['intent_id'] for g in node['gold_intents']}
        assert len({m['prediction_id'] for m in matches})==len(matches)
        assert len({m['gold_id'] for m in matches})==len(matches)
        assert all(m['prediction_id'] in pids and m['gold_id'] in gids for m in matches)
        tp=len(matches);fp=len(pids)-tp;fn=len(gids)-tp
        scores[method]={'tp':tp,'fp':fp,'fn':fn,'matches':matches,
                       'recall':tp/len(gids) if gids else None,
                       'precision':tp/len(pids) if pids else (1 if not gids else 0),
                       'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1,
                       'exact':not fp and not fn}
    return scores


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    path=root/'data/recall_v1_results.json';raw=path.read_bytes();original=json.loads(raw)
    dataset=json.loads((root/'data/recall_dataset_v1.json').read_bytes())
    calls=[json.loads(line) for line in (root/'data/recall_v1_model_calls.jsonl').read_text('utf-8').splitlines()]
    nodes={n['id']:n for c in dataset['cases'] for n in c['evaluation_nodes']}
    result=copy.deepcopy(original);repairs=[]
    for row in result['results']:
        if 'judge_error' not in row:continue
        assert row['judge_error'].startswith('JSONDecodeError:'),'not a JSON parse failure'
        methods,payload=payload_for(row,nodes[row['id']])
        candidates=[(i,c) for i,c in enumerate(calls) if c.get('purpose')=='blind_semantic_intent_match'
                    and 'raw_response' in c and c.get('payload')==payload]
        assert candidates,('no exact-payload raw response',row['id'])
        index,call=candidates[0]
        response,repaired=append_root_brace(call['raw_response'])
        row['intent_scores']=scores_from_response(response,row,nodes[row['id']],methods)
        row['judge_error_before_brace_repair']=row.pop('judge_error')
        repair={'node':row['id'],'call_record_index_zero_based':index,
                'raw_sha256':hashlib.sha256(call['raw_response'].encode()).hexdigest(),
                'repaired_sha256':hashlib.sha256(repaired.encode()).hexdigest(),
                'appended_suffix':'}','method':'append exactly one missing final root brace; first exact-payload raw response'}
        row['judge_brace_repair']=repair;repairs.append(repair)
    assert hash_json(frozen_predictions(original))==hash_json(frozen_predictions(result))
    for before,after in zip(original['results'],result['results']):
        if 'intent_scores' in before:assert before['intent_scores']==after['intent_scores']
    assert all('intent_scores' in r and 'judge_error' not in r for r in result['results'])
    backup=root/'data/recall_v1_before_brace_repair.json'
    assert not backup.exists() or backup.read_bytes()==raw,'backup would be overwritten'
    backup.write_bytes(raw)
    result['brace_repair_note']='Remaining JSON parse failures recovered by appending exactly one missing final root brace to the first recorded exact-payload raw response. No new model calls or semantic edits. All previously successful judgments retained.'
    result['before_brace_repair_sha256']=hashlib.sha256(raw).hexdigest()
    result['judge_brace_repairs']=repairs
    # Keep the old helper summary for provenance; fresh audited metrics are computed separately.
    result['summary_before_brace_repair']=result.pop('summary',{})
    result['phase']='completed'
    path.write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode())
    print(json.dumps({'repaired':len(repairs),'scored':len(result['results']),
                      'prediction_hash_unchanged':hash_json(frozen_predictions(original))==hash_json(frozen_predictions(result))}))
