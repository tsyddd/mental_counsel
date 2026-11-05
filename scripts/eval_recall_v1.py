"""Serial, checkpointed retrieval ablation with frozen data and field-only tagging.

Uses the configured production embedding/model, but not graph memory updates.
All LLM calls are serial. Raw artifacts are saved before any scoring.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from platform_app import mindtrace as mt
from platform_app.mindtrace_logic import lexical,cosine

CALL_LOG=None
ORIGINAL_ASK=mt.ask


def audited_ask(system,payload,purpose):
    started=time.monotonic()
    record={'purpose':purpose,'system':system,'payload':payload}
    try:
        value=ORIGINAL_ASK(system,payload,purpose);record['response']=value
        return value
    except Exception as exc:
        record['error']=type(exc).__name__+': '+str(exc)[:800]
        raise
    finally:
        record['seconds']=round(time.monotonic()-started,3)
        if CALL_LOG:
            with CALL_LOG.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(json.dumps(value,ensure_ascii=False,indent=2,default=str).encode())
    temp.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def score_fields(a,b):
    def overlap(key):
        x=set(a[key])-{'其他','未知'};y=set(b[key])-{'其他','未知'}
        return len(x&y)/len(x|y) if x|y else 0
    return .5*overlap('scene')+.5*overlap('activity')


def fields_valid(value,dataset):
    return isinstance(value,dict) and all(isinstance(value.get(k),list) and value[k]
       and all(isinstance(x,str) and x in dataset[k+'_values'] for x in value[k]) for k in ['scene','activity'])


def tag_batch(items,dataset):
    response=mt.ask('为每段输入独立填写固定情境字段，不推断未提到的事实。只允许scene、activity两个字段，多选。'
                    '依据内容不是关键词猜测，证据不足用未知。不要生成意图，不建立条目间关联。'
                    '返回{items:[{id,scene:[枚举值],activity:[枚举值]}]}。',
                    {'scene_values':dataset['scene_values'],'activity_values':dataset['activity_values'],
                     'items':[{'id':x[0],'text':x[1]} for x in items]},'recall_scene_activity_annotation')
    out={x['id']:x for x in response.get('items',[]) if isinstance(x,dict) and 'id' in x}
    assert set(out)=={x[0] for x in items},'incomplete scene/activity annotation'
    assert all(fields_valid(x,dataset) for x in out.values()),'invalid scene/activity enum'
    return {key:{k:out[key][k] for k in ['scene','activity']} for key,_ in items}


def retrieval_metrics(ids,gold,k):
    hits=sum(x in gold for x in ids[:k]);ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
    dcg=sum(1/math.log2(i+2) for i,x in enumerate(ids[:k]) if x in gold)
    return {'recall':hits/len(gold) if gold else None,'precision':hits/k,
            'ndcg':dcg/ideal if ideal else None,'hits':hits,'relevant':len(gold)}


def predict(query,events):
    result=mt.ask('根据当前用户发言及检索到的历史事件，识别用户当前有证据支持且尚未解决的候选意图。'
       '可有多个意图或没有意图；区别主题与目标，不把第三方、已解决、正常满意、明确拒绝当成当前个人意图。'
       '不将当前发言独自推断的目标冒充历史证据；历史证据不足不要补编。'
       '每条保留主体、对象和期望改变；同义不重复，不同目标不合并；不确定可澄清，不能当成用户已确认。'
       '只允许引用给定事件ID。返回{intents:[{id,subject,counterpart,desired_change,support_event_ids:[string]}]}。',
       {'current_user_message':query,'retrieved_events':events},'recall_downstream_intent_extraction')
    intents=result.get('intents')
    assert isinstance(intents,list),'invalid intent output'
    allowed={e['id'] for e in events}
    for i,item in enumerate(intents):
        assert isinstance(item,dict) and isinstance(item.get('desired_change'),str),'invalid intent fields'
        support=item.get('support_event_ids')
        assert isinstance(support,list) and support and all(x in allowed for x in support),'invalid intent evidence'
        item['id']='p'+str(i+1)
    return intents


def judge(row,node):
    methods=['two_route','three_route'];random.Random(row['id']).shuffle(methods)
    groups={f'group_{i}':row['predictions'][method] for i,method in enumerate(methods)}
    evidence={e['id']:e for arm in row['contexts'].values() for e in arm}
    result=mt.ask('评估匿名两组候选意图。比较主体、交互对象和期望改变的语义，允许合理改写和领域重叠。'
       '不得仅按主题/类别相同匹配；过于笼统丢失必要对象不能匹配。每个预测只匹配一个参考，每个参考只匹配一个预测。'
       '预测目标必须有其引用的历史事件支持，当前问题只能辅助消歧，不能补造缺失历史。'
       '第三方、已解决、明确拒绝不能算当前个人目标。存在两个不同目标的一句复合预测只计一次。'
       '返回{groups:{group_0:{matches:[{prediction_id,gold_id,reason}]},group_1:{matches:[{prediction_id,gold_id,reason}]}}}。',
       {'current_user_message':node['query'],
        'gold_intents':[{k:v for k,v in g.items() if k in ['intent_id','subject','counterpart','desired_change']} for g in node['gold_intents']],
        'historical_events':list(evidence.values()),'groups':groups},'blind_semantic_intent_match')
    out={}
    for j,method in enumerate(methods):
        matches=result['groups'][f'group_{j}']['matches']
        assert isinstance(matches,list),'invalid matches'
        pids={i['id'] for i in row['predictions'][method]};gids={g['intent_id'] for g in node['gold_intents']}
        assert len({x['prediction_id'] for x in matches})==len(matches),'duplicate prediction match'
        assert len({x['gold_id'] for x in matches})==len(matches),'duplicate gold match'
        assert all(x['prediction_id'] in pids and x['gold_id'] in gids for x in matches),'unknown match ID'
        tp=len(matches);fp=len(pids)-tp;fn=len(gids)-tp
        out[method]={'tp':tp,'fp':fp,'fn':fn,'matches':matches,
                     'recall':tp/len(gids) if gids else None,
                     'precision':tp/len(pids) if pids else (1 if not gids else 0),
                     'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1,
                     'exact':not fp and not fn}
    return out


def summarize(result):
    summary={}
    for mode in ['all','long','cross']:
        rows=[r for r in result['results'] if mode=='all' or r['mode']==mode]
        item={'completed_nodes':len(rows)}
        for kind in ['all','multi_intent','negative_status_check']:
            selected=[r for r in rows if kind=='all' or r['query_kind']==kind]
            arms={}
            for method in ['two_route','three_route']:
                retr={}
                for k in [3,5,10]:
                    vals=[r['retrieval'][method][str(k)] for r in selected]
                    retr[str(k)]={metric:sum(v[metric] for v in vals)/len(vals) if vals else None for metric in ['recall','precision','ndcg']}
                scored=[r['intent_scores'][method] for r in selected if 'intent_scores' in r]
                tp=sum(x['tp'] for x in scored);fp=sum(x['fp'] for x in scored);fn=sum(x['fn'] for x in scored)
                arms[method]={'retrieval':retr,'intent':{'scored_nodes':len(scored),'tp':tp,'fp':fp,'fn':fn,
                    'micro_recall':tp/(tp+fn) if tp+fn else None,'micro_precision':tp/(tp+fp) if tp+fp else None,
                    'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                    'exact_nodes':sum(x['exact'] for x in scored)}}
            item[kind]=arms
        summary[mode]=item
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=Path(a.dataset).read_bytes();dataset=json.loads(raw);path=Path(a.output)
    CALL_LOG=path.with_name('recall-v1-model-calls.jsonl');mt.ask=audited_ask
    cachepath=path.with_name('recall-v1-cache.json')
    model=mt.APIClient().model
    cache=json.loads(cachepath.read_bytes()) if cachepath.exists() else {'model':model,'tags':{},'vectors':{}}
    assert cache['model']==model
    result=json.loads(path.read_bytes()) if path.exists() else {'dataset_sha256':hashlib.sha256(raw).hexdigest(),
       'model':model,'config':dataset['frozen_config'],'event_unit':dataset['event_unit'],
       'scope':dataset['scope'],'limitations':dataset['limitations']+['same-model assisted semantic judge; pending independent review'],
       'results':[],'failures':[],'phase':'annotation'}
    assert result['dataset_sha256']==hashlib.sha256(raw).hexdigest() and result['model']==model
    save(path,result)
    # Query and event tagging use separate calls; neither contains relevance labels or gold goals.
    eventtexts={digest(e['content']):e['content'] for c in dataset['cases'] for e in c['events']}
    querytexts={digest(n['query']):n['query'] for c in dataset['cases'] for n in c['evaluation_nodes']}
    for kind,texts in [('event',eventtexts),('query',querytexts)]:
        pending=[(key,text) for key,text in texts.items() if key not in cache['tags']]
        for start in range(0,len(pending),6):
            batch=pending[start:start+6]
            cache['tags'].update(tag_batch(batch,dataset));save(cachepath,cache)
            print(json.dumps({'phase':'annotation','kind':kind,'done':min(start+6,len(pending)),'pending_at_start':len(pending)}),flush=True)
    for prefix,texts in [('passage: ',eventtexts),('query: ',querytexts)]:
        pending=[(prefix+key,prefix+text) for key,text in texts.items() if prefix+key not in cache['vectors']]
        for start in range(0,len(pending),24):
            batch=pending[start:start+24];vectors=mt.embed([x[1] for x in batch])
            cache['vectors'].update({x[0]:v for x,v in zip(batch,vectors)});save(cachepath,cache)
    result['phase']='prediction';save(path,result)
    finished={r['id'] for r in result['results']}|{r['id'] for r in result['failures']}
    for case in dataset['cases']:
        for node in case['evaluation_nodes']:
            if node['id'] in finished:continue
            try:
                query=node['query'];qkey=digest(query);qvec=cache['vectors']['query: '+qkey];qtag=cache['tags'][qkey]
                corpus=[e for e in case['events'] if e['global_turn']<node['at_global_turn']]
                scored=[]
                for event in corpus:
                    key=digest(event['content']);tag=cache['tags'][key]
                    ks=.75*cosine(qvec,cache['vectors']['passage: '+key])+.25*lexical(query,event['content'])
                    context_score=score_fields(qtag,tag)
                    scored.append((event,ks,.8*ks+.2*context_score,context_score))
                ranked={method:[x[0]['id'] for x in sorted(scored,key=lambda x:(-x[col],x[0]['id']))]
                        for method,col in [('two_route',1),('three_route',2)]}
                fieldrank=[x[0]['id'] for x in sorted(scored,key=lambda x:(-x[3],x[0]['id'])) if x[3]>0][:5]
                union=list(dict.fromkeys(ranked['two_route'][:10]+fieldrank))
                gold=set(node['gold_relevant_event_ids'])
                row={'id':node['id'],'case_id':case['id'],'scenario_id':case['scenario_id'],'mode':case['mode'],
                     'at_global_turn':node['at_global_turn'],'query':query,'query_kind':node['query_kind'],
                     'query_fields':qtag,'gold_relevant_event_ids':sorted(gold),'gold_intents':node['gold_intents'],
                     'ranked':ranked,'retrieval':{m:{str(k):retrieval_metrics(ids,gold,k) for k in [3,5,10]} for m,ids in ranked.items()},
                     'union_diagnostic':{'ids':union,'candidate_count':len(union),'recall':len(set(union)&gold)/len(gold)},
                     'event_scores':[{'id':e['id'],'two_route':ks,'three_route':ksc,'field_score':cs,
                                      'fields':cache['tags'][digest(e['content'])]} for e,ks,ksc,cs in scored],
                     'contexts':{},'predictions':{}}
                byid={e['id']:e for e in corpus}
                # Both branches receive at most ten complete event entries, same evidence schema.
                # Scene/activity are ranking features only, not additional information for the reader.
                result['results'].append(row);save(path,result)
                for method in ['two_route','three_route']:
                    context=[byid[e] for e in ranked[method][:10]];row['contexts'][method]=context
                    row['predictions'][method]=predict(query,context)
                    save(path,result)
                result['summary']=summarize(result);save(path,result)
                print(json.dumps({'completed':len(result['results']),'node':node['id'],
                                  'two_r10':row['retrieval']['two_route']['10']['recall'],
                                  'three_r10':row['retrieval']['three_route']['10']['recall']}),flush=True)
            except Exception as exc:
                result['failures'].append({'id':node['id'],'stage':'prediction','error':type(exc).__name__+': '+str(exc)[:500]})
                save(path,result);print(json.dumps(result['failures'][-1]),flush=True)
    result['phase']='judging';save(path,result)
    nodes={n['id']:n for c in dataset['cases'] for n in c['evaluation_nodes']}
    for row in result['results']:
        if 'intent_scores' in row or 'judge_error' in row:continue
        if set(row['predictions'])!={'two_route','three_route'}:
            row['judge_error']='incomplete prediction pair; failure retained';save(path,result);continue
        try:row['intent_scores']=judge(row,nodes[row['id']])
        except Exception as exc:row['judge_error']=type(exc).__name__+': '+str(exc)[:500]
        result['summary']=summarize(result);save(path,result)
        print(json.dumps({'judged':row['id'],'error':row.get('judge_error')}),flush=True)
    result['phase']='completed';result['summary']=summarize(result);save(path,result)
    print(json.dumps({'phase':'completed','nodes':len(result['results']),'failures':len(result['failures'])}),flush=True)
