"""Audit frozen retrieval artifacts and compute paired scenario-level intervals."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values)/len(values) if values else None


def bootstrap(groups):
    values=[mean(v) for _,v in sorted(groups.items())]
    if not values:return None
    rng=random.Random(20261105)
    draws=sorted(mean([rng.choice(values) for _ in values]) for _ in range(10000))
    return {'scenario_groups':len(values),'mean_difference':mean(values),
            'percentile_95ci':[draws[249],draws[9749]],
            'note':'three minus two; paired synthetic development scenarios, not population inference'}


def summarize(dataset,result,cache):
    nodes={n['id']:(c,n) for c in dataset['cases'] for n in c['evaluation_nodes']}
    rows={r['id']:r for r in result['results']}
    assert len(rows)==len(result['results']) and rows.keys()<=nodes.keys()
    errors=[];improvements=[];coverage=[]
    for key,row in rows.items():
        case,node=nodes[key]
        assert row['gold_relevant_event_ids']==sorted(node['gold_relevant_event_ids'])
        assert row['gold_intents']==node['gold_intents']
        corpus={e['id']:e for e in case['events'] if e['global_turn']<node['at_global_turn']}
        gold=set(node['gold_relevant_event_ids'])
        for method in ['two_route','three_route']:
            ranked=row['ranked'][method]
            assert len(ranked)==len(corpus) and set(ranked)==set(corpus)
            for k in [3,5,10]:
                m=row['retrieval'][method][str(k)]
                hits=len(set(ranked[:k])&gold)
                assert abs(m['recall']-hits/len(gold))<1e-12
                assert m['hits']==hits
            if method in row['contexts']:
                assert row['contexts'][method]==[corpus[x] for x in ranked[:10]]
        two=set(row['ranked']['two_route'][:10]);three=set(row['ranked']['three_route'][:10])
        diff=row['retrieval']['three_route']['10']['recall']-row['retrieval']['two_route']['10']['recall']
        detail={'node':key,'query':row['query'],'gold':sorted(gold),
                'delta_r10':diff,'three_only_relevant':sorted((three-two)&gold),
                'two_only_relevant':sorted((two-three)&gold)}
        if diff<0:errors.append(detail)
        elif diff>0:improvements.append(detail)
        for method in ['two_route','three_route']:
            complete=sum(set(f"e{n:03d}" for n in i['evidence_global_turns'])<=set(row['ranked'][method][:10]) for i in node['gold_intents'])
            any_hit=sum(bool(set(f"e{n:03d}" for n in i['evidence_global_turns'])&set(row['ranked'][method][:10])) for i in node['gold_intents'])
            coverage.append({'node':key,'method':method,'all_evidence_intents':complete,
                             'any_evidence_intents':any_hit,'gold_count':len(node['gold_intents'])})
    output={'phase':result['phase'],'dataset_sha256':result['dataset_sha256'],'model':result['model'],
            'expected_cases':len(dataset['cases']),'expected_nodes':len(nodes),'retrieved_nodes':len(rows),
            'prediction_failures':result['failures'],
            'judge_errors':[{'id':r['id'],'error':r['judge_error']} for r in rows.values() if 'judge_error' in r],
            'unique_tagged_texts':len(cache['tags']),'summary':{},'r10_regressions':errors,'r10_improvements':improvements,
            'evidence_coverage':coverage,'limitations':result['limitations']}
    for mode in ['all','long','cross']:
        for cutoff in [None,30,60]:
            name=mode+('' if cutoff is None else '_t'+str(cutoff))
            target=[(c,n) for c,n in nodes.values() if (mode=='all' or c['mode']==mode) and (cutoff is None or n['at_global_turn']==cutoff)]
            selected=[rows[n['id']] for _,n in target if n['id'] in rows]
            group={'expected_nodes':len(target),'retrieved_nodes':len(selected),'arms':{}}
            for method in ['two_route','three_route']:
                retrieval={}
                for k in [3,5,10]:
                    retrieval[str(k)]={metric:mean([r['retrieval'][method][str(k)][metric] for r in selected]) for metric in ['recall','precision','ndcg']}
                scored=[r['intent_scores'][method] for r in selected if 'intent_scores' in r]
                tp=sum(x['tp'] for x in scored);fp=sum(x['fp'] for x in scored);fn=sum(x['fn'] for x in scored)
                expected_gold=sum(len(n['gold_intents']) for _,n in target)
                negatives=[r for r in selected if not r['gold_intents'] and method in r['predictions']]
                group['arms'][method]={'retrieval':retrieval,
                   'intent':{'scored_nodes':len(scored),'tp':tp,'fp':fp,'fn':fn,'expected_gold':expected_gold,
                             'micro_recall_scored':tp/(tp+fn) if tp+fn else None,
                             'micro_precision_scored':tp/(tp+fp) if tp+fp else None,
                             'micro_f1_scored':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                             'recall_all_expected_lower_bound':tp/expected_gold if expected_gold else None,
                             'exact_nodes':sum(x['exact'] for x in scored),
                             'negative_predicted_nodes':len(negatives),
                             'negative_false_alarm_nodes':sum(bool(r['predictions'][method]) for r in negatives)}}
            rg=defaultdict(list);ig=defaultdict(list)
            for r in selected:
                rg[r['scenario_id']].append(r['retrieval']['three_route']['10']['recall']-r['retrieval']['two_route']['10']['recall'])
                if r['gold_intents'] and 'intent_scores' in r:
                    ig[r['scenario_id']].append(r['intent_scores']['three_route']['recall']-r['intent_scores']['two_route']['recall'])
            group['paired_r10_delta']=bootstrap(rg);group['paired_intent_recall_delta']=bootstrap(ig)
            group['wins_ties_losses_r10']={'wins':sum(r['retrieval']['three_route']['10']['hits']>r['retrieval']['two_route']['10']['hits'] for r in selected),
                                         'ties':sum(r['retrieval']['three_route']['10']['hits']==r['retrieval']['two_route']['10']['hits'] for r in selected),
                                         'losses':sum(r['retrieval']['three_route']['10']['hits']<r['retrieval']['two_route']['10']['hits'] for r in selected)}
            output['summary'][name]=group
    return output


def report(summary):
    fmt=lambda x:'—' if x is None else f'{100*x:.1f}%'
    lines=['# 两路与三路召回及下游意图识别：冻结诊断实验','',
       '数据为有意覆盖情境补漏的合成开发诊断集，非独立自然对话测试。长会话和跨会话各 30 例，每例两个评测节点，共 120 个节点。事件单元为预切分用户原话；本轮没有执行生产事件抽取或图扩展。','',
       f"模型请求标识：`{summary['model']}`。已取得检索结果 {summary['retrieved_nodes']}/{summary['expected_nodes']} 个节点。预测失败 {len(summary['prediction_failures'])}，评审错误 {len(summary['judge_errors'])}。",'',
       '## 相同 Top-K 的事件召回','',
       '| 范围 | 两路 R@3 | 三路 R@3 | 两路 R@5 | 三路 R@5 | 两路 R@10 | 三路 R@10 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,label in [('long','长会话'),('cross','跨会话'),('all','全部')]:
        arms=summary['summary'][mode]['arms'];cells=[]
        for k in ['3','5','10']:
            cells.extend(fmt(arms[m]['retrieval'][k]['recall']) for m in ['two_route','three_route'])
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','## 相同 Top10 历史事件阅读器的意图识别','',
       '| 范围 | 两路意图 Recall | 三路意图 Recall | 两路 Precision | 三路 Precision | 两路 F1 | 三路 F1 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,label in [('long','长会话'),('cross','跨会话'),('all','全部')]:
        arms=summary['summary'][mode]['arms'];cells=[]
        for metric in ['micro_recall_scored','micro_precision_scored','micro_f1_scored']:
            cells.extend(fmt(arms[m]['intent'][metric]) for m in ['two_route','three_route'])
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    overall=summary['summary']['all'];delta=overall['paired_r10_delta']
    if delta:
        lines+=['',f"按人物聚类的 R@10 差值（三路减两路）为 {100*delta['mean_difference']:.2f} 个百分点，bootstrap 95% 区间 [{100*delta['percentile_95ci'][0]:.2f}, {100*delta['percentile_95ci'][1]:.2f}] 个百分点。该区间只描述当前固定开发诊断集。"]
    lines+=['',f"节点 R@10 胜/平/负：{overall['wins_ties_losses_r10']}。没有因三路持平或退步而删除样本。",'',
      '表中意图指标为取得语义评审的节点；完整汇总另给所有预定参考意图作为分母的 recall 保守下界。若存在失败或评审错误，不能只用成功节点分数宣称整体收益。','',
      '## 口径与限制','',
      '- 两路为当前 0.75 语义 + 0.25 词项 Jaccard；三路为 0.8 两路 + 0.2 固定 scene/activity 字段重合。测试前冻结参数。',
      '- 标签由模型仅看文本与固定枚举填写，不看 gold。查询与事件分开标注。两组阅读器均只接收各自 Top10 事件，不给三路额外标签或原始对话。',
      '- 另存候选并集补漏数据；它可能有更多候选，不作为相同预算更优的证据。',
      '- 意图匹配是同模型辅助的一对一语义与证据评审，未独立人工复核。召回命中不等于用户接受该意图。',
      '- v4 开发事件被复用，查询按轨迹编写；两种会话模式和两个节点配对依赖。字段错误、模板化和重复语料限制泛化解释。',
      '- 长/跨会话使用配对文字，本轮无时间检索或生产图谱构建，初始召回的差异不应归因于跨会话推理能力。',
      '- 每组最多十个完整事件，但 token 数未严格相等；不声称总成本更低。执行并发为 1。','',
      '[实验协议](docs/recall-ablation-v1.md) · [冻结数据](data/recall_dataset_v1.json) · [完整结果](data/recall_v1_results.json) · [指标与错误](data/recall_v1_summary.json) · [标签及向量](data/recall_v1_cache.json) · [调用记录](data/recall_v1_model_calls.jsonl)','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--input',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    raw=Path(a.dataset).read_bytes();result=json.loads(Path(a.input).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==result['dataset_sha256']
    summary=summarize(json.loads(raw),result,json.loads(Path(a.cache).read_bytes()))
    Path(a.output).write_bytes((json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode())
    Path(a.report).write_bytes(report(summary).encode())
    print(json.dumps({'phase':summary['phase'],'nodes':summary['retrieved_nodes'],'summary':summary['summary']['all']},ensure_ascii=False))
