"""Offline audit and report for the actual graph/full-history comparison."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(dataset,result):
    expected={n['id']:(c,n) for c in dataset['cases'] for n in c['evaluation_nodes']}
    assert len({r['id'] for r in result['results']})==len(result['results'])
    source_checks=0
    for row in result['results']:
        case,node=expected[row['id']];cutoff=node['at_global_turn']
        assert row['gold_intents']==node['gold_intents']
        messages=row['full_history_input'];assert len(messages)==2*cutoff-1
        assert messages[-1]['role']=='user' and messages[-1]['content']==node['query']
        assert messages[-1]['global_turn']==cutoff
        assert all(m['global_turn']<cutoff for m in messages[:-1])
        for m in messages:
            source=case['turns'][m['global_turn']-1]
            assert m['content']==source[m['role']] and m['session_number']==source['session']
        mids={m['id']:m for m in messages if m['role']=='user'}
        events={e['id']:e for e in row['graph_events']}
        for event in events.values():
            assert event['user_id']==row['user_id']
            assert set(event['source_ids'])<=mids.keys()
            sources=[mids[mid] for mid in event['source_ids']]
            assert all(m['session_id']==event['source_session'] for m in sources)
            assert min(m['turn'] for m in sources)==event['start_turn']
            assert max(m['turn'] for m in sources)==event['end_turn']
            assert event['eligible_history_support']==any(m['global_turn']<cutoff for m in sources)
            source_checks+=1
        assert len(row['predictions']['graph'])==len(row['graph_intents'])
        for pred,original in zip(row['predictions']['graph'],row['graph_intents']):
            assert pred['description']==original['label']
            assert pred['source_event_ids']==original['support_event_ids']
            assert original['status'] in ['tentative','confirmed']
            assert set(pred['source_event_ids'])<=events.keys()
        for intent in row['predictions'].get('full_history',[]):
            assert set(intent['source_message_ids'])<=set(mids)-{messages[-1]['id']}
        if 'scores' in row:
            gids={g['intent_id'] for g in node['gold_intents']}
            for method in ['graph','full_history']:
                score=row['scores'][method];matches=score['matches'];pids={p['id'] for p in row['predictions'][method]}
                assert len({m['prediction_id'] for m in matches})==len(matches)
                assert len({m['gold_id'] for m in matches})==len(matches)
                assert all(m['prediction_id'] in pids and m['gold_id'] in gids for m in matches)
                assert score['tp']==len(matches) and score['fp']==len(pids)-len(matches) and score['fn']==len(gids)-len(matches)
    summary={}
    for mode in ['all','long','cross']:
        for cut in [None,30,60]:
            name=mode+('' if cut is None else '_t'+str(cut))
            targets=[n for c,n in expected.values() if (mode=='all' or c['mode']==mode) and (cut is None or n['at_global_turn']==cut)]
            ids={n['id'] for n in targets};rows=[r for r in result['results'] if r['id'] in ids]
            scored=[r for r in rows if 'scores' in r]
            item={'expected_nodes':len(targets),'exported_nodes':len(rows),'scored_nodes':len(scored),'arms':{}}
            for method in ['graph','full_history']:
                tp=sum(r['scores'][method]['tp'] for r in scored);fp=sum(r['scores'][method]['fp'] for r in scored);fn=sum(r['scores'][method]['fn'] for r in scored)
                gold=sum(len(n['gold_intents']) for n in targets)
                item['arms'][method]={'tp':tp,'fp':fp,'fn':fn,'expected_gold':gold,
                   'micro_recall_scored':tp/(tp+fn) if tp+fn else None,
                   'micro_precision_scored':tp/(tp+fp) if tp+fp else None,
                   'micro_f1_scored':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                   'recall_all_expected_lower_bound':tp/gold if gold else None}
            summary[name]=item
    return {'dataset_sha256':result['dataset_sha256'],'model':result['model'],'phase':result['phase'],
            'source_events_checked':source_checks,'scope':result['scope'],
            'graph_implementation':result['graph_implementation'],'summary':summary,
            'failures':result['failures'],'judge_errors':[{'id':r['id'],'error':r['judge_error']} for r in result['results'] if 'judge_error' in r],
            'limitations':result['limitations']}


def report(summary):
    fmt=lambda value:'—' if value is None else f'{100*value:.2f}%'
    lines=['# MindTrace 意图识别主对比：事件→意图图谱 vs 完整会话','',
       '主方法读取实际生产 Worker 维护的活跃意图节点，不再让模型读取 Top10 事件生成答案。完整历史基线读取截至当前用户发言的全部带时间、角色和会话边界的双方对话。','',
       f"请求模型：`{summary['model']}`。实际图谱版本：{summary['graph_implementation']}。",'',
       '| 范围 | 已评分/预定节点 | 图谱 Recall | 完整历史 Recall | 图谱 Precision | 完整历史 Precision | 图谱 F1 | 完整历史 F1 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,label in [('long','长会话'),('cross','跨会话'),('all','全部')]:
        s=summary['summary'][mode];cells=[]
        for metric in ['micro_recall_scored','micro_precision_scored','micro_f1_scored']:
            cells.extend(fmt(s['arms'][m][metric]) for m in ['graph','full_history'])
        lines.append('| '+label+f" | {s['scored_nodes']}/{s['expected_nodes']} | "+' | '.join(cells)+' |')
    lines+=['',f"回放/预测失败记录 {len(summary['failures'])}，评分错误 {len(summary['judge_errors'])}。已核对 {summary['source_events_checked']} 条检查点事件来源。",'',
       '上表只计算成功评分节点，完整汇总另列所有预定 gold 分母的 Recall 下界。存在失败或缺失时不能把成功子集当作完整成绩。', '',
       '本批为复用开发素材的合成诊断集，两个会话模式和两个节点配对依赖；尚未独立人工审核。两边基础模型相同，但累计调用预算不同。候选识别不等于用户确认或临床效果。','',
       '[评测协议](docs/intent-main-graph-vs-full-v1.md) · [冻结数据](data/recall_dataset_v1.json) · [完整结果](data/graph_full_v1_results.json) · [汇总与错误](data/graph_full_v1_summary.json) · [基线及评分调用](data/graph_full_v1_calls.jsonl)','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    raw=Path(a.dataset).read_bytes();result=json.loads(Path(a.input).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==result['dataset_sha256']
    value=summarize(json.loads(raw),result)
    Path(a.output).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
    Path(a.report).write_bytes(report(value).encode())
    print(json.dumps(value['summary']['all']))
