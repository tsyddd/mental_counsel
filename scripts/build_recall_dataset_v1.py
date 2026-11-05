"""Freeze checkpointed retrieval diagnostics before model tagging or ranking.

Reuses development material; deliberately probes contextual evidence recall.
Event units are user-authored turns, not production-extracted events.
"""
import copy
import hashlib
import json
from pathlib import Path
from build_intent_dataset_v3 import SCENARIOS

QUERIES = {
 'study_extra_classes':'最近学习总是占满我的一天，越忙越没有精神，想到后面的安排就发愁',
 'study_exam_schedule':'复习时我一直在往计划里塞东西，结果状态越来越差',
 'study_committee':'班里交给我的事和自己的学习挤在一起，我又有点接不住了',
 'peer_canteen':'宿舍里一起出门吃饭那件事，我还是会犹豫，之后又有点失落',
 'peer_club':'社团里大家一起做事情的时候，我还是站在旁边，不知道怎么参与',
 'peer_group_project':'和同组同学合作的时候，我明明有想法却总没说出来',
 'parent_career':'毕业去哪里这件事，我自己的打算和家里的期待还是没对上',
 'parent_privacy':'在家里个人东西被碰到的事，我还是介意，也不知道怎么跟妈妈说',
 'parent_emotion':'实习不顺的时候，面对爸妈我还是说不出真实的难过',
 'sleep_phone':'晚上明明想休息却又停不下手机，第二天的状态还是不好',
 'sleep_weekend_shift':'一到周末我的作息就往后跑，接下来要早起的时候又很难受',
 'sleep_bed_work':'我发现躺在床上还是会想到那些资料，到了该睡的时候反而清醒',
 'work_weekend':'休息日又像随时待命，自己的安排总被同事的消息打断',
 'work_extra_tasks':'别人拜托的工作又堆到我这里，自己的事情总是被拖到后面',
 'work_leave':'人虽然在放假，心里还是挂着单位那些联系，没能真正歇下来',
 'boundary_roommate':'室友用我东西这件事，我还是有些不舒服，却一直没说清楚',
 'boundary_favors':'朋友一开口我就又挪掉自己的安排，回来才觉得为难',
 'boundary_money':'朋友那笔钱和后来的请求还压在心里，我自己的开支也要顾',
}


def build():
    root=Path(__file__).resolve().parents[1]
    source=(root/'data/intent_dataset_v4.json').read_bytes()
    data=json.loads(source)
    cases=[]
    for original in data['cases']:
        case=copy.deepcopy(original)
        case.pop('requires_new_runner',None)
        active=case.pop('gold_intents')
        negative=case.pop('negative_tracks')
        tracks=active+negative
        nodes=[]
        for cutoff in [30,60]:
            if active:
                query='；另外，'.join(QUERIES[a['track_id']] for a in active)
                query+='。结合我之前说过的经历，你觉得我现在有哪些需要澄清或调整的方向？'
                relevant=[n for a in active for n in a['evidence_global_turns'] if n<cutoff]
            else:
                query='前面提到的那些经历，现在有哪些是我本人还需要继续处理的？请结合之前的情况判断，别替我增加没有表达的目标。'
                relevant=[n for a in negative for n in a['evidence_global_turns'] if n<cutoff]
            turn=case['turns'][cutoff-1]
            turn['user']=query
            turn['assistant']='我会结合你之前说过的情况，区分经历、当前需要和已经结束的事情。'
            gold=[]
            for a in active:
                g=copy.deepcopy(a)
                g['evidence_global_turns']=[n for n in g['evidence_global_turns'] if n<cutoff]
                g['supporting_event_groups']=[[n] for n in g['evidence_global_turns']]
                gold.append(g)
            nodes.append({'id':case['id']+f'_t{cutoff}','at_global_turn':cutoff,
                          'query':query,'query_kind':'multi_intent' if active else 'negative_status_check',
                          'gold_relevant_event_ids':[f'e{n:03d}' for n in sorted(relevant)],
                          'gold_intents':gold,
                          'negative_tracks':[dict(a,evidence_global_turns=[n for n in a['evidence_global_turns'] if n<cutoff]) for a in negative]})
        case['evaluation_nodes']=nodes
        # Every pre-query user turn is a source-preserving atomic event unit.
        # Do not use manually selected gold tracks as the retrieval corpus.
        case['events']=[{'id':f"e{t['global_turn']:03d}",'content':t['user'],
                         'session':t['session'],'turn':t['turn'],'global_turn':t['global_turn'],
                         'sent_at':t['sent_at'],'source_message_key':t['user_message_key']}
                        for t in case['turns'] if t['global_turn'] not in [30,60]]
        cases.append(case)
    assert len(cases)==60 and sum(len(c['evaluation_nodes']) for c in cases)==120
    for c in cases:
        for node in c['evaluation_nodes']:
            assert all(int(e[1:])<node['at_global_turn'] for e in node['gold_relevant_event_ids'])
            assert len(node['gold_relevant_event_ids'])==len(set(node['gold_relevant_event_ids']))
            assert all(e['global_turn']<node['at_global_turn'] for e in c['events'] if e['id'] in node['gold_relevant_event_ids'])
    return {'version':'recall-v1','synthetic':True,'source_v4_sha256':hashlib.sha256(source).hexdigest(),
            'n_scenarios':30,'n_cases':60,'n_checkpoints':120,
            'case_modes':'30 long 1x60; 30 cross 2x30; each has checkpoints at 30 and 60',
            'event_unit':'one pre-query user turn; source-preserving presegmented diagnostic; not production extractor output',
            'scope':'initial retrieval and identical downstream intent extraction; graph expansion disabled in both arms',
            'limitations':['reuses development v4 events','targeted contextual-recall diagnostic, not random natural conversations',
                           'queries authored with scenario knowledge; no independent human relevance review',
                           'paired modes and two checkpoints are dependent','no claim of clinical efficacy'],
            'scene_values':['学业','家庭','同伴','工作','亲密关系','日常生活','其他','未知'],
            'activity_values':['学习与课程','任务安排','人际沟通','集体参与','休息睡眠','物品使用','金钱往来','其他','未知'],
            'frozen_config':{'concurrency':1,'ks_weights':{'semantic':0.75,'keyword':0.25},
                             'ksc_weights':{'two_route':0.8,'scene_activity':0.2},
                             'field_score':'mean of scene/activity Jaccard; unknown/other ignored',
                             'ks':[3,5,10],'intent_context_k':10,'bootstrap_unit':'scenario_id',
                             'union_diagnostic':'two-route top10 union field top5; not equal-budget comparison'},
            'cases':cases}


if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'data/recall_dataset_v1.json'
    p.write_bytes((json.dumps(build(),ensure_ascii=False,indent=2)+'\n').encode())
    print('frozen sha256',hashlib.sha256(p.read_bytes()).hexdigest())
    print('60 cases, 120 checkpoints; 30 or 60 rounds; 2 checkpoints per case')
