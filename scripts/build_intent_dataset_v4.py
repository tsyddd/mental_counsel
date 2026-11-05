"""Multi-intent diagnostic revision; gold is never a prediction input.

Reuses reviewed-in-development v3 event tracks, not independent held-out people.
No claim of human review or statistically independent new test observations.
"""
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from build_intent_dataset_v3 import SCENARIOS, DISTRACTORS, REPLIES

GOALS = [
 ('学习安排', '家长及课程安排', '调整补习与作业负担，保留睡眠和休息时间'),
 ('复习安排', '自己的复习计划', '调整过密的复习安排，恢复吃饭和睡眠时间'),
 ('班干部任务', '老师及班级活动安排', '协调活动任务与学习负担，为作业和休息留出时间'),
 ('同伴参与', '室友', '减少约饭时的回避，参与自己在意的同伴活动'),
 ('同伴参与', '社团同学', '减少社团活动中的回避，加入合作与互动'),
 ('同伴参与', '课程项目同组同学', '在小组合作中表达想法并参与讨论'),
 ('职业自主与家庭沟通', '父母', '厘清就业地点选择，并向父母表达与其期待不同的想法'),
 ('家庭隐私边界', '母亲', '表达日记和信件被翻看的感受，协商私人物品的隐私边界'),
 ('家庭情绪表达', '父母', '表达实习受挫的真实感受，并获得理解与倾听'),
 ('睡眠作息', '睡前手机使用', '减少睡前刷手机造成的晚睡与起床困难'),
 ('睡眠作息', '周末与工作日作息', '减少周末作息漂移造成的周一早起困难'),
 ('睡眠作息', '床与工作环境', '分离床上工作与就寝环境，改善入睡困难'),
 ('工作边界', '周末联系自己的同事', '协调非工作时间的响应安排，保留私人时间'),
 ('工作边界', '请求代做任务的同事', '协调额外任务分工，减少自己的加班与积压'),
 ('工作边界', '休假期间联系自己的同事', '按交接安排处理休假联络，保留休假时间'),
 ('私人关系边界', '室友', '协商借用和使用个人物品的边界'),
 ('私人关系边界', '提出帮忙请求的朋友', '协调帮忙与自己的安排，表达可接受的时间边界'),
 ('私人关系边界', '借钱的朋友', '沟通已有欠款和新的借款请求，保护自己的必要支出'),
]

# Cohort-compatible combinations. Twelve two-intent and twelve three-intent tracks.
COMBINATIONS = [
 (0,3),(0,7),(1,4),(1,8),(2,5),(2,7),
 (6,8),(6,15),(9,12),(10,13),(11,14),(16,17),
 (0,3,7),(0,4,8),(1,3,7),(1,5,8),(2,4,7),(2,5,8),
 (6,9,15),(6,10,16),(8,11,17),(9,12,15),(10,13,16),(11,14,17),
]
NEGATIVE_PAIRS = [(18,22),(19,25),(20,24),(21,23),(26,28),(27,29)]


def negative_kind(index):
    return ('third_party' if index < 22 else 'resolved' if index < 26
            else 'normal_satisfied' if index < 28 else 'explicitly_rejected')


def build():
    cases = []
    for index in range(30):
        positive = index < 24
        active = list(COMBINATIONS[index]) if positive else []
        # Every multi-intent case also contains a hard-negative trajectory.
        negatives = [18 + index % 12] if positive else list(NEGATIVE_PAIRS[index - 24])
        tracks = active + negatives
        rng = random.Random('mindtrace-v4:' + str(index))
        before = sorted(rng.sample(range(2,29), 2 * len(tracks)))
        after = sorted(rng.sample(range(32,59), 2 * len(tracks)))
        placements = {}
        annotations = []
        for slot, track in enumerate(tracks):
            name, _, events = SCENARIOS[track]
            positions = [before[slot], before[slot + len(tracks)],
                         after[slot], after[slot + len(tracks)]]
            for number, text in zip(positions, events):
                assert number not in placements
                placements[number] = text
            annotation = {'track_id': name, 'evidence_global_turns': positions,
                          'final_status': 'active_candidate' if track in active else negative_kind(track)}
            if track in active:
                domain, counterpart, goal = GOALS[track]
                annotation.update(intent_id='gold_' + name, subject='user', domain=domain,
                                  counterpart=counterpart, desired_change=goal,
                                  requires_user_confirmation=True,
                                  supporting_event_groups=[[n] for n in positions],
                                  minimum_required_fields=['counterpart','desired_change'],
                                  acceptable_paraphrases=True)
            else:
                annotation['exclusion_reason'] = negative_kind(track)
                annotation['note'] = 'final negative status does not imply never active at an earlier checkpoint'
            annotations.append(annotation)
        fillers = DISTRACTORS * 3
        rng.shuffle(fillers)
        for mode in ['long','cross']:
            turns = []
            for number in range(1,61):
                session = 1 if mode == 'long' else (number - 1) // 30 + 1
                local = number if mode == 'long' else (number - 1) % 30 + 1
                sent_at = datetime(2026,9,1,19,tzinfo=timezone(timedelta(hours=8))) + timedelta(
                    days=index * 2 + (session - 1) * 7, minutes=local * 2)
                turns.append({'session':session,'turn':local,'global_turn':number,
                              'user_message_key':f'u{number:03d}',
                              'sent_at':sent_at.isoformat(),
                              'user':placements.get(number,fillers[number-1]),
                              'assistant':REPLIES[(index+number) % len(REPLIES)]})
            gold = [a for a in annotations if a['final_status'] == 'active_candidate']
            excluded = [a for a in annotations if a['final_status'] != 'active_candidate']
            cases.append({'id':f'multi_{index+1:02d}_{mode}',
                          'scenario_id':f'multi_{index+1:02d}', 'mode':mode,
                          'gold_intents':gold, 'negative_tracks':excluded,
                          'evaluation_checkpoint_global_turn':60,
                          'gold_intent_count':len(gold),'turns':turns})
    for mode in ['long','cross']:
        selected = [c for c in cases if c['mode'] == mode]
        assert len(selected) == 30
        assert Counter(c['gold_intent_count'] for c in selected) == {2:12,3:12,0:6}
        for c in selected:
            assert min(Counter(t['session'] for t in c['turns']).values()) >= 30
            assert c['negative_tracks']
    for left,right in zip(cases[::2],cases[1::2]):
        assert [t['user'] for t in left['turns']] == [t['user'] for t in right['turns']]
    return {'version':4,'synthetic':True,'status':'frozen_authored_diagnostic_pending_independent_review',
            'n_scenarios':30,'n_paired_variants':60,
            'round_definition':'one user message and one scripted assistant response',
            'per_mode_intent_counts':{'two_intents':12,'three_intents':12,'no_active_intent':6},
            'source':'recombined v3 development event tracks; not independent held-out data',
            'prediction_input_fields':['session','turn','global_turn','sent_at','role','content','message_id'],
            'gold_visibility':'gold_intents and negative_tracks are evaluation-only; never passed to prediction',
            'scoring':'one-to-one semantic matching on actor, counterpart and desired change; report TP/FP/FN, micro/macro F1, negative false positives and exact-set accuracy separately',
            'requires_new_runner':True,'cases':cases}


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'data/intent_dataset_v4.json'
    path.write_bytes((json.dumps(build(),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('sha256',hashlib.sha256(path.read_bytes()).hexdigest())
    print('60 cases; 12 two-intent + 12 three-intent + 6 negative cases per mode')
