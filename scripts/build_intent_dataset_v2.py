"""Freeze authored event→intent scenarios before running any model predictions.

Paired long/cross variants share a scenario and are NOT independent cases.
This is an engineering diagnostic, not a clinical or held-out generalization set.
"""
import hashlib
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path

BASE = {
 'study_load': {'gold':'study_rest_balance','events':[
    '这周有三天写完作业已经凌晨一点，早读时我困得撑不住。',
    '每天放学后先去上补习班，回到家九点多，还要完成学校作业。',
    '原本周日下午有半天休息，后来安排了新的辅导课，连续两周都没有空下来。',
    '妈妈觉得成绩还没起色，昨天说下周再加一门课；我想到晚上又得更晚睡就发愁。']},
 'peer_connection': {'gold':'peer_participation','events':[
    '上周班级分组做海报，我在旁边等着，最后没有加入，回家还在想那次活动。',
    '周三午休同学们一起玩游戏，我站在门口看了一会儿就走了，下午总觉得空落落的。',
    '昨天社团又分了组，我拿着材料不知道站哪儿，最后一个人完成，看到别人一起笑有点难受。',
    '今天一位同学叫我一起做展示，我很高兴但没答应，担心自己插进去会让人不自在。']},
 'third_party': {'gold':None,'events':[
    '同桌告诉我，她这周三天写作业到凌晨一点，早读很困；说的是她，不是我的情况。',
    '她说自己放学后补习，回家已经九点。我自己的作业安排正常，晚上能休息。',
    '同桌的妈妈给她周日又排了一门课。她来找我讲了这件事，我只是转述。',
    '她家里还想加课，听完我有些感慨；但我目前没有这个困扰，也不需要为自己调整安排。']},
 'normal_routine': {'gold':None,'events':[
    '我周二去上了自己选的绘画课，回来把作业做完，十点前就睡了，感觉挺轻松。',
    '周四又上了一次绘画课，老师讲了颜色搭配，我很喜欢，回家时间和以前一样。',
    '周末家里问我要不要增加一次课，我觉得当前安排正好，家里也尊重我的决定。',
    '这周还是照原来的安排，作业和休息都有时间，没有需要改变或者继续讨论的问题。']},
}
DISTRACTORS = [
 '今天午饭我和同学聊了一会儿学校的新图书角。',
 '体育课练了接力跑，大家轮流计时，我记得自己的成绩比上次快一点。',
 '昨天我整理了书包，把旧练习纸放进了资料夹。',
 '科学课做了一个关于水的实验，我记下了步骤，觉得很有意思。',
 '最近班级换了黑板报，画的是秋天的校园。',
 '我周末和家人散步，看到了路边的银杏树，聊了些日常小事。',
 '今天课间有人分享了一本科普书，我翻了几页关于海洋的内容。',
 '我把明天要用的文具都准备好了，放在书桌左边。',
]
REPLIES = ['我听到了。你可以按自己的节奏继续说。','这件事发生在怎样的情境里？',
           '谢谢你讲这些。还有哪些细节是你想补充的？','我会先听你把经历说完。']


def build():
 cases=[]
 for name,base in BASE.items():
  for mode in ['long','cross']:
   turns=[]
   for n in range(1,25):
    text=base['events'][[2,9,17,22].index(n)] if n in [2,9,17,22] else DISTRACTORS[(n-1)%len(DISTRACTORS)]
    session=(n-1)//8+1 if mode=='cross' else 1
    local=(n-1)%8+1 if mode=='cross' else n
    at=datetime(2026,10,1,19,0,tzinfo=timezone(timedelta(hours=8)))+timedelta(days=(session-1)*7,minutes=local*2)
    turns.append({'session':session,'turn':local,'global_turn':n,'sent_at':at.isoformat(),
                  'user':text,'assistant':REPLIES[(n-1)%len(REPLIES)]})
   cases.append({'id':name+'_'+mode,'scenario_id':name,'mode':mode,'gold_category':base['gold'],
                 'gold_has_intent':base['gold'] is not None,'key_global_turns':[2,9,17,22],
                 'turns':turns})
 return {'version':2,'synthetic':True,'authorship':'authored diagnostic scenarios; no independent human/clinical review',
         'n_independent_scenarios':len(BASE),'n_paired_variants':len(cases),
         'protocol':'all timestamped user AND assistant messages up to each final checkpoint; paired variants grouped',
         'cases':cases}


if __name__=='__main__':
 path=Path(__file__).resolve().parents[1]/'data/intent_dataset_v2.json'
 path.parent.mkdir(exist_ok=True)
 raw=json.dumps(build(),ensure_ascii=False,indent=2)+'\n'
 # Match the initially frozen Windows artifact byte-for-byte on every platform.
 path.write_bytes(raw.replace('\n','\r\n').encode('utf-8'))
 print(path.name,hashlib.sha256(path.read_bytes()).hexdigest())
