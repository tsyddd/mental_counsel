"""Behavioral regressions: isolation, confirmation, fixed windows, queue waiting.
Run in the API container: python -m unittest discover -s /tests -v
"""
import json
import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

from platform_app import mindtrace as mt
from platform_app.mindtrace_logic import bounded_window,validate_events,valid_intent
from platform_app.mindtrace_preferences import apply_preferences,read_preferences,tombstones,user_path


class Policies(unittest.TestCase):
    def test_window_budget_and_provenance(self):
        event={'id':'e1','source_session':'s2','start_turn':7,'end_turn':8,'content':'学习很累'*80,'occurred_at':'上周'}
        intent={'id':'i1','source_session':'s1','extracted_turn':12,'status':'tentative','label':'协商学习安排'}
        window=bounded_window([intent],[event],600)
        self.assertLessEqual(len(window['content']),600)
        self.assertIn('第12轮形成',window['content'])
        self.assertIn('第7-8轮',window['content'])
        self.assertEqual(bounded_window([intent],[event],10)['event_ids'],[])

    def test_only_user_grounding(self):
        messages=[{'id':'u1','role':'user','turn_no':2},{'id':'a1','role':'assistant','turn_no':2}]
        payload={'events':[{'summary':'x','subject':'user','source_ids':['a1']},
                           {'summary':'y','subject':'user','source_ids':['u1']},
                           {'summary':'z','subject':'friend','source_ids':['u1']}]}
        result=validate_events(payload,messages)
        self.assertEqual([x['content'] for x in result],['y'])
        self.assertEqual(result[0]['start_turn'],2)

    def test_intent_cannot_self_confirm_or_invent_evidence(self):
        op={'action':'create','label':'x','support_event_ids':['e1'],'status':'confirmed'}
        self.assertFalse(valid_intent(op,{'e1'},set()))
        op['status']='tentative';self.assertTrue(valid_intent(op,{'e1'},set()))
        op['support_event_ids']=['other_user_event'];self.assertFalse(valid_intent(op,{'e1'},set()))

    def test_preferences_file_dedup_limits_and_rejection(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'USER_DATA_ROOT':root}):
            user=str(uuid.uuid4())
            apply_preferences(user,add=['一次只问一个问题。','一次只问一个问题'])
            self.assertEqual(len(read_preferences(user)),1)
            apply_preferences(user,remove=['一次只问一个问题'],add=['先听完再建议'])
            self.assertEqual(read_preferences(user),['先听完再建议'])
            tombstones(user,'一次只问一个问题')
            self.assertIn('一次只问一个问题',tombstones(user))
            with self.assertRaises(ValueError):
                apply_preferences(user,add=[str(i) for i in range(13)])
            self.assertEqual(read_preferences(user),['先听完再建议'])
            self.assertEqual(user_path(user).name,'user.md')

    def test_preference_conflict_never_overwrites_before_confirmation(self):
        user=str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'USER_DATA_ROOT':root}),\
             patch.object(mt,'ask',return_value={'relation':'conflict','old':'简短回复','explicit':True}),\
             patch.object(mt,'add_confirmation') as confirmation:
            apply_preferences(user,add=['简短回复'])
            mt.process_preferences(user,'s1',[{'text':'详细回复','source_id':'u1'}],[{'id':'u1','role':'user'}])
            self.assertEqual(read_preferences(user),['简短回复'])
            confirmation.assert_called_once()

    def test_chat_waits_and_does_not_answer_before_memory_ready(self):
        jobs=[[{'id':'j','status':'running'}],[],[]]
        from contextlib import contextmanager
        @contextmanager
        def locked(*args,**kwargs):yield True
        with patch.object(mt,'pending_jobs',side_effect=jobs),patch.object(mt,'user_lock',locked),\
             patch.object(mt.time,'sleep'),patch.object(mt,'answer_locked',return_value=iter(['ANSWER'])) as answer:
            stream=mt.chat_stream('u','s','新问题')
            first=next(stream)
            self.assertIn('event: status',first)
            answer.assert_not_called()
            self.assertEqual(next(stream),'ANSWER')

    def test_failed_memory_blocks_stale_reply(self):
        with patch.object(mt,'pending_jobs',return_value=[{'status':'failed'}]),\
             patch.object(mt,'answer_locked') as answer:
            self.assertIn('event: error',next(mt.chat_stream('u','s','q')))
            answer.assert_not_called()

    def test_cross_user_preference_isolation(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'USER_DATA_ROOT':root}):
            a,b=str(uuid.uuid4()),str(uuid.uuid4())
            apply_preferences(a,add=['喜欢画画'])
            self.assertEqual(read_preferences(b),[])


if __name__=='__main__':unittest.main()
