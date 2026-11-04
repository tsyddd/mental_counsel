import json
import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

from platform_app import mindtrace as mt
from platform_app.mindtrace_preferences import apply_preferences,read_preferences


class DatabaseIntegration(unittest.TestCase):
    def setUp(self):
        self.user=str(uuid.uuid4());self.session=str(uuid.uuid4())
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{'USER_DATA_ROOT':self.tmp.name});self.env.start()
        with mt.conn() as db:
            db.execute("INSERT INTO app_users(id,email,display_name,password_hash) VALUES (%s,%s,'test','disabled')",(self.user,self.user+'@example.test'))
            db.execute('INSERT INTO sessions(id,user_id) VALUES (%s,%s)',(self.session,self.user))

    def tearDown(self):
        with mt.conn() as db:
            db.execute('DELETE FROM mt_snapshots WHERE user_id=%s',(self.user,))
            db.execute('DELETE FROM mt_confirmations WHERE user_id=%s',(self.user,))
            db.execute('DELETE FROM sessions WHERE user_id=%s',(self.user,))
            db.execute('DELETE FROM app_users WHERE id=%s',(self.user,))
        self.env.stop();self.tmp.cleanup()

    def test_durable_batch_idempotency(self):
        one=mt.enqueue_batch(self.user,self.session,6)
        two=mt.enqueue_batch(self.user,self.session,6)
        self.assertEqual(one,two)
        self.assertEqual(len(mt.all_rows('SELECT id FROM mt_jobs WHERE session_id=%s',(self.session,))),1)

    def test_preference_confirmation_persists_only_selected_value(self):
        apply_preferences(self.user,add=['简短回复'])
        mt.add_confirmation(self.user,self.session,'preference',{'text':'详细回复','old':'简短回复'},'要改吗？')
        row=mt.all_rows('SELECT id FROM mt_confirmations WHERE user_id=%s',(self.user,))[0]
        self.assertEqual(read_preferences(self.user),['简短回复'])
        mt.confirm(self.user,uuid.UUID(row['id']),mt.Decision(accept=True))
        self.assertEqual(read_preferences(self.user),['详细回复'])
        self.assertEqual(mt.all_rows('SELECT status FROM mt_confirmations WHERE id=%s',(row['id'],))[0]['status'],'accepted')

    def test_intent_user_confirmation_and_wrong_scope(self):
        iid=str(uuid.uuid4())
        with mt.conn() as db:
            db.execute("INSERT INTO mt_intents(id,user_id,label,category,source_session,extracted_turn) VALUES (%s,%s,'调整学习安排','study_rest_balance',%s,6)",(iid,self.user,self.session))
            db.execute('''INSERT INTO mt_snapshots(user_id,session_id,processed_turn,content,event_ids,intent_ids)
                VALUES (%s,%s,6,%s,'[]',%s)''',(self.user,self.session,
                '[事件 e | 会话 s 第2轮] 证据\n[意图 '+iid+' | 会话 s 第6轮形成 | tentative] 调整学习安排',json.dumps([iid])))
        mt.add_confirmation(self.user,self.session,'intent',{'intent_id':iid},'愿意讨论吗？')
        cid=mt.all_rows('SELECT id FROM mt_confirmations WHERE user_id=%s',(self.user,))[0]['id']
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):mt.confirm(str(uuid.uuid4()),uuid.UUID(cid),mt.Decision(accept=True))
        with self.assertRaises(HTTPException):mt.confirm(self.user,uuid.UUID(cid),mt.Decision(accept=True,expected_label='过时的目标'))
        mt.confirm(self.user,uuid.UUID(cid),mt.Decision(accept=False))
        self.assertEqual(mt.load_intents(self.user)[0]['status'],'rejected')
        snapshot=mt.all_rows('SELECT content,intent_ids FROM mt_snapshots WHERE user_id=%s',(self.user,))[0]
        self.assertIn('证据',snapshot['content'])
        self.assertNotIn('[意图',snapshot['content'])
        self.assertEqual(snapshot['intent_ids'],[])

    def test_pgvector_storage_and_fixed_window(self):
        eid=str(uuid.uuid4());vec=[1.0]+[0.0]*383
        with mt.conn() as db:
            db.execute('''INSERT INTO mt_events(id,user_id,content,source_ids,source_session,start_turn,end_turn,embedding)
                VALUES (%s,%s,'晚上补习后休息不足',%s,%s,2,4,%s::vector)''',
                (eid,self.user,'["synthetic-source"]',self.session,mt.vector_literal(vec)))
        with patch.object(mt,'embed',return_value=[vec]):
            window=mt.publish_window(self.user,self.session,6,[{'role':'user','content':'补习之后很累'}])
        self.assertIn(eid,window['event_ids']);self.assertIn('第2-4轮',window['content'])
        self.assertLessEqual(len(window['content']),mt.WINDOW_CHARS)
        self.assertEqual(mt.all_rows('SELECT processed_turn FROM mt_snapshots WHERE user_id=%s',(self.user,))[0]['processed_turn'],6)

    def test_one_message_can_contain_two_distinct_episodes(self):
        common={'source_ids':['same-message'],'subject':'user','start_turn':1,'end_turn':1}
        extracted=[{**common,'content':'周一补习到九点','occurred_at':'周一'},
                   {**common,'content':'周三补习到十点','occurred_at':'周三'}]
        vec=[1.0]+[0.0]*383
        with patch.object(mt,'embed',return_value=[vec,vec]),patch.object(mt,'ask',return_value={'duplicate_id':None}):
            added=mt.insert_events(self.user,self.session,extracted,[])
        self.assertEqual(len(added),2)
        self.assertEqual(len(mt.load_events(self.user)),2)


if __name__=='__main__':unittest.main()
