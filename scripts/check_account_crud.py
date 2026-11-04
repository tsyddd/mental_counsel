import json
import uuid
from pathlib import Path
import requests

h=requests.Session();base='http://gateway:8080'
r=h.post(base+'/api/auth/register',json={'email':uuid.uuid4().hex+'@example.test',
 'password':uuid.uuid4().hex,'display_name':'CRUD测试'},timeout=20);r.raise_for_status()
checks={'create':r.status_code==200}
r=h.get(base+'/api/auth/me',timeout=20);checks['read']=r.status_code==200
r=h.put(base+'/api/auth/me',json={'display_name':'已更新'},timeout=20)
checks['update']=r.status_code==200 and r.json().get('display_name')=='已更新'
r=h.delete(base+'/api/auth/me',timeout=20);checks['delete']=r.status_code==200
checks['deleted_session_invalid']=h.get(base+'/api/auth/me',timeout=20).status_code==401
Path('/checkpoints/mindtrace-crud-check.json').write_text(json.dumps(checks),encoding='utf-8')
print(json.dumps(checks),flush=True)
raise SystemExit(0 if all(checks.values()) else 1)
