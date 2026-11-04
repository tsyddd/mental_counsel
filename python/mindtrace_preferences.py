"""Bounded user.md is the authoritative preference store; pending proposals live in SQL."""
import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from .mindtrace_logic import normalize

MAX_PREFS = 12
MAX_MARKDOWN = 1200


def user_path(user_id):
    uid = str(uuid.UUID(str(user_id)))
    return Path(os.getenv('USER_DATA_ROOT', '/data/users')) / uid / 'user.md'


@contextmanager
def locked_user(user_id):
    path = user_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (path.parent / '.user.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield path


def read_preferences(user_id):
    path = user_path(user_id)
    if not path.exists():
        return []
    return [line[2:].strip() for line in path.read_text('utf-8').splitlines()
            if line.startswith('- ')][:MAX_PREFS]


def write_preferences(path, items):
    unique = {}
    for item in items:
        text = ' '.join(str(item).split())[:150]
        if text:
            unique.setdefault(normalize(text), text)
    items = list(unique.values())
    markdown = '# 用户偏好\n\n' + ''.join('- '+p+'\n' for p in items)
    if len(items) > MAX_PREFS or len(markdown) > MAX_MARKDOWN:
        raise ValueError('偏好空间已满，请先删除或合并偏好')
    temporary = path.with_name('user.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(markdown, encoding='utf-8')
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return items


def apply_preferences(user_id, add=(), remove=()):
    with locked_user(user_id) as path:
        old = read_preferences(user_id)
        removing = {normalize(x) for x in remove}
        items = [x for x in old if normalize(x) not in removing]
        return write_preferences(path, items + list(add))


def tombstones(user_id, add=None):
    with locked_user(user_id) as path:
        file = path.parent / '.rejected-preferences.json'
        values = json.loads(file.read_text('utf-8')) if file.exists() else []
        if add:
            values = list(dict.fromkeys(values + [normalize(add)]))[-100:]
            temp = file.with_suffix('.tmp')
            temp.write_text(json.dumps(values, ensure_ascii=False), 'utf-8')
            temp.chmod(0o600)
            os.replace(temp, file)
        return values
