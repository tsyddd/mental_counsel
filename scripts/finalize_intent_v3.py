"""Wait for all 60 outcomes, then map descriptions, export sources, audit results."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path('/checkpoints')
raw = root / 'intent-v3-raw.json'
dataset = Path('/tmp/intent_dataset_v3.json')
expected = len(json.loads(dataset.read_bytes())['cases'])
started = time.monotonic()
try:
    while True:
        try:
            result = json.loads(raw.read_bytes())
            total = len(result['results']) + len(result['failures'])
            if total == expected:
                break
            assert total < expected, 'unexpected extra cases'
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() - started > 6 * 3600:
            raise TimeoutError('not all frozen cases completed within six hours')
        time.sleep(5)
    commands = [
        ['/tmp/score_intent_categories_v3.py', '--dataset', str(dataset), '--input', str(raw),
         '--output', str(root / 'intent-v3-scored.json'), '--workers', '8'],
        ['/tmp/export_intent_v3.py', '--dataset', str(dataset), '--input', str(root / 'intent-v3-scored.json'),
         '--output', str(root / 'intent-v3-audited.json')],
        ['/tmp/summarize_intent_v3.py', '--dataset', str(dataset), '--input', str(root / 'intent-v3-audited.json'),
         '--output', str(root / 'intent-v3-summary.json')],
    ]
    for command in commands:
        subprocess.run([sys.executable, '-u', *command], check=True)
    status = {'status': 'completed', 'expected': expected}
except Exception as exc:
    status = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)[:500]}
(root / 'intent-v3-finalize-status.json').write_text(json.dumps(status), encoding='utf-8')
print(json.dumps(status), flush=True)
