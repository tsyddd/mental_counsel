"""Drain interrupted batches through the sole production worker, then resume serially."""
import subprocess
import sys
import time
from platform_app import mindtrace as mt

while True:
    rows = mt.all_rows("SELECT count(*) AS n FROM mt_jobs WHERE status IN ('queued','running')")
    pending = rows[0]['n']
    print('Draining existing memory batches:', pending, flush=True)
    if not pending:
        break
    time.sleep(10)
subprocess.run([sys.executable, '-u', '/tmp/eval_intent_v3.py',
                '--dataset', '/tmp/intent_dataset_v3.json',
                '--output', '/checkpoints/intent-v3-raw.json',
                '--limit', '60', '--workers', '1', '--resume'], check=True)
