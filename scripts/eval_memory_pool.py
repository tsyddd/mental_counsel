"""Temporary evaluation consumers; use the production memory worker functions.

No model/prompt changes. User advisory locks still serialize each user's jobs.
Run only in a dedicated disposable container; default eight consumers.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from platform_app import mindtrace as mt


def consume():
    while True:
        try:
            if not mt.process_next():
                time.sleep(.5)
        except Exception as exc:
            print(type(exc).__name__ + ': ' + str(exc)[:300], flush=True)
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    mt.embed(['passage: evaluation warmup'])
    print('Production memory consumer pool ready', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(8, args.workers))) as pool:
        list(pool.map(lambda _: consume(), range(max(1, min(8, args.workers)))))
