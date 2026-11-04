"""Blind fixed-taxonomy mapping, preserving errors and frozen predictions."""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from platform_app import mindtrace as mt


def map_row(row, definitions):
    methods = ['graph', 'baseline']
    random.Random(row['id']).shuffle(methods)
    groups = {f'group_{i}': [str(x.get('label') or x.get('description') or '')
                            for x in row[m + '_intents']] for i, m in enumerate(methods)}
    try:
        mapping = mt.ask('只按给定意图描述映射类别，不推测遗漏意图，不读取对话。'
                         '每组独立；语义相同只计一次；没有描述返回空数组。'
                         '返回{groups:{group_0:[类别代码],group_1:[类别代码]}}。',
                         {'definitions': definitions, 'groups': groups},
                         'blind_intent_label_mapping_v3').get('groups') if any(groups.values()) else {k: [] for k in groups}
        assert isinstance(mapping, dict) and set(mapping) == set(groups), 'incomplete mapping groups'
        for i, method in enumerate(methods):
            values = mapping[f'group_{i}']
            assert isinstance(values, list) and all(isinstance(x, str) and x in definitions for x in values), 'unknown category'
            predicted = sorted(set(values))
            row[method + '_mapped_categories'] = predicted
            gold = [row['gold_category']] if row['gold_category'] else []
            row[method + '_category_correct'] = predicted == gold and bool(row[method + '_prediction']) == row['gold_has_intent']
    except Exception as exc:
        row['category_mapping_error'] = type(exc).__name__ + ': ' + str(exc)[:500]
        for method in methods:
            row[method + '_mapped_categories'] = None
            row[method + '_category_correct'] = False
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    raw = Path(a.dataset).read_bytes()
    source = Path(a.input).read_bytes()
    result = json.loads(source)
    assert hashlib.sha256(raw).hexdigest() == result['dataset_sha256']
    definitions = json.loads(raw)['category_definitions']
    result['category_mapping_note'] = 'same-model blind description mapping; fixed taxonomy; not independent human validation; errors retained'
    result['frozen_prediction_sha256'] = hashlib.sha256(source).hexdigest()
    with ThreadPoolExecutor(max_workers=max(1, min(8, a.workers))) as pool:
        futures = {pool.submit(map_row, dict(row), definitions): row['id'] for row in result['results']}
        mapped = {}
        for future in as_completed(futures):
            row = future.result()
            mapped[row['id']] = row
            result['results'] = [mapped.get(r['id'], r) for r in result['results']]
            Path(a.output).write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode())
            print(json.dumps({'mapped': row['id'], 'graph': row['graph_mapped_categories'],
                              'baseline': row['baseline_mapped_categories'],
                              'error': row.get('category_mapping_error')}, ensure_ascii=False), flush=True)
