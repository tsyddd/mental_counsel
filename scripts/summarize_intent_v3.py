"""Offline audit of frozen predictions; never calls a model or modifies predictions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(dataset, result):
    cases = {c['id']: c for c in dataset['cases']}
    rows = {r['id']: r for r in result['results']}
    failures = {r['case']: r for r in result['failures']}
    assert len(rows) == len(result['results']), 'duplicate predictions'
    assert not (rows.keys() & failures.keys()), 'case both predicted and failed'
    assert rows.keys() <= cases.keys() and failures.keys() <= cases.keys()
    source_checks = 0
    invalid_baseline_sources = []
    evidence = []
    for cid, row in rows.items():
        case = cases[cid]
        assert row['gold_category'] == case['gold_category']
        assert row['gold_has_intent'] == case['gold_has_intent']
        assert len(row['messages']) == len(case['turns']) * 2
        ids = {m['id']: m for m in row['messages']}
        user_ids = {m['id'] for m in row['messages'] if m['role'] == 'user'}
        events = {e['id']: e for e in row['events']}
        for e in events.values():
            assert e['source_ids'] and set(e['source_ids']) <= user_ids
            sources = [ids[s] for s in e['source_ids']]
            assert all(s['session_id'] == e['source_session'] for s in sources)
            turn = lambda m: m.get('turn', m.get('turn_no'))
            assert e['start_turn'] == min(turn(s) for s in sources)
            assert e['end_turn'] == max(turn(s) for s in sources)
            source_checks += 1
        for intent in row['graph_intents']:
            assert set(intent['support_event_ids']) <= events.keys()
        for intent in row['baseline_intents']:
            for mid in intent.get('support_message_ids', []):
                if mid not in user_ids:
                    invalid_baseline_sources.append({'case': cid, 'message_id': mid})
        evidence.append({'case': cid, 'event_count': len(events),
                         'window_chars': row['window_chars']})
    summaries = {}
    errors = []
    for mode in ['long', 'cross', 'all']:
        selected = [c for c in cases.values() if mode == 'all' or c['mode'] == mode]
        out = {'expected': len(selected), 'completed': sum(c['id'] in rows for c in selected),
               'failed': sum(c['id'] in failures for c in selected),
               'pending': sum(c['id'] not in rows and c['id'] not in failures for c in selected)}
        for method in ['graph', 'baseline']:
            available = [rows[c['id']] for c in selected if c['id'] in rows]
            matrix = Counter(('tp' if r['gold_has_intent'] else 'fp') if r[method + '_prediction']
                             else ('fn' if r['gold_has_intent'] else 'tn') for r in available)
            correct = matrix['tp'] + matrix['tn']
            tp, fp, fn = matrix['tp'], matrix['fp'], matrix['fn']
            out[method] = {**{k: matrix[k] for k in ['tp','tn','fp','fn']},
                           'correct': correct, 'accuracy_all_expected': correct / len(selected),
                           'precision_completed': tp / (tp + fp) if tp + fp else 0,
                           'recall_completed': tp / (tp + fn) if tp + fn else 0,
                           'f1_completed': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
            if available and all(method + '_category_correct' in r for r in available):
                n = sum(r[method + '_category_correct'] for r in available)
                out[method]['category_correct'] = n
                out[method]['category_accuracy_all_expected'] = n / len(selected)
        paired = [rows[c['id']] for c in selected if c['id'] in rows]
        out['discordant_binary'] = {
            'graph_only_correct': sum(r['graph_correct'] and not r['baseline_correct'] for r in paired),
            'baseline_only_correct': sum(r['baseline_correct'] and not r['graph_correct'] for r in paired)}
        summaries[mode] = out
    for cid, r in rows.items():
        wrong = [m for m in ['graph','baseline'] if not r[m + '_correct']
                 or r.get(m + '_category_correct') is False]
        if wrong:
            errors.append({'id': cid, 'wrong_methods': wrong, 'gold': r['gold_category'],
                           'graph_intents': [i.get('label') or i.get('description') for i in r['graph_intents']],
                           'baseline_intents': [i.get('label') or i.get('description') for i in r['baseline_intents']],
                           'graph_categories': r.get('graph_mapped_categories'),
                           'baseline_categories': r.get('baseline_mapped_categories')})
    return {'dataset_sha256': result['dataset_sha256'], 'model': result['model'],
            'independent_trajectories': dataset['n_independent_scenarios'], 'summary': summaries,
            'source_events_checked': source_checks, 'invalid_baseline_source_ids': invalid_baseline_sources,
            'max_window_chars': max((e['window_chars'] for e in evidence), default=0),
            'failures': list(failures.values()), 'errors': errors,
            'limitations': ['authored synthetic labels, no independent human review',
                            'long/cross modes paired by trajectory',
                            'category mapping is model assisted',
                            'only final checkpoints; no claim of multi-event necessity']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    raw = Path(a.dataset).read_bytes()
    result = json.loads(Path(a.input).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == result['dataset_sha256']
    summary = summarize(json.loads(raw), result)
    Path(a.output).write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ['errors']}, ensure_ascii=False))
