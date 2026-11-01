"""Merge completed intent-ablation shards and report both scenario types."""
import argparse
import json
from pathlib import Path


def metrics(rows, arm):
    tp = sum(r["gold"] is not None and r[arm]["has_intent"] for r in rows)
    tn = sum(r["gold"] is None and not r[arm]["has_intent"] for r in rows)
    fp = sum(r["gold"] is None and r[arm]["has_intent"] for r in rows)
    fn = sum(r["gold"] is not None and not r[arm]["has_intent"] for r in rows)
    exact = sum((r["gold"] is None and not r[arm]["has_intent"]) or
                (r["gold"] is not None and r[arm]["has_intent"] and r["gold"] == r[arm]["label"])
                for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "binary_accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "label_accuracy": exact / len(rows) if rows else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    found = {}
    for path in args.shards:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["cases"]:
            if "full" not in row or "baseline" not in row:
                continue
            if row["id"] in found:
                raise ValueError(f"duplicate completed case: {row['id']}")
            found[row["id"]] = row
    if len(found) != 50:
        raise ValueError(f"expected 50 completed cases, found {len(found)}")
    rows = [found[key] for key in sorted(found)]
    counts = {mode: sum(r["mode"] == mode for r in rows)
              for mode in ("cross_session", "single_30_turn")}
    if counts != {"cross_session": 40, "single_30_turn": 10}:
        raise ValueError(f"unexpected scenario counts: {counts}")
    groups = {"all": rows}
    groups.update({mode: [r for r in rows if r["mode"] == mode] for mode in counts})
    summary = {name: {arm: metrics(group, arm) for arm in ("full", "baseline")}
               for name, group in groups.items()}
    cleaned = []
    for row in rows:
        cleaned.append({key: row[key] for key in (
            "id", "mode", "turn_count", "gold", "gold_evidence_turns",
            "draft_turns", "turns", "gpt_paraphrase_valid", "full", "baseline")})
    result = {
        "dataset": "50 synthetic counseling intent cases, CogEval-inspired, not official CogEval-Bench",
        "model": "gpt-6-sol", "scenario_counts": counts,
        "summary": summary, "cases": cleaned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
