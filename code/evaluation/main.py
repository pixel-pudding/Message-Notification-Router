"""Evaluation: compare a predicted output CSV (from a --sample run of
code/main.py) against dataset/sample_messages.csv ground truth. Prints a
per-row match table, action/message_type/combined accuracy, and lists every
mismatch with the agent's reasoning so we know exactly where to focus
iteration time.

Usage:
    python code/evaluation/main.py [--predicted CSV] [--ground-truth CSV]

Default --predicted is dataset/output_sample.csv, produced by:
    python code/main.py --sample --no-resume
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402 -- must follow the sys.path fix-up above


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate(predicted_path: Path, ground_truth_path: Path) -> None:
    predicted = {row["message_id"]: row for row in _read_csv(predicted_path)}
    ground_truth = {row["message_id"]: row for row in _read_csv(ground_truth_path)}

    rows = []
    missing_predictions = []

    for message_id, truth in ground_truth.items():
        pred = predicted.get(message_id)
        if pred is None:
            missing_predictions.append(message_id)
            continue

        action_match = pred["action"] == truth["action"]
        type_match = pred["message_type"] == truth["message_type"]
        rows.append(
            {
                "message_id": message_id,
                "pred_action": pred["action"],
                "true_action": truth["action"],
                "action_match": action_match,
                "pred_type": pred["message_type"],
                "true_type": truth["message_type"],
                "type_match": type_match,
                "reason": pred.get("reason", ""),
            }
        )

    total = len(rows)
    action_correct = sum(r["action_match"] for r in rows)
    type_correct = sum(r["type_match"] for r in rows)
    both_correct = sum(r["action_match"] and r["type_match"] for r in rows)

    print(f"{'message_id':<16} {'pred_action':<10} {'true_action':<10} {'match':<6} {'pred_type':<16} {'true_type':<16} {'match':<6}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['message_id']:<16} {r['pred_action']:<10} {r['true_action']:<10} {'OK' if r['action_match'] else 'X':<6} "
            f"{r['pred_type']:<16} {r['true_type']:<16} {'OK' if r['type_match'] else 'X':<6}"
        )

    print("\n=== SUMMARY ===")
    print(f"rows evaluated:         {total}")
    if missing_predictions:
        print(f"missing predictions:    {len(missing_predictions)} ({', '.join(missing_predictions)})")
    if total:
        print(f"action accuracy:        {action_correct}/{total} ({action_correct / total:.1%})")
        print(f"message_type accuracy:  {type_correct}/{total} ({type_correct / total:.1%})")
        print(f"combined accuracy:      {both_correct}/{total} ({both_correct / total:.1%})")
    else:
        print("no rows to evaluate -- check --predicted/--ground-truth paths")

    mismatches = [r for r in rows if not (r["action_match"] and r["type_match"])]
    if mismatches:
        print(f"\n=== MISMATCHES ({len(mismatches)}/{total}) ===")
        for r in mismatches:
            print(f"\n--- {r['message_id']} ---")
            action_flag = "match" if r["action_match"] else "MISMATCH"
            type_flag = "match" if r["type_match"] else "MISMATCH"
            print(f"  action:       predicted={r['pred_action']!r} vs true={r['true_action']!r} ({action_flag})")
            print(f"  message_type: predicted={r['pred_type']!r} vs true={r['true_type']!r} ({type_flag})")
            print(f"  reason: {r['reason']}")
    else:
        print("\nNo mismatches.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predicted", type=Path, default=config.DATASET_DIR / "output_sample.csv")
    parser.add_argument("--ground-truth", type=Path, default=config.SAMPLE_MESSAGES_CSV)
    args = parser.parse_args()
    evaluate(args.predicted, args.ground_truth)


if __name__ == "__main__":
    main()
