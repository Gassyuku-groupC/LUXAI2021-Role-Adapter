"""Require a candidate Teacher to beat both fixed reference opponents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_OPPONENTS = ("best_agent", "first")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-by-opponent", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min-win-rate", type=float, default=0.5)
    parser.add_argument("--max-timeout-rate", type=float, default=0.0)
    parser.add_argument("--require-positive-margins", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.ranking_by_opponent.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["checkpoint"] == args.checkpoint
        ]
    by_opponent = {row["opponent"]: row for row in rows}
    checks = {}
    for opponent in REQUIRED_OPPONENTS:
        row = by_opponent.get(opponent)
        reasons = []
        if row is None:
            reasons.append("missing_results")
        else:
            if float(row["win_rate"]) <= args.min_win_rate:
                reasons.append("win_rate_not_above_threshold")
            if float(row["timeout_rate"]) > args.max_timeout_rate:
                reasons.append("timeout_rate_above_threshold")
            if args.require_positive_margins:
                if float(row["mean_city_margin"]) <= 0:
                    reasons.append("city_margin_not_positive")
                if float(row["mean_unit_margin"]) <= 0:
                    reasons.append("unit_margin_not_positive")
        checks[opponent] = {"passed": not reasons, "reasons": reasons, "metrics": row}

    report = {
        "checkpoint": args.checkpoint,
        "passed": all(item["passed"] for item in checks.values()),
        "criteria": {
            "win_rate_strictly_above": args.min_win_rate,
            "max_timeout_rate": args.max_timeout_rate,
            "positive_city_and_unit_margins": args.require_positive_margins,
        },
        "opponents": checks,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
