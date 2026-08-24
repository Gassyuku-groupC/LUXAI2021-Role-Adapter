#!/usr/bin/env python3
"""Summarize per-turn Role agent inference timings archived by replay generation."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


MAP_RE = re.compile(r"map_(\d+)x\1_")
INFERENCE_RE = re.compile(r"Model inference:\s*([0-9]+(?:\.[0-9]+)?)")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    files_by_group: dict[tuple[str, int], int] = defaultdict(int)
    for path in args.root.rglob("*.agent_current.log"):
        match = MAP_RE.search(path.name)
        if not match:
            continue
        map_size = int(match.group(1))
        checkpoint = path.parent.parent.name
        values = [float(value) for value in INFERENCE_RE.findall(
            path.read_text(encoding="utf-8", errors="replace")
        )]
        if values:
            grouped[(checkpoint, map_size)].extend(values)
            files_by_group[(checkpoint, map_size)] += 1

    summary = []
    for checkpoint, map_size in sorted(grouped):
        values = grouped[(checkpoint, map_size)]
        summary.append({
            "checkpoint": checkpoint,
            "map_size": map_size,
            "log_files": files_by_group[(checkpoint, map_size)],
            "turn_samples": len(values),
            "mean_inference_seconds": sum(values) / len(values),
            "p95_inference_seconds": percentile(values, 0.95),
            "max_inference_seconds": max(values),
        })
    output = json.dumps(summary, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
