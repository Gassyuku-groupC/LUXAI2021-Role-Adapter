#!/usr/bin/env python3
"""Aggregate paired checkpoint matches into promotion metrics and a ranking."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from evaluate_replays import player_metrics
except ModuleNotFoundError:
    from scripts.evaluate_replays import player_metrics


PLAYER_SUFFIX = re.compile(r"_p([01])$")
MAP_NAME = re.compile(r"map_(\d+)x\1_")


def candidate_player(path: Path) -> int:
    match = PLAYER_SUFFIX.search(path.stem)
    if not match:
        raise ValueError(f"Cannot infer candidate side from {path.name}")
    return int(match.group(1))


def build_city_count(replay: dict, player: int) -> int:
    return sum(
        1
        for turn in replay.get("allCommands", [])
        for command in turn
        if int(command.get("agentID", -1)) == player
        and str(command.get("command", "")).startswith("bcity ")
    )


def load_games(root: Path) -> tuple[list[dict], dict[str, int], dict[tuple[str, int], int]]:
    games = []
    failures = defaultdict(int)
    failures_by_map = defaultdict(int)
    for manifest_path in root.glob("*/manifest.json"):
        label = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest_failures = manifest.get("failures", [])
        failures[label] += len(manifest_failures)
        for failure in manifest_failures:
            match = MAP_NAME.search(str(failure.get("name", "")))
            if match:
                failures_by_map[(label, int(match.group(1)))] += 1
        for item in manifest.get("completed", []):
            replay_path = Path(item["replay"])
            if not replay_path.is_absolute():
                replay_path = root.parents[2] / replay_path
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            player = candidate_player(replay_path)
            candidate = player_metrics(replay, player)
            opponent = player_metrics(replay, 1 - player)
            games.append({
                "checkpoint": label,
                "file": str(replay_path),
                "map_size": int(item["map_size"]),
                "seed": int(item["seed"]),
                "side": player,
                "win": int(candidate["rank"] == 1),
                "city_margin": candidate["city_tiles"] - opponent["city_tiles"],
                "unit_margin": candidate["units"] - opponent["units"],
                "worst_night_city_loss": candidate["max_night_city_loss"],
                "build_city_count": build_city_count(replay, player),
                "turns": candidate["turns"],
            })
    return games, failures, failures_by_map


def aggregate_by_map(games: list[dict], failures: dict[tuple[str, int], int]) -> list[dict]:
    grouped = defaultdict(list)
    for game in games:
        grouped[(game["checkpoint"], game["map_size"])].append(game)
    rows = []
    for checkpoint, map_size in sorted(set(grouped) | set(failures)):
        items = grouped[(checkpoint, map_size)]
        completed = len(items)
        failed = failures[(checkpoint, map_size)]
        expected = completed + failed
        rows.append({
            "checkpoint": checkpoint,
            "map_size": map_size,
            "completed": completed,
            "failed": failed,
            "timeout_rate": failed / expected if expected else 1.0,
            "win_rate": sum(x["win"] for x in items) / completed if completed else 0.0,
            "mean_city_margin": sum(x["city_margin"] for x in items) / completed if completed else -999.0,
            "mean_unit_margin": sum(x["unit_margin"] for x in items) / completed if completed else -999.0,
            "worst_night_city_loss": max((x["worst_night_city_loss"] for x in items), default=999),
            "build_city_per_game": sum(x["build_city_count"] for x in items) / completed if completed else 0.0,
        })
    return rows


def aggregate(games: list[dict], failures: dict[str, int]) -> list[dict]:
    grouped = defaultdict(list)
    for game in games:
        grouped[game["checkpoint"]].append(game)
    labels = sorted(set(grouped) | set(failures))
    rows = []
    for label in labels:
        items = grouped[label]
        completed = len(items)
        failed = failures[label]
        expected = completed + failed
        rows.append({
            "checkpoint": label,
            "completed": completed,
            "failed": failed,
            "timeout_rate": failed / expected if expected else 1.0,
            "win_rate": sum(x["win"] for x in items) / completed if completed else 0.0,
            "mean_city_margin": sum(x["city_margin"] for x in items) / completed if completed else -999.0,
            "mean_unit_margin": sum(x["unit_margin"] for x in items) / completed if completed else -999.0,
            "worst_night_city_loss": max((x["worst_night_city_loss"] for x in items), default=999),
            "build_city_per_game": sum(x["build_city_count"] for x in items) / completed if completed else 0.0,
        })
    rows.sort(
        key=lambda x: (
            x["win_rate"],
            -x["timeout_rate"],
            x["mean_city_margin"],
            x["mean_unit_margin"],
            -x["worst_night_city_loss"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root
    output.mkdir(parents=True, exist_ok=True)
    games, failures, failures_by_map = load_games(args.root)
    ranking = aggregate(games, failures)
    by_map = aggregate_by_map(games, failures_by_map)
    (output / "summary.json").write_text(
        json.dumps({"ranking": ranking, "by_map": by_map, "games": games}, indent=2), encoding="utf-8"
    )
    for name, rows in (("ranking.csv", ranking), ("ranking_by_map.csv", by_map), ("games.csv", games)):
        if not rows:
            continue
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
