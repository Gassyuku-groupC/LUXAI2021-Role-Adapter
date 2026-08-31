#!/usr/bin/env python3
"""Deduplicate and summarize every evaluated Role-enabled deployment agent."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_replays import player_metrics  # noqa: E402


MATCH_NAME = re.compile(r"map_(\d+)x\1_vs_(.+)_(\d+)_p([01])$")
SUPPORTED_OPPONENTS = {"best_agent", "first"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/role_agent_history_summary"),
    )
    return parser.parse_args()


def deployment_metadata(agent_root: Path) -> dict[str, dict]:
    result = {}
    for package in sorted(agent_root.iterdir()):
        config_path = package / "lux_ai" / "rl_agent" / "rl_agent_config.yaml"
        if not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        role = config.get("role_assignment") or {}
        if not bool(role.get("enabled", False)):
            continue
        result[package.name] = {
            "rot180": "Rot180" in (config.get("data_augmentations") or []),
            "fuel_station_removed": "fuel_station" not in config_path.read_text(
                encoding="utf-8"
            ).lower(),
            "package": str(package.resolve()),
        }
    return result


def infer_opponent(path: Path) -> str | None:
    match = MATCH_NAME.search(path.stem)
    return match.group(2) if match else None


def semantic_key(checkpoint: str, opponent: str, map_size: int, seed: int, side: int):
    return checkpoint, opponent, map_size, seed, side


def load_completed(selection_root: Path, deployments: dict[str, dict]):
    records = {}
    conflicts = defaultdict(set)
    missing_replays = []
    for games_path in selection_root.rglob("games.csv"):
        with games_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            checkpoint = row.get("checkpoint", "")
            if checkpoint not in deployments:
                continue
            replay_path = Path(row.get("file", ""))
            opponent = row.get("opponent") or infer_opponent(replay_path)
            if opponent not in SUPPORTED_OPPONENTS:
                continue
            if not replay_path.is_absolute():
                replay_path = ROOT / replay_path
            if not replay_path.is_file():
                missing_replays.append(str(replay_path))
                continue
            map_size = int(row["map_size"])
            seed = int(row["seed"])
            side = int(row["side"])
            key = semantic_key(checkpoint, opponent, map_size, seed, side)
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            candidate = player_metrics(replay, side)
            reference = player_metrics(replay, 1 - side)
            outcome = (
                int(candidate["rank"] == 1),
                int(candidate["city_tiles"] - reference["city_tiles"]),
                int(candidate["units"] - reference["units"]),
            )
            conflicts[key].add(outcome)
            record = {
                "checkpoint": checkpoint,
                "opponent": opponent,
                "map_size": map_size,
                "seed": seed,
                "side": side,
                "win": outcome[0],
                "score": int(candidate["city_tiles"]),
                "opponent_score": int(reference["city_tiles"]),
                "units": int(candidate["units"]),
                "opponent_units": int(reference["units"]),
                "city_margin": outcome[1],
                "unit_margin": outcome[2],
                "worst_night_city_loss": int(candidate["max_night_city_loss"]),
                "turns": int(candidate["turns"]),
                "replay": str(replay_path.resolve()),
                "source_csv": str(games_path.resolve()),
                "source_mtime": games_path.stat().st_mtime,
            }
            previous = records.get(key)
            if previous is None or record["source_mtime"] >= previous["source_mtime"]:
                records[key] = record
    conflict_rows = [
        {"key": list(key), "outcomes": [list(item) for item in sorted(values)]}
        for key, values in conflicts.items()
        if len(values) > 1
    ]
    return records, conflict_rows, sorted(set(missing_replays))


def load_failures(selection_root: Path, deployments: dict[str, dict], completed_keys: set):
    failures = {}
    for manifest_path in selection_root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        current_agent = Path(str(manifest.get("current_agent", ""))).name
        checkpoint = current_agent if current_agent in deployments else manifest_path.parent.name
        if checkpoint not in deployments:
            continue
        for failure in manifest.get("failures", []):
            match = MATCH_NAME.search(str(failure.get("name", "")))
            if not match:
                continue
            map_size, opponent, seed, side = match.groups()
            if opponent not in SUPPORTED_OPPONENTS:
                continue
            key = semantic_key(checkpoint, opponent, int(map_size), int(seed), int(side))
            if key in completed_keys:
                continue
            failures[key] = {
                "checkpoint": checkpoint,
                "opponent": opponent,
                "map_size": int(map_size),
                "seed": int(seed),
                "side": int(side),
                "error": str(failure.get("error", "")),
                "manifest": str(manifest_path.resolve()),
            }
    return failures


def aggregate(records: list[dict], failures: list[dict], deployments: dict[str, dict], fields):
    grouped = defaultdict(list)
    failed = defaultdict(int)
    for row in records:
        grouped[tuple(row[field] for field in fields)].append(row)
    for row in failures:
        failed[tuple(row[field] for field in fields)] += 1
    output = []
    for key in sorted(set(grouped) | set(failed)):
        games = grouped[key]
        completed = len(games)
        failed_count = failed[key]
        expected = completed + failed_count
        checkpoint = key[fields.index("checkpoint")]
        row = dict(zip(fields, key))
        row.update({
            "rot180": deployments[checkpoint]["rot180"],
            "fuel_station_removed": deployments[checkpoint]["fuel_station_removed"],
            "completed": completed,
            "failed": failed_count,
            "timeout_rate": failed_count / expected if expected else 1.0,
            "win_rate": sum(item["win"] for item in games) / completed if completed else 0.0,
            "mean_score": sum(item["score"] for item in games) / completed if completed else 0.0,
            "mean_opponent_score": sum(item["opponent_score"] for item in games) / completed if completed else 0.0,
            "mean_city_margin": sum(item["city_margin"] for item in games) / completed if completed else -999.0,
            "mean_units": sum(item["units"] for item in games) / completed if completed else 0.0,
            "mean_unit_margin": sum(item["unit_margin"] for item in games) / completed if completed else -999.0,
            "worst_night_city_loss": max((item["worst_night_city_loss"] for item in games), default=999),
            "unique_seeds": len({item["seed"] for item in games}),
        })
        output.append(row)
    output.sort(
        key=lambda row: (
            row["win_rate"],
            -row["timeout_rate"],
            row["mean_city_margin"],
            row["mean_unit_margin"],
            row["completed"],
        ),
        reverse=True,
    )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Rank | Agent | Rot180 | Games | Seeds | Timeout | Win rate | Mean score | City margin | Unit margin |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{row['checkpoint']}` | {'ON' if row['rot180'] else 'OFF'} | "
            f"{row['completed']} | {row['unique_seeds']} | {row['timeout_rate']:.1%} | "
            f"{row['win_rate']:.1%} | {row['mean_score']:.2f} | "
            f"{row['mean_city_margin']:+.2f} | {row['mean_unit_margin']:+.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    deployments = deployment_metadata(root / "outputs" / "checkpoint_selection" / "agents")
    records, conflicts, missing = load_completed(
        root / "outputs" / "checkpoint_selection", deployments
    )
    failures = load_failures(
        root / "outputs" / "checkpoint_selection", deployments, set(records)
    )
    completed_rows = list(records.values())
    failure_rows = list(failures.values())
    overall = aggregate(completed_rows, failure_rows, deployments, ("checkpoint",))
    by_opponent = aggregate(
        completed_rows, failure_rows, deployments, ("checkpoint", "opponent")
    )
    by_map_opponent = aggregate(
        completed_rows,
        failure_rows,
        deployments,
        ("checkpoint", "map_size", "opponent"),
    )
    write_csv(output / "role_agents_overall.csv", overall)
    write_csv(output / "role_agents_by_opponent.csv", by_opponent)
    write_csv(output / "role_agents_by_map_opponent.csv", by_map_opponent)
    write_csv(output / "deduplicated_games.csv", completed_rows)
    write_csv(output / "deduplicated_failures.csv", failure_rows)
    audit = {
        "role_deployments": len(deployments),
        "deduplicated_completed_games": len(completed_rows),
        "unresolved_failures": len(failure_rows),
        "semantic_conflicts": conflicts,
        "missing_replays": missing,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        "# Role Agent Historical Evaluation\n\n"
        "Only Role-enabled deployment packages evaluated against `best_agent` or `first` are included. "
        "Games are deduplicated by agent, opponent, map size, seed, and side. "
        "Mean score is the candidate's final city-tile count, the primary Lux 2021 score.\n\n"
        + markdown_table(overall)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(markdown_table(overall[:20]))


if __name__ == "__main__":
    main()
