#!/usr/bin/env python3
"""Build a deduplicated replay catalog for Group C / best_agent games."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BEST_MATCH = re.compile(r"_vs_best_agent_(\d+)_p([01])$")
FIRST_BATTLE = re.compile(r"^(\d+)x\1_([A-H])_([A-H])$")


def is_group_c_team_name(name: str) -> bool:
    normalized = name.replace("/", "\\").lower()
    return normalized in {"c", "group c", "team c"} or "\\luxai\\c\\" in normalized


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def replay_format(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("stateful") and payload.get("allCommands") is not None:
        return "stateful"
    if payload.get("allCommands") is not None:
        return "commands"
    if payload.get("steps"):
        return "kaggle_steps"
    return None


def action_payload(payload: Dict[str, Any]) -> Any:
    if payload.get("allCommands") is not None:
        return payload["allCommands"]
    actions = []
    for step in payload.get("steps") or []:
        actions.append([
            item.get("action") or [] if isinstance(item, dict) else []
            for item in step[:2]
        ])
    return actions


def fingerprint(payload: Dict[str, Any]) -> str:
    canonical = {
        "seed": payload.get("seed") or payload.get("configuration", {}).get("seed"),
        "width": payload.get("width") or payload.get("configuration", {}).get("width"),
        "height": payload.get("height") or payload.get("configuration", {}).get("height"),
        "actions": action_payload(payload),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranks(payload: Dict[str, Any]) -> Dict[int, int]:
    return {
        int(item["agentID"]): int(item["rank"])
        for item in payload.get("results", {}).get("ranks", [])
    }


def first_battle_results(path: Optional[Path]) -> Dict[tuple, Dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["map_size"]), row["first"], row["second"])] = row
    return rows


def infer_identity(
    path: Path,
    payload: Dict[str, Any],
    results: Dict[tuple, Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    normalized = str(path).replace("/", "\\")
    stem = path.stem.removesuffix(".commands").removesuffix(".roles")
    parent_size = re.fullmatch(r"(12|16|24|32)x\1", path.parent.name)
    first_match = re.fullmatch(r"([A-H])_([A-H])", stem)
    generated_match = re.fullmatch(r"(12|16|24|32)x\1_([A-H])_([A-H])", stem)
    is_first_battle = "first battle" in normalized.lower() and parent_size and first_match
    is_generated_copy = "c_group_loss_audit" in normalized.lower() and generated_match
    teams = first_match.groups() if is_first_battle else generated_match.groups()[1:] if is_generated_copy else ()
    if teams and "C" in teams:
        map_size = int(parent_size.group(1)) if is_first_battle else int(generated_match.group(1))
        first, second = teams
        player = 0 if first == "C" else 1
        opponent = second if player == 0 else first
        result = results.get((map_size, first, second), {})
        if str(result.get("draw", "")).upper() == "TRUE":
            outcome = "draw"
        elif result.get("winner"):
            outcome = "win" if result["winner"] == "C" else "loss"
        else:
            outcome = "win" if ranks(payload).get(player) == 1 else "loss"
        return {
            "teacher_identity": "C/best_agent",
            "teacher_player": player,
            "opponent": opponent,
            "outcome": outcome,
            "source": "first_battle_C",
            "source_replay_key": f"{map_size}:{first}:{second}",
        }

    match = BEST_MATCH.search(stem)
    if match:
        candidate_player = int(match.group(2))
        player = 1 - candidate_player
        rank = ranks(payload).get(player)
        outcome = "unknown" if rank is None else "win" if rank == 1 else "loss"
        checkpoint = next(
            (part for part in reversed(path.parts[:-1]) if part not in {"replays", "outputs"}),
            "unknown",
        )
        return {
            "teacher_identity": "C/best_agent",
            "teacher_player": player,
            "opponent": checkpoint,
            "outcome": outcome,
            "source": "deployed_vs_best_agent",
        }

    details = payload.get("teamDetails") or []
    for player, detail in enumerate(details[:2]):
        name = str(detail.get("name", ""))
        if "best_agent" in name.lower() or is_group_c_team_name(name):
            rank = ranks(payload).get(player)
            return {
                "teacher_identity": "C/best_agent",
                "teacher_player": player,
                "opponent": str(details[1 - player].get("name", "unknown")) if len(details) > 1 else "unknown",
                "outcome": "unknown" if rank is None else "win" if rank == 1 else "loss",
                "source": "team_details_best_or_C",
            }
    return None


def candidate_paths(roots: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            lowered = str(path).lower()
            if not path.name.endswith(".commands.json"):
                command_sibling = path.with_name(f"{path.stem}.commands.json")
                if command_sibling.is_file() and "first battle" not in lowered:
                    continue
            likely = (
                "first battle" in lowered
                or "best_agent" in lowered
                or "c_group_loss_audit" in lowered
                or path.name.endswith(".commands.json")
            )
            if likely and path not in seen and not path.name.endswith(".roles.json"):
                seen.add(path)
                yield path


def metric_summary(payload: Dict[str, Any], player: int) -> Dict[str, Any]:
    states = payload.get("stateful") or []
    if not states:
        return {"turns": len(payload.get("allCommands") or [])}
    city_margins, unit_margins, night_losses = [], [], []
    city_counts = []
    for state in states:
        cities = [0, 0]
        for city in state.get("cities", {}).values():
            cities[int(city["team"])] += len(city.get("cityCells", []))
        units = [len(state["teamStates"][str(index)]["units"]) for index in (0, 1)]
        city_counts.append(cities[player])
        city_margins.append(cities[player] - cities[1 - player])
        unit_margins.append(units[player] - units[1 - player])
    for turn in range(len(states) - 1):
        if (turn + 1) % 40 >= 30:
            loss = city_counts[turn] - city_counts[turn + 1]
            if loss > 0:
                night_losses.append((turn + 1, loss))
    decisive_deficit = -1
    for turn in range(len(city_margins)):
        if city_margins[turn] < 0 and all(value < 0 for value in city_margins[turn:]):
            decisive_deficit = turn
            break
    max_night_turn, max_night_loss = max(night_losses, key=lambda item: item[1], default=(-1, 0))
    failure_turn = max_night_turn if max_night_loss >= 20 else decisive_deficit
    return {
        "turns": int(states[-1].get("turn", len(states) - 1)),
        "final_city_margin": city_margins[-1],
        "final_unit_margin": unit_margins[-1],
        "max_night_loss": max_night_loss,
        "max_night_loss_turn": max_night_turn,
        "decisive_deficit_turn": decisive_deficit,
        "failure_turn": failure_turn,
        "safe_prefix_end": max(failure_turn - 20, 0) if failure_turn >= 0 else -1,
    }


def split_for(fingerprint_value: str, map_size: int, seed: Any) -> str:
    # Every replay sharing a map/seed pair stays in one split. Fingerprint is a
    # fallback only for replay formats without a usable seed.
    group_key = f"{map_size}:{seed}" if seed not in (None, "") else f"{map_size}:{fingerprint_value}"
    bucket = int(hashlib.sha256(group_key.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "calibration"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row if key != "duplicate_paths"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--first-battle-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result_lookup = first_battle_results(args.first_battle_results)
    grouped = defaultdict(list)
    scanned = 0
    for path in candidate_paths(args.root):
        payload = load_json(path)
        if payload is None or replay_format(payload) is None:
            continue
        identity = infer_identity(path, payload, result_lookup)
        if identity is None:
            continue
        scanned += 1
        replay_hash = fingerprint(payload)
        if identity["source"] == "first_battle_C":
            replay_hash = hashlib.sha256(
                f"first_battle_C:{identity['source_replay_key']}".encode("utf-8")
            ).hexdigest()
        grouped[replay_hash].append({
            "path": path,
            "format": replay_format(payload),
            "identity": identity,
            "width": int(payload.get("width") or payload.get("configuration", {}).get("width") or 0),
            "seed": payload.get("seed") or payload.get("configuration", {}).get("seed"),
        })

    rows = []
    format_priority = {"stateful": 0, "commands": 1, "kaggle_steps": 2}
    for replay_hash, copies in grouped.items():
        copies.sort(key=lambda item: (format_priority[item["format"]], len(str(item["path"]))))
        canonical = copies[0]
        canonical_path = canonical["path"]
        canonical_payload = load_json(canonical_path)
        identity = canonical["identity"]
        stateful_copy = next((item for item in copies if item["format"] == "stateful"), None)
        metric_path = stateful_copy["path"] if stateful_copy else canonical_path
        if stateful_copy is None and canonical_path.name.endswith(".commands.json"):
            sibling = canonical_path.with_name(canonical_path.name.replace(".commands.json", ".json"))
            if sibling.is_file():
                metric_path = sibling
        metric_payload = load_json(metric_path) if metric_path != canonical_path else canonical_payload
        width = canonical["width"]
        row = {
            "replay_id": replay_hash[:16],
            "fingerprint": replay_hash,
            "canonical_path": str(canonical_path),
            "metric_path": str(metric_path),
            "format": canonical["format"],
            "map_size": width,
            "seed": canonical["seed"],
            "duplicate_count": len(copies),
            "duplicate_paths": [str(item["path"]) for item in copies],
            **identity,
            **metric_summary(metric_payload, int(identity["teacher_player"])),
        }
        row["split"] = split_for(replay_hash, width, row["seed"])
        rows.append(row)
    rows.sort(key=lambda row: (row["map_size"], row["source"], row["replay_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "catalog.csv", rows)
    (args.output_dir / "catalog.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = {
        "recognized_files": scanned,
        "unique_replays": len(rows),
        "duplicates_removed": scanned - len(rows),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_map": dict(Counter(str(row["map_size"]) for row in rows)),
        "by_outcome": dict(Counter(row["outcome"] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
