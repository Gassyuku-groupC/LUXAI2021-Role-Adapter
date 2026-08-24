#!/usr/bin/env python3
"""Create a frame-level weighted BC index from the best/C replay catalog."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


EXTERNAL_EXPERTS = {"B", "G", "D"}


def load_replay(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def city_tiles(state: Dict[str, Any], player: int) -> int:
    return sum(
        len(city.get("cityCells", []))
        for city in state.get("cities", {}).values()
        if int(city.get("team", -1)) == player
    )


def action_rows(payload: Dict[str, Any], player: int) -> Dict[int, List[str]]:
    result = {}
    if payload.get("allCommands") is not None:
        for turn, commands in enumerate(payload["allCommands"]):
            actions = [
                str(item.get("command", ""))
                for item in commands
                if int(item.get("agentID", -1)) == player and item.get("command")
            ]
            if actions:
                result[turn] = actions
        return result
    for action_step in range(1, len(payload.get("steps") or [])):
        actions = payload["steps"][action_step][player].get("action") or []
        if actions:
            result[action_step - 1] = [str(action) for action in actions]
    return result


def stateful_copy(entry: Dict[str, Any]) -> Dict[str, Any]:
    metric_path = Path(entry["metric_path"])
    if metric_path.is_file():
        payload = load_replay(str(metric_path))
        if payload.get("stateful"):
            return payload
    return {}


def frame_weight(
    *, entry: Dict[str, Any], turn: int, player: int, states: List[Dict[str, Any]],
    external: bool,
) -> tuple[float, str, int]:
    map_size = int(entry["map_size"])
    outcome = entry["outcome"]
    failure_turn = int(entry.get("failure_turn", -1) or -1)
    safe_end = int(entry.get("safe_prefix_end", -1) or -1)
    loss_next = 0
    if states and turn + 1 < len(states):
        loss_next = max(city_tiles(states[turn], player) - city_tiles(states[turn + 1], player), 0)

    if external:
        weight = 2.5 if entry["opponent"] == "B" and map_size >= 24 else 1.75
        reasons = ["external_winner", f"expert_{entry['opponent']}"]
        if map_size >= 24:
            weight *= 1.25
            reasons.append("large_map")
        if turn >= 120:
            weight *= 1.25
            reasons.append("coordination_phase")
        if loss_next:
            weight *= 0.20
            reasons.append("downweight_teacher_city_loss")
        return weight, "+".join(reasons), loss_next

    if outcome == "loss":
        if failure_turn >= 0 and turn > safe_end:
            return 0.0, "best_failure_window_excluded", loss_next
        weight = 0.35
        reasons = ["best_loss_safe_prefix"]
    elif outcome == "draw":
        weight = 0.60
        reasons = ["best_draw"]
    else:
        weight = 1.0
        reasons = ["best_win"]
    if map_size <= 16:
        weight *= 1.25
        reasons.append("small_map_anchor")
    elif map_size >= 24 and turn >= 120:
        weight *= 1.15
        reasons.append("large_map_coordination")
    if turn % 40 >= 25:
        weight *= 1.30
        reasons.append("night_window")
    if loss_next:
        weight *= 0.20
        reasons.append("downweight_city_loss")
    return weight, "+".join(reasons), loss_next


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rows, excluded = [], []
    for entry in catalog:
        payload = load_replay(entry["canonical_path"])
        stateful = stateful_copy(entry)
        states = stateful.get("stateful") or []
        best_player = int(entry["teacher_player"])
        teachers = [(best_player, False, "best_anchor")]
        if entry["source"] == "first_battle_C" and entry["outcome"] == "loss" and entry["opponent"] in EXTERNAL_EXPERTS:
            teachers.append((1 - best_player, True, "external_winner"))
        for player, external, sample_type in teachers:
            for turn, actions in action_rows(payload, player).items():
                weight, reason, loss_next = frame_weight(
                    entry=entry, turn=turn, player=player, states=states, external=external
                )
                row = {
                    "replay_id": entry["replay_id"],
                    "fingerprint": entry["fingerprint"],
                    "split": entry["split"],
                    "source": entry["source"],
                    "file": entry["canonical_path"],
                    "metric_file": entry["metric_path"],
                    "map_size": entry["map_size"],
                    "seed": entry["seed"],
                    "outcome_for_best": entry["outcome"],
                    "opponent": entry["opponent"],
                    "teacher_player": player,
                    "teacher_identity": entry["opponent"] if external else "C/best_agent",
                    "sample_type": sample_type,
                    "state_step": turn,
                    "action_step": turn,
                    "action_count": len(actions),
                    "actions_json": json.dumps(actions, separators=(",", ":")),
                    "city_loss_next": loss_next,
                    "failure_turn": entry.get("failure_turn", -1),
                    "weight": round(weight, 6),
                    "weight_reason": reason,
                }
                (rows if weight > 0 else excluded).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "bc_index.csv", rows)
    write_csv(args.output_dir / "excluded_failure_windows.csv", excluded)
    summary = {
        "catalog_replays": len(catalog),
        "included_frames": len(rows),
        "excluded_frames": len(excluded),
        "included_weight": sum(float(row["weight"]) for row in rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_teacher": dict(Counter(row["teacher_identity"] for row in rows)),
        "by_map": dict(Counter(str(row["map_size"]) for row in rows)),
        "by_sample_type": dict(Counter(row["sample_type"] for row in rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
