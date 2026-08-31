#!/usr/bin/env python3
"""Mine pivotal 24/32 replay windows and split them into preference datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def state_hash(state: dict[str, Any], player: int) -> str:
    encoded = json.dumps(
        {"player": player, "state": state}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def team_state(state: dict[str, Any], player: int) -> dict[str, Any]:
    return state.get("teamStates", {}).get(str(player), {})


def state_features(state: dict[str, Any], player: int) -> tuple[float, ...]:
    teams = []
    for side in (player, 1 - player):
        team = team_state(state, side)
        units = list((team.get("units") or {}).values())
        cities = [c for c in state.get("cities", {}).values() if int(c.get("team", -1)) == side]
        teams.extend((
            sum(len(c.get("cityCells", [])) for c in cities),
            len(units),
            float(team.get("researchPoints", 0)),
            sum(float(c.get("fuel", 0)) for c in cities),
            sum(sum(float(v) for v in (u.get("cargo") or {}).values()) for u in units),
        ))
    return tuple(teams)


def feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    scales = (10.0, 10.0, 50.0, 500.0, 500.0) * 2
    return math.sqrt(sum(((a - b) / scale) ** 2 for a, b, scale in zip(left, right, scales)))


def city_count(state: dict[str, Any], player: int) -> int:
    return sum(
        len(city.get("cityCells", []))
        for city in state.get("cities", {}).values()
        if int(city.get("team", -1)) == player
    )


def economy_signature(state: dict[str, Any], player: int) -> tuple[int, int, int, int]:
    team = team_state(state, player)
    units = list((team.get("units") or {}).values())
    return (
        city_count(state, player),
        len(units),
        int(team.get("researchPoints", 0)),
        int(sum(sum(float(v) for v in (unit.get("cargo") or {}).values()) for unit in units)),
    )


def actions_by_turn(payload: dict[str, Any], player: int) -> dict[int, list[str]]:
    result = {}
    if payload.get("allCommands") is not None:
        for turn, commands in enumerate(payload["allCommands"]):
            result[turn] = sorted(
                str(item["command"])
                for item in commands
                if int(item.get("agentID", -1)) == player and item.get("command")
            )
    else:
        for step in range(1, len(payload.get("steps") or [])):
            result[step - 1] = sorted(payload["steps"][step][player].get("action") or [])
    return result


def first_persistent(values: list[bool], width: int = 3) -> int:
    for index in range(max(0, len(values) - width + 1)):
        if all(values[index:index + width]):
            return index
    return -1


def contains_build_city(actions: list[str]) -> bool:
    return any(action.strip().startswith("bcity ") for action in actions)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_rows(catalog: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in catalog:
        if int(row.get("map_size", 0)) not in (24, 32):
            continue
        if row.get("seed") in (None, "") or row.get("outcome") not in {"win", "loss"}:
            continue
        if not Path(row["metric_path"]).is_file() or not Path(row["canonical_path"]).is_file():
            continue
        grouped[(int(row["map_size"]), str(row["seed"]), int(row["teacher_player"]))].append(row)
    pairs = []
    for rows in grouped.values():
        wins = [row for row in rows if row["outcome"] == "win"]
        losses = [row for row in rows if row["outcome"] == "loss"]
        if not wins or not losses:
            continue
        win = max(wins, key=lambda row: (int(row.get("final_city_margin", 0)), row["replay_id"]))
        for loss in losses:
            pairs.append((win, loss))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--city-delta", type=int, default=2)
    parser.add_argument("--near-distance", type=float, default=0.35)
    args = parser.parse_args()

    catalog = load_json(args.catalog)
    strict, weak, focal, pair_reports = [], [], [], []
    for win, loss in pair_rows(catalog):
        win_stateful = load_json(win["metric_path"])
        loss_stateful = load_json(loss["metric_path"])
        win_states = win_stateful.get("stateful") or []
        loss_states = loss_stateful.get("stateful") or []
        if not win_states or not loss_states:
            continue
        player = int(loss["teacher_player"])
        win_commands = load_json(win["canonical_path"])
        loss_commands = load_json(loss["canonical_path"])
        win_actions = actions_by_turn(win_commands, player)
        loss_actions = actions_by_turn(loss_commands, player)
        turns = min(len(win_states), len(loss_states), max(len(win_actions), len(loss_actions)))
        action_diff = [win_actions.get(t, []) != loss_actions.get(t, []) for t in range(turns)]
        economy_diff = [
            economy_signature(win_states[t], player) != economy_signature(loss_states[t], player)
            for t in range(turns)
        ]
        city_diff = [
            abs(city_count(win_states[t], player) - city_count(loss_states[t], player)) >= args.city_delta
            for t in range(turns)
        ]
        t_action = next((t for t, differs in enumerate(action_diff) if differs), -1)
        t_economy = first_persistent(economy_diff)
        t_star = first_persistent(city_diff)
        pivots = [turn for turn in (t_action, t_economy, t_star) if turn >= 0]
        if not pivots:
            continue
        pivot = min(pivots)
        end = t_star if t_star >= 0 else min(turns - 1, pivot + args.window)
        start = max(0, pivot - args.window)
        pair_id = f"{win['replay_id']}__{loss['replay_id']}"
        pair_reports.append({
            "pair_id": pair_id, "map_size": loss["map_size"], "seed": loss["seed"],
            "player": player, "split": loss["split"], "t_action": t_action,
            "t_economy": t_economy, "t_star": t_star, "window_start": start,
            "window_end": end, "win_replay": win["replay_id"], "loss_replay": loss["replay_id"],
        })
        for turn in range(start, min(end + 1, turns)):
            preferred = win_actions.get(turn, [])
            rejected = loss_actions.get(turn, [])
            if not preferred or preferred == rejected:
                continue
            common = {
                "pair_id": pair_id, "replay_id": win["replay_id"], "split": loss["split"],
                "map_size": loss["map_size"], "seed": loss["seed"], "teacher_player": player,
                "state_step": turn, "action_step": turn, "t_action": t_action,
                "t_economy": t_economy, "t_star": t_star, "window_start": start,
                "window_end": end, "preferred_actions_json": json.dumps(preferred, separators=(",", ":")),
                "rejected_actions_json": json.dumps(rejected, separators=(",", ":")),
            }
            exact = state_hash(win_states[turn], player) == state_hash(loss_states[turn], player)
            distance = feature_distance(
                state_features(win_states[turn], player), state_features(loss_states[turn], player)
            )
            if rejected and not contains_build_city(rejected):
                if exact:
                    strict.append({
                        **common, "sample_type": "strict_dpo", "file": loss["metric_path"],
                        "preferred_file": win["canonical_path"], "rejected_file": loss["canonical_path"],
                        "state_distance": 0.0, "weight": 1.0, "weight_reason": "same_state_win_vs_loss",
                    })
                elif distance <= args.near_distance:
                    weak.append({
                        **common, "sample_type": "near_weak_preference", "file": win["metric_path"],
                        "preferred_file": win["canonical_path"], "rejected_file": loss["canonical_path"],
                        "state_distance": round(distance, 6), "weight": 0.25,
                        "weight_reason": "near_state_win_vs_loss",
                    })
            focal.append({
                **common, "sample_type": "critical_focal_bc", "file": win["metric_path"],
                "preferred_file": win["canonical_path"], "rejected_file": "",
                "rejected_actions_json": "[]", "state_distance": round(distance, 6),
                "weight": 1.0 if turn >= pivot else 0.75,
                "weight_reason": "winning_pivotal_window",
            })

    # Replay-only external winners cannot form strict same-state DPO pairs.
    # They remain valid focal-BC demonstrations on their own observed states.
    for loss in catalog:
        if (
            loss.get("source") != "first_battle_C"
            or loss.get("outcome") != "loss"
            or loss.get("opponent") not in {"B", "G", "D"}
            or int(loss.get("map_size", 0)) not in (24, 32)
            or not Path(loss.get("metric_path", "")).is_file()
            or not Path(loss.get("canonical_path", "")).is_file()
        ):
            continue
        replay = load_json(loss["metric_path"])
        states = replay.get("stateful") or []
        if not states:
            continue
        loser = int(loss["teacher_player"])
        winner = 1 - loser
        commands = actions_by_turn(load_json(loss["canonical_path"]), winner)
        city_deficit = [
            city_count(state, loser) - city_count(state, winner) <= -args.city_delta
            for state in states
        ]
        economy_deficit = [
            economy_signature(state, loser)[:3] < economy_signature(state, winner)[:3]
            for state in states
        ]
        t_economy = first_persistent(economy_deficit)
        t_star = first_persistent(city_deficit)
        pivot_candidates = [turn for turn in (t_economy, t_star) if turn >= 0]
        if not pivot_candidates:
            continue
        pivot = min(pivot_candidates)
        end = t_star if t_star >= 0 else min(len(states) - 1, pivot + args.window)
        start = max(0, pivot - args.window)
        pair_id = f"external_focal__{loss['replay_id']}"
        pair_reports.append({
            "pair_id": pair_id, "map_size": loss["map_size"], "seed": loss["seed"],
            "player": winner, "split": loss["split"], "t_action": -1,
            "t_economy": t_economy, "t_star": t_star, "window_start": start,
            "window_end": end, "win_replay": loss["replay_id"], "loss_replay": loss["replay_id"],
        })
        for turn in range(start, min(end + 1, len(states))):
            preferred = commands.get(turn, [])
            if not preferred:
                continue
            focal.append({
                "pair_id": pair_id, "replay_id": loss["replay_id"], "split": loss["split"],
                "map_size": loss["map_size"], "seed": loss["seed"], "teacher_player": winner,
                "state_step": turn, "action_step": turn, "t_action": -1,
                "t_economy": t_economy, "t_star": t_star, "window_start": start,
                "window_end": end,
                "preferred_actions_json": json.dumps(preferred, separators=(",", ":")),
                "rejected_actions_json": "[]", "sample_type": "critical_focal_bc",
                "file": loss["metric_path"], "preferred_file": loss["canonical_path"],
                "rejected_file": "", "state_distance": -1.0, "weight": 1.5,
                "weight_reason": f"external_{loss['opponent']}_winner_pivotal_window",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "strict_same_state_dpo.csv", strict)
    write_csv(args.output_dir / "near_weak_preferences.csv", weak)
    write_csv(args.output_dir / "critical_focal_bc.csv", focal)
    write_csv(args.output_dir / "pair_divergence.csv", pair_reports)
    combined = strict + weak + focal
    write_csv(args.output_dir / "critical_preferences.csv", combined)
    summary = {
        "pairs": len(pair_reports), "strict_same_state_dpo": len(strict),
        "near_weak_preferences": len(weak), "critical_focal_bc": len(focal),
        "build_city_rejected_samples": sum(
            contains_build_city(json.loads(row["rejected_actions_json"])) for row in combined
        ),
        "by_map": dict(Counter(str(row["map_size"]) for row in combined)),
        "by_split": dict(Counter(row["split"] for row in combined)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
