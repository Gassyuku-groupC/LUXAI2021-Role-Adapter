#!/usr/bin/env python3
"""Extract replay imitation-learning tensors from a weighted index."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from imitation_learning_utils import (
    action_placeholder,
    advance_manual_env,
    build_manual_env,
    env_output_for_current_state,
    load_agent_flags,
    teacher_actions_to_mask,
)
from lux_ai.rl_agent.role_assignment import RoleAssignmentConfig
from lux_ai.rl_agent.trainable_role_bias import RoleBiasCodeBuilder, attach_role_bias_codes


def parse_int_set(text: str) -> set[int]:
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def stateful_updates(state: dict) -> List[str]:
    updates = []
    for team_text, team_state in state.get("teamStates", {}).items():
        team = int(team_text)
        updates.append(f"rp {team} {team_state.get('researchPoints', 0)}")
        for unit_id, unit in team_state.get("units", {}).items():
            cargo = unit.get("cargo", {})
            updates.append(
                f"u {unit.get('type', 0)} {team} {unit_id} {unit['x']} {unit['y']} "
                f"{unit.get('cooldown', 0)} {cargo.get('wood', 0)} "
                f"{cargo.get('coal', 0)} {cargo.get('uranium', 0)}"
            )
    for y, row in enumerate(state.get("map", [])):
        for x, cell in enumerate(row):
            resource = cell.get("resource")
            if resource:
                updates.append(f"r {resource['type']} {x} {y} {resource.get('amount', 0)}")
            road = float(cell.get("road", 0) or 0)
            if road:
                updates.append(f"ccd {x} {y} {road}")
    for city_id, city in state.get("cities", {}).items():
        team = int(city["team"])
        updates.append(f"c {team} {city_id} {city.get('fuel', 0)} {city.get('lightupkeep', 0)}")
        for tile in city.get("cityCells", []):
            updates.append(
                f"ct {team} {city_id} {tile['x']} {tile['y']} {tile.get('cooldown', 0)}"
            )
    updates.append("D_DONE")
    return updates


def replay_updates(replay: dict, step: int) -> List[str]:
    if replay.get("stateful"):
        updates = stateful_updates(replay["stateful"][step])
        if step == 0:
            return ["0", f"{replay['width']} {replay['height']}", *updates]
        return updates
    return list(replay["steps"][step][0]["observation"].get("updates") or [])


def load_index(path: Path, max_rows: int = 0, map_sizes: set[int] | None = None, wanted_split: str = "") -> Dict[Path, List[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            if wanted_split and row.get("split") != wanted_split:
                continue
            width = int(row.get("width") or row.get("map_size") or 0)
            if map_sizes and width not in map_sizes:
                continue
            row["width"] = width
            row["episode_id"] = row.get("episode_id") or row.get("replay_id") or row.get("fingerprint")
            row["teacher_team"] = row.get("teacher_team") or row.get("teacher_identity") or "unknown"
            row["state_step"] = int(row["state_step"])
            row["action_step"] = int(row["action_step"])
            row["teacher_player"] = int(row["teacher_player"])
            row["weight"] = float(row["weight"])
            grouped[Path(row.get("metric_file") or row["file"])].append(row)
            if max_rows and sum(len(v) for v in grouped.values()) >= max_rows:
                break
    for rows in grouped.values():
        rows.sort(key=lambda item: item["state_step"])
    return dict(grouped)


def _append_sample(buffers: dict, env_output: dict, target: dict, meta: dict, weight: float) -> None:
    buffers["weights"].append(torch.tensor(weight, dtype=torch.float32))
    buffers["critical_mask"].append(torch.tensor(float(meta.get("critical_mask", 0.0)), dtype=torch.float32))
    buffers["counterfactual_scale"].append(torch.tensor(float(meta.get("counterfactual_scale", 1.0)), dtype=torch.float32))
    sample_type = {"strict_dpo": 0, "near_weak_preference": 1, "critical_focal_bc": 2}.get(
        str(meta.get("sample_type", "critical_focal_bc")), 2
    )
    buffers["sample_type"].append(torch.tensor(sample_type, dtype=torch.int64))
    buffers["meta"].append({key: value for key, value in meta.items() if key != "rejected_target"})
    for key, value in env_output["obs"].items():
        buffers["obs"][key].append(value.squeeze(0).cpu())
    for key, value in env_output["info"]["available_actions_mask"].items():
        buffers["available_actions_mask"][key].append(value.squeeze(0).cpu())
    buffers["input_mask"].append(env_output["info"]["input_mask"].squeeze(0).cpu())
    for key, value in env_output["info"].get("role_bias_codes", {}).items():
        buffers["role_bias_codes"][key].append(value.squeeze(0).cpu())
    buffers["role_bias_scale"].append(env_output["info"]["role_bias_scale"].squeeze(0).cpu())
    for key, value in target.items():
        buffers["actions_taken"][key].append(torch.from_numpy(value).cpu())
    for key, value in meta["rejected_target"].items():
        buffers["rejected_actions"][key].append(torch.from_numpy(value).cpu())


def _empty_buffers() -> dict:
    return {
        "obs": defaultdict(list),
        "available_actions_mask": defaultdict(list),
        "actions_taken": defaultdict(list),
        "rejected_actions": defaultdict(list),
        "input_mask": [],
        "role_bias_codes": defaultdict(list),
        "role_bias_scale": [],
        "weights": [],
        "critical_mask": [],
        "counterfactual_scale": [],
        "sample_type": [],
        "meta": [],
    }


def _flush(buffers: dict, output_dir: Path, shard_index: int) -> int:
    if not buffers["weights"]:
        return shard_index
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "obs": {key: torch.stack(values) for key, values in buffers["obs"].items()},
        "available_actions_mask": {
            key: torch.stack(values) for key, values in buffers["available_actions_mask"].items()
        },
        "actions_taken": {
            key: torch.stack(values).to(torch.bool) for key, values in buffers["actions_taken"].items()
        },
        "rejected_actions": {
            key: torch.stack(values).to(torch.bool) for key, values in buffers["rejected_actions"].items()
        },
        "input_mask": torch.stack(buffers["input_mask"]),
        "role_bias_codes": {key: torch.stack(values) for key, values in buffers["role_bias_codes"].items()},
        "role_bias_scale": torch.stack(buffers["role_bias_scale"]),
        "weights": torch.stack(buffers["weights"]),
        "critical_mask": torch.stack(buffers["critical_mask"]),
        "counterfactual_scale": torch.stack(buffers["counterfactual_scale"]),
        "sample_type": torch.stack(buffers["sample_type"]),
        "meta": buffers["meta"],
    }
    out_path = output_dir / f"shard_{shard_index:05d}.pt"
    torch.save(shard, out_path)
    print(f"wrote {out_path} samples={len(buffers['weights'])}")
    return shard_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract imitation-learning tensor shards.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("dataset/processed/imitation_index_hq.csv"),
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=Path("outputs/auto_league_dagger_v10_shadow/best_agent"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed/imitation_shards_hq"),
    )
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--split", choices=["", "train", "validation", "calibration"], default="")
    parser.add_argument("--critical-min-scale-delta", type=float, default=0.02)
    parser.add_argument(
        "--role-assignment-config",
        type=Path,
        help="YAML containing role_assignment; overrides the agent model config for role-code extraction.",
    )
    parser.add_argument(
        "--map-sizes",
        default="",
        help="Comma-separated map sizes to keep, for example 12,16. Empty keeps all maps.",
    )
    args = parser.parse_args()

    flags = load_agent_flags(args.agent_dir)
    grouped = load_index(args.index, args.max_rows, parse_int_set(args.map_sizes), args.split)
    role_mapping = getattr(flags, "role_assignment", {})
    role_base_dir = None
    if args.role_assignment_config:
        role_payload = yaml.safe_load(args.role_assignment_config.read_text(encoding="utf-8")) or {}
        role_mapping = role_payload.get("role_assignment", role_payload)
        role_base_dir = args.role_assignment_config.parent
    role_config = RoleAssignmentConfig.from_mapping(role_mapping, base_dir=role_base_dir)
    buffers = _empty_buffers()
    shard_index = 0
    total = 0

    for replay_path, rows in grouped.items():
        with replay_path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        step_count = len(replay.get("stateful") or replay.get("steps") or [])
        if not step_count:
            continue
        env = build_manual_env(flags, replay_updates(replay, 0))
        role_builder = RoleBiasCodeBuilder(role_config)
        placeholder = action_placeholder(env)
        current_step = 0

        for row in rows:
            while current_step < row["state_step"]:
                current_step += 1
                advance_manual_env(
                    env,
                    current_step,
                    replay_updates(replay, current_step),
                )
            env_output = attach_role_bias_codes(env_output_for_current_state(env, placeholder), env, role_builder)
            actions = json.loads(
                row.get("preferred_actions_json") or row.get("actions_json") or "[]"
            )
            if not actions and replay.get("steps"):
                actions = replay["steps"][row["action_step"]][row["teacher_player"]].get("action") or []
            target = teacher_actions_to_mask(
                env.unwrapped[0].game_state,
                row["teacher_player"],
                actions,
            )
            if not any(value.any() for value in target.values()):
                continue
            rejected_actions = json.loads(row.get("rejected_actions_json") or "[]")
            if any(str(action).strip().startswith("bcity ") for action in rejected_actions):
                continue
            rejected_target = teacher_actions_to_mask(
                env.unwrapped[0].game_state,
                row["teacher_player"],
                rejected_actions,
            )
            _append_sample(
                buffers,
                env_output,
                target,
                {
                    **row,
                    "file": str(replay_path),
                    "episode_id": row["episode_id"],
                    "state_step": row["state_step"],
                    "action_step": row["action_step"],
                    "teacher_player": row["teacher_player"],
                    "teacher_team": row["teacher_team"],
                    "counterfactual_scale": float(row.get("counterfactual_scale", 1.0) or 1.0),
                    "weight_reason": row.get("weight_reason", ""),
                    "critical_mask": float(
                        abs(float(row.get("counterfactual_scale", 1.0) or 1.0) - 1.0)
                        >= args.critical_min_scale_delta
                    ),
                    "sample_type": row.get("sample_type", "critical_focal_bc"),
                    "rejected_target": rejected_target,
                },
                row["weight"],
            )
            total += 1
            if len(buffers["weights"]) >= args.shard_size:
                shard_index = _flush(buffers, args.output_dir, shard_index)
                buffers = _empty_buffers()

    shard_index = _flush(buffers, args.output_dir, shard_index)
    print(f"done: samples={total} shards={shard_index} output={args.output_dir}")


if __name__ == "__main__":
    main()
