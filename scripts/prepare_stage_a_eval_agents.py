#!/usr/bin/env python3
"""Package the constrained Stage-A checkpoint and its exact zero-delta baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_checkpoint_agents import build_agent, resolved_model_config  # noqa: E402


def load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    return state


def zero_local_projection(state: dict) -> dict:
    result = {key: value.detach().clone() for key, value in state.items()}
    matched = 0
    for key, value in result.items():
        if key.startswith("role_local_adapter.output_heads."):
            value.zero_()
            matched += 1
    if not matched:
        raise ValueError("No role local output projection was found in baseline state")
    return result


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch-checkpoint", type=Path,
        default=Path("outputs/role_conditioned_bc_stageA_constrained/epoch_01.pt"),
    )
    parser.add_argument(
        "--baseline-agent", type=Path,
        default=Path("outputs/checkpoint_selection/agents/local_05120"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/checkpoint_selection/agents"),
    )
    parser.add_argument("--max-delta", type=float, default=0.05)
    args = parser.parse_args()

    epoch_path = (ROOT / args.epoch_checkpoint).resolve()
    baseline_agent = (ROOT / args.baseline_agent).resolve()
    output_root = (ROOT / args.output_root).resolve()
    artifacts = ROOT / "outputs" / "role_conditioned_bc_stageA_constrained" / "eval_artifacts"
    baseline_checkpoint = artifacts / "stageA_baseline.pt"
    epoch_export = artifacts / "stageA_epoch01.pt"
    save_state(
        baseline_checkpoint,
        zero_local_projection(load_state(baseline_agent / "lux_ai/rl_agent/candidate_weights.pt")),
    )
    save_state(epoch_export, load_state(epoch_path))

    config = resolved_model_config(ROOT, role_local=True)
    config["role_local_max_delta"] = float(args.max_delta)
    config["spatial_risk_sidecar_enabled"] = False
    config["risk_gate_enabled"] = False
    template = ROOT / "outputs/submission_packages/best_agent"
    for label, checkpoint in (
        ("stageA_baseline", baseline_checkpoint),
        ("stageA_epoch01", epoch_export),
    ):
        build_agent(ROOT, template, output_root / label, checkpoint, config)
        config_path = output_root / label / "lux_ai/rl_agent/config.yaml"
        runtime_model_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        runtime_model_config["role_local_max_delta"] = float(args.max_delta)
        runtime_model_config["spatial_risk_sidecar_enabled"] = False
        runtime_model_config["risk_gate_enabled"] = False
        config_path.write_text(
            yaml.safe_dump(runtime_model_config, sort_keys=False), encoding="utf-8"
        )
        print(f"Prepared {label}: {output_root / label}")


if __name__ == "__main__":
    main()
