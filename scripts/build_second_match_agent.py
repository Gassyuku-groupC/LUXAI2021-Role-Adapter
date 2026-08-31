#!/usr/bin/env python3
"""Build the validated routed_teacher_final deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--base-agent",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents/role_05376_nofs"),
    )
    parser.add_argument(
        "--map24-agent",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents/log_03584"),
    )
    parser.add_argument(
        "--map12-agent",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents/er100_35072"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission_packages/routed_teacher_final"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    base_agent = (root / args.base_agent).resolve() if not args.base_agent.is_absolute() else args.base_agent
    map24_agent = (root / args.map24_agent).resolve() if not args.map24_agent.is_absolute() else args.map24_agent
    map12_agent = None
    if args.map12_agent is not None:
        map12_agent = (
            (root / args.map12_agent).resolve()
            if not args.map12_agent.is_absolute()
            else args.map12_agent
        )
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists; pass --force to replace it: {output}")
        shutil.rmtree(output)

    shutil.copytree(base_agent, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(
        root / "lux_ai",
        output / "lux_ai",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pt",
            "config.yaml",
            "rl_agent_config.yaml",
            "role_city_bias_params.yaml",
        ),
    )
    shutil.copy2(root / "main.py", output / "main.py")

    rl_dir = output / "lux_ai" / "rl_agent"
    map24_checkpoint = rl_dir / "map24_candidate_weights.pt"
    map24_config = rl_dir / "config_map24.yaml"
    shutil.copy2(map24_agent / "lux_ai" / "rl_agent" / "candidate_weights.pt", map24_checkpoint)
    shutil.copy2(map24_agent / "lux_ai" / "rl_agent" / "config.yaml", map24_config)

    map12_checkpoint = None
    map12_config = None
    if map12_agent is not None:
        map12_checkpoint = rl_dir / "map12_candidate_weights.pt"
        map12_config = rl_dir / "config_map12.yaml"
        shutil.copy2(map12_agent / "lux_ai" / "rl_agent" / "candidate_weights.pt", map12_checkpoint)
        shutil.copy2(map12_agent / "lux_ai" / "rl_agent" / "config.yaml", map12_config)

    runtime_path = rl_dir / "rl_agent_config.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
    runtime["data_augmentations"] = ["Rot180"]
    runtime["runtime_scorer_gate_enabled"] = False
    role = runtime.setdefault("role_assignment", {})
    role["enabled"] = True
    role["bias_enabled"] = True
    role["bias_disabled_map_sizes"] = [12]
    routes = {
        24: {
            "model_config": map24_config.name,
            "checkpoint": map24_checkpoint.name,
        }
    }
    if map12_checkpoint is not None and map12_config is not None:
        routes[12] = {
            "model_config": map12_config.name,
            "checkpoint": map12_checkpoint.name,
        }
    runtime["model_route_by_map_size"] = routes
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")

    role_source = (rl_dir / "role_assignment.py").read_text(encoding="utf-8")
    if "FUEL_STATION =" in role_source:
        raise RuntimeError("Second-match package must not contain active FuelStation logic.")
    if "Rot180" not in runtime.get("data_augmentations", []):
        raise RuntimeError("Second-match package must enable Rot180.")

    artifacts = {
        "base_checkpoint": rl_dir / "candidate_weights.pt",
        "base_model_config": rl_dir / "config.yaml",
        "map24_checkpoint": map24_checkpoint,
        "map24_model_config": map24_config,
        "runtime_config": runtime_path,
        "role_bias_config": rl_dir / "role_city_bias_params.yaml",
    }
    if map12_checkpoint is not None and map12_config is not None:
        artifacts["map12_checkpoint"] = map12_checkpoint
        artifacts["map12_model_config"] = map12_config
    manifest = {
        "identity": "routed_teacher_final",
        "rot180": True,
        "risk_gate_enabled": False,
        "fuel_station_enabled": False,
        "routes": {
            "12": "map12 specialist" if map12_agent is not None else "base Actor with Role disabled",
            "16": base_agent.name,
            "24": map24_agent.name,
            "32": base_agent.name,
        },
        "sha256": {name: sha256(path) for name, path in artifacts.items()},
    }
    (output / "SECOND_MATCH_AGENT.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Built {output}")
    map12_label = map12_agent.name if map12_agent is not None else "base Actor with Role disabled"
    print(
        f"Routes: 12={map12_label}, 16={base_agent.name}, "
        f"24={map24_agent.name}, 32={base_agent.name}"
    )
    print("Rot180=ENABLED | RiskGate=DISABLED | FuelStation=REMOVED")


if __name__ == "__main__":
    main()
