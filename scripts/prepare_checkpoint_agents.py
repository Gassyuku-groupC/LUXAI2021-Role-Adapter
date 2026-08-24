#!/usr/bin/env python3
"""Build executable agent packages for BC/APPO checkpoint comparison."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


DEFAULT_CHECKPOINTS = {
    "bc": "outputs/first_actor_sidecar_bc_v3/best.pt",
    "10816": "10816_weights.pt",
    "20128": "20128_weights.pt",
    "30272": "30272_weights.pt",
    "40288": "40288_weights.pt",
    "50112": "50112_weights.pt",
    "60288": "60288_weights.pt",
    "70560": "70560_weights.pt",
}

ROLE_BIAS_PREFIX = "role_bias_layer.bias_params."


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(path)


def load_model_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    state = checkpoint["model_state_dict"]
    nonfinite = [
        name for name, value in state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite checkpoint tensors in {path}: {nonfinite[:10]}")
    return state


def split_role_bias_state(state: dict) -> tuple[dict, dict[str, float]]:
    model_state = {}
    role_biases = {}
    for name, value in state.items():
        if not name.startswith(ROLE_BIAS_PREFIX):
            model_state[name] = value
            continue
        bias_name = name.removeprefix(ROLE_BIAS_PREFIX)
        if not torch.is_tensor(value) or value.numel() != 1:
            raise ValueError(f"Role bias must be a scalar tensor: {name}")
        role_biases[bias_name] = float(value.detach().cpu())
    return model_state, role_biases


def is_role_local_state(state: dict) -> bool:
    return any(name.startswith("role_local_adapter.") for name in state)


def resolved_model_config(root: Path, role_local: bool = False) -> dict:
    with initialize_config_dir(config_dir=str((root / "conf").resolve()), version_base=None):
        config = compose(
            config_name="conv_role_curriculum_stage4" if role_local else "conv_sidecar_appo_vtrace"
        )
    data = OmegaConf.to_container(config, resolve=True)
    data["load_dir"] = None
    data["checkpoint_file"] = "candidate_weights.pt"
    data["student_pretrain_checkpoint"] = None
    data["weights_only"] = True
    return data


def build_agent(
    root: Path,
    template: Path,
    output: Path,
    checkpoint: Path,
    config: dict,
    preserve_runtime_config: bool = False,
    enable_role_adapter: bool = False,
) -> None:
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(
        template,
        output,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "strategy_scorers"),
    )
    shutil.copytree(
        root / "lux_ai",
        output / "lux_ai",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt"),
    )
    shutil.copy2(root / "main.py", output / "main.py")
    rl_dir = output / "lux_ai" / "rl_agent"
    raw_state = load_model_state(checkpoint)
    role_local = is_role_local_state(raw_state)
    if role_local:
        model_state, learned_role_biases = raw_state, {}
        config = resolved_model_config(root, role_local=True)
    else:
        model_state, learned_role_biases = split_role_bias_state(raw_state)
    torch.save({"model_state_dict": model_state}, rl_dir / "candidate_weights.pt")
    (rl_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if preserve_runtime_config:
        shutil.copy2(
            template / "lux_ai" / "rl_agent" / "rl_agent_config.yaml",
            rl_dir / "rl_agent_config.yaml",
        )
    else:
        agent_config_path = rl_dir / "rl_agent_config.yaml"
        agent_config = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
        agent_config["runtime_scorer_gate_enabled"] = False
        agent_config["data_augmentations"] = []
        agent_config_path.write_text(
            yaml.safe_dump(agent_config, sort_keys=False), encoding="utf-8"
        )
    if enable_role_adapter or role_local:
        agent_config_path = rl_dir / "rl_agent_config.yaml"
        agent_config = yaml.safe_load(agent_config_path.read_text(encoding="utf-8")) or {}
        role_config = agent_config.setdefault("role_assignment", {})
        role_config.update({
            "enabled": True,
            "dry_run_logging": False,
            "bias_enabled": True,
            "learnable_biases": False,
            "bias_params_path": "role_city_bias_params.yaml",
            "annotate_summary": False,
            "cooldown_turns": 5,
            "firefighter_override_cooldown": True,
            "preserve_build_city_logit": True,
            "bias_disabled_map_sizes": [12],
            "bias_disabled_players_by_map": {},
            "bias_scale_by_map_size": {12: 0.0, 16: 0.60, 24: 0.50, 32: 0.30},
            "max_biased_workers_by_map_size": {12: 32, 16: 64, 24: 64, 32: 64},
            "safety_only_map_sizes": [],
            "update_time_budget_seconds": 1.5,
        })
        agent_config_path.write_text(
            yaml.safe_dump(agent_config, sort_keys=False), encoding="utf-8"
        )
        if learned_role_biases:
            (rl_dir / "role_city_bias_params.yaml").write_text(
                yaml.safe_dump(
                    {"role_city_bias_params": learned_role_biases}, sort_keys=True
                ),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--template-agent",
        type=Path,
        default=Path("outputs/submission_packages/best_agent"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents"),
    )
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint)
    parser.add_argument("--disable-risk-gate", action="store_true")
    parser.add_argument("--preserve-runtime-config", action="store_true")
    parser.add_argument("--enable-role-adapter", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    template = (root / args.template_agent).resolve() if not args.template_agent.is_absolute() else args.template_agent
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    checkpoints = dict(args.checkpoint or DEFAULT_CHECKPOINTS.items())
    config = resolved_model_config(root)
    if args.disable_risk_gate:
        config["risk_gate_enabled"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, raw_path in checkpoints.items():
        raw_path = Path(raw_path)
        checkpoint = (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint {label}: {checkpoint}")
        destination = output_dir / label
        build_agent(
            root,
            template,
            destination,
            checkpoint,
            config,
            preserve_runtime_config=args.preserve_runtime_config,
            enable_role_adapter=args.enable_role_adapter,
        )
        print(f"Prepared {label}: {checkpoint} -> {destination}")


if __name__ == "__main__":
    main()
