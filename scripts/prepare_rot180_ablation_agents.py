#!/usr/bin/env python3
"""Create inference-identical Role agents that differ only in Rot180 TTA policy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


VARIANTS = {
    "role_rot180_all": {"data_augmentations": ["Rot180"], "disable_above": 0},
    "role_rot180_16_only": {"data_augmentations": ["Rot180"], "disable_above": 16},
    "role_no_rot180": {"data_augmentations": [], "disable_above": 0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-agent",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents/role_05376"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_selection/agents"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = args.source_agent if args.source_agent.is_absolute() else root / args.source_agent
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config_relpath = Path("lux_ai/rl_agent/rl_agent_config.yaml")
    if not (source / "main.py").is_file() or not (source / config_relpath).is_file():
        raise FileNotFoundError(f"Invalid source agent: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, settings in VARIANTS.items():
        destination = output_dir / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        utils_path = destination / "lux_ai/utils.py"
        utils_text = utils_path.read_text(encoding="utf-8")
        utils_path.write_text(
            utils_text.replace("val['duration']:.2f", "val['duration']:.4f"),
            encoding="utf-8",
        )
        config_path = destination / config_relpath
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["data_augmentations"] = settings["data_augmentations"]
        config["runtime_no_augmentations_above_map_size"] = settings["disable_above"]
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"Prepared {name}: augmentations={settings['data_augmentations']} "
              f"disable_above={settings['disable_above']}")


if __name__ == "__main__":
    main()
