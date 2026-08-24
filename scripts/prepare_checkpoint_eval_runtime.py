#!/usr/bin/env python3
"""Prepare reproducible agent copies for paired GPU evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import yaml


def set_device(agent_dir: Path, device: str) -> None:
    path = agent_dir / "lux_ai" / "rl_agent" / "rl_agent_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config["device"] = device
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def copy_agent(source: Path, destination: Path, device: str) -> None:
    if (destination / "main.py").is_file():
        set_device(destination, device)
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    set_device(destination, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-names", nargs="+", default=None)
    parser.add_argument("--opponent-root", type=Path, required=True)
    parser.add_argument("--best-agent", type=Path, required=True)
    parser.add_argument("--first-agent", type=Path, required=True)
    parser.add_argument("--stage350-agent", type=Path, required=True)
    parser.add_argument("--stage400-agent", type=Path, required=True)
    parser.add_argument(
        "--opponent-names",
        nargs="+",
        choices=("best_agent", "first", "stage350", "stage400"),
        default=("best_agent", "first", "stage350", "stage400"),
    )
    args = parser.parse_args()

    candidates = (
        (args.candidate_root / name for name in args.candidate_names)
        if args.candidate_names
        else args.candidate_root.iterdir()
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "main.py").exists():
            set_device(candidate, "cuda:0")

    args.opponent_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "best_agent": args.best_agent,
        "first": args.first_agent,
        "stage350": args.stage350_agent,
        "stage400": args.stage400_agent,
    }
    for name in args.opponent_names:
        source = sources[name]
        if not (source / "main.py").exists():
            raise FileNotFoundError(f"Agent package not found: {source}")
        destination = args.opponent_root / name
        copy_agent(source.resolve(), destination, "cuda:0")
        print(f"Prepared {name}: {destination} (cuda:0)")


if __name__ == "__main__":
    main()
