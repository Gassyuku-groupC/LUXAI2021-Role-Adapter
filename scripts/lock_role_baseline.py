#!/usr/bin/env python3
"""Create an immutable local copy and hash manifest for the promoted Role agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LOCKED_FILES = (
    "lux_ai/rl_agent/candidate_weights.pt",
    "lux_ai/rl_agent/config.yaml",
    "lux_ai/rl_agent/rl_agent_config.yaml",
    "lux_ai/rl_agent/role_city_bias_params.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "BASELINE_LOCK.json"):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/checkpoint_selection/agents/role_05376"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_packages/role_05376_rot180_locked"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = args.source if args.source.is_absolute() else root / args.source
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Locked baseline already exists: {output}")
        shutil.rmtree(output)
    for relative in (Path("main.py"), *(Path(name) for name in LOCKED_FILES)):
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)

    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime = (output / "lux_ai/rl_agent/rl_agent_config.yaml").read_text(encoding="utf-8")
    if "Rot180" not in runtime:
        raise ValueError("The promoted baseline must retain Rot180 inference augmentation")
    manifest = {
        "schema_version": 1,
        "identity": "role_05376+Rot180",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "git_branch": git_value(root, "branch", "--show-current"),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_dirty_at_lock": bool(git_value(root, "status", "--porcelain")),
        "package_tree_sha256": tree_sha256(output),
        "files": {name: sha256(output / name) for name in LOCKED_FILES},
    }
    (output / "BASELINE_LOCK.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
