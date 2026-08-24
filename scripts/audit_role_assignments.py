#!/usr/bin/env python3
"""Audit SacrificialDecay assignments against stateful Lux replays."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


ROLE = "SacrificialDecay"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def player_cities(state: Dict[str, Any], player: int) -> Dict[str, Dict[str, Any]]:
    return {
        city_id: city
        for city_id, city in state.get("cities", {}).items()
        if int(city.get("team", -1)) == player
    }


def tile_count(city: Optional[Dict[str, Any]]) -> int:
    return len((city or {}).get("cityCells", []))


def city_at(states: List[Dict[str, Any]], turn: int, player: int, city_id: str):
    if turn < 0 or turn >= len(states):
        return None
    return player_cities(states[turn], player).get(city_id)


def first_missing_turn(
    states: List[Dict[str, Any]], start: int, end: int, player: int, city_id: str
) -> Optional[int]:
    for turn in range(start + 1, min(end, len(states) - 1) + 1):
        if city_at(states, turn, player, city_id) is None:
            return turn
    return None


def city_night_tile_loss(
    states: List[Dict[str, Any]], start: int, end: int, player: int, city_id: str
) -> int:
    loss = 0
    for turn in range(start, min(end, len(states) - 1)):
        if turn % 40 not in range(30, 40):
            continue
        before = city_at(states, turn, player, city_id)
        after = city_at(states, turn + 1, player, city_id)
        loss += max(tile_count(before) - tile_count(after), 0)
    return loss


def resolve_replay(sidecar_path: Path, sidecar: Dict[str, Any]) -> Optional[Path]:
    replay_file = sidecar.get("replay_file")
    candidates = []
    if replay_file:
        candidates.append(sidecar_path.parent / replay_file)
    stem = sidecar_path.name.removesuffix(".roles.json")
    candidates.extend(
        sidecar_path.parent / f"{stem}{suffix}"
        for suffix in (".json", ".commands.json")
    )
    return next((path for path in candidates if path.is_file()), None)


def transition_events(frames: List[Dict[str, Any]]) -> Iterable[tuple]:
    previous = set()
    for frame in frames:
        current = {
            city["id"]
            for city in frame.get("cities", [])
            if city.get("role") == ROLE or city.get("abandon") is True
        }
        by_id = {city["id"]: city for city in frame.get("cities", [])}
        for city_id in sorted(current - previous):
            yield frame, by_id[city_id]
        previous = current


def audit_one(sidecar_path: Path) -> tuple:
    sidecar = load_json(sidecar_path)
    replay_path = resolve_replay(sidecar_path, sidecar)
    if replay_path is None:
        raise FileNotFoundError(f"Replay not found for {sidecar_path}")
    replay = load_json(replay_path)
    states = replay.get("stateful") or []
    if not states:
        raise ValueError(f"Replay is not stateful: {replay_path}")

    frames = sorted(sidecar.get("frames", []), key=lambda frame: int(frame["turn"]))
    player = int(sidecar.get("player", frames[0].get("player", -1) if frames else -1))
    checkpoint = sidecar_path.parent.parent.name
    frame_stats = []
    sacrificial_by_turn = {}
    for frame in frames:
        cities = frame.get("cities", [])
        sacrificed = [c for c in cities if c.get("role") == ROLE or c.get("abandon") is True]
        turn = int(frame["turn"])
        sacrificial_by_turn[turn] = {c["id"] for c in sacrificed}
        frame_stats.append((turn, len(sacrificed), len(cities)))

    events = []
    for frame, city in transition_events(frames):
        turn = int(frame["turn"])
        city_id = city["id"]
        initial = city_at(states, turn, player, city_id)
        initial_tiles = tile_count(initial) or len(city.get("tiles", []))
        row = {
            "checkpoint": checkpoint,
            "sidecar": str(sidecar_path),
            "replay": str(replay_path),
            "map_size": int(sidecar.get("width") or replay.get("width") or frame.get("map_size", 0)),
            "seed": replay.get("seed"),
            "player": player,
            "turn": turn,
            "city_id": city_id,
            "reason": city.get("reason", ""),
            "nights_of_fuel": city.get("nights_of_fuel"),
            "initial_tiles": initial_tiles,
            "death_turn_20": first_missing_turn(states, turn, turn + 20, player, city_id),
            "death_turn_40": first_missing_turn(states, turn, turn + 40, player, city_id),
        }
        for horizon in (20, 40):
            future_turn = min(turn + horizon, len(states) - 1)
            future = city_at(states, future_turn, player, city_id)
            row[f"observable_{horizon}"] = int(turn + horizon < len(states))
            row[f"alive_{horizon}"] = int(future is not None)
            row[f"tiles_{horizon}"] = tile_count(future)
            row[f"tile_retention_{horizon}"] = (
                tile_count(future) / initial_tiles if initial_tiles else 0.0
            )
            row[f"night_tile_loss_{horizon}"] = city_night_tile_loss(
                states, turn, turn + horizon, player, city_id
            )
        row["turn_zero_assignment"] = int(turn == 0)
        events.append(row)

    night_losses = []
    historically_sacrificial = set()
    for turn in range(min(len(states) - 1, 360)):
        historically_sacrificial.update(sacrificial_by_turn.get(turn, set()))
        if turn % 40 not in range(30, 40):
            continue
        before = player_cities(states[turn], player)
        after = player_cities(states[turn + 1], player)
        loss = sum(max(tile_count(city) - tile_count(after.get(cid)), 0) for cid, city in before.items())
        if loss:
            currently_sacrificial = sacrificial_by_turn.get(turn, set())
            current_attributed = sum(
                max(tile_count(city) - tile_count(after.get(cid)), 0)
                for cid, city in before.items()
                if cid in currently_sacrificial
            )
            historical_attributed = sum(
                max(tile_count(city) - tile_count(after.get(cid)), 0)
                for cid, city in before.items()
                if cid in historically_sacrificial
            )
            night_losses.append({
                "turn": turn,
                "loss": loss,
                "current_sacrificial_loss": current_attributed,
                "historical_sacrificial_loss": historical_attributed,
            })

    ratios = [sacrificed / total for _, sacrificed, total in frame_stats if total]
    over_budget = [turn for turn, sacrificed, total in frame_stats if total and sacrificed > max(1, total * 0.10)]
    summary = {
        "checkpoint": checkpoint,
        "sidecar": str(sidecar_path),
        "replay": str(replay_path),
        "map_size": int(sidecar.get("width") or replay.get("width") or 0),
        "seed": replay.get("seed"),
        "player": player,
        "frames": len(frames),
        "sacrificial_events": len(events),
        "turn_zero_events": sum(row["turn_zero_assignment"] for row in events),
        "mean_sacrificial_share": mean(ratios) if ratios else 0.0,
        "max_sacrificial_share": max(ratios, default=0.0),
        "over_budget_turns": len(over_budget),
        "first_over_budget_turn": min(over_budget, default=None),
        "night_city_tile_loss": sum(row["loss"] for row in night_losses),
        "night_loss_while_sacrificial": sum(row["current_sacrificial_loss"] for row in night_losses),
        "night_loss_after_sacrificial": sum(row["historical_sacrificial_loss"] for row in night_losses),
    }
    return summary, events


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(summaries: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    observable_40 = [row for row in events if row["observable_40"]]
    by_checkpoint = {}
    for checkpoint in sorted({row["checkpoint"] for row in summaries}):
        replay_rows = [row for row in summaries if row["checkpoint"] == checkpoint]
        event_rows = [row for row in events if row["checkpoint"] == checkpoint]
        observed = [row for row in event_rows if row["observable_40"]]
        night_loss = sum(row["night_city_tile_loss"] for row in replay_rows)
        attributed = sum(row["night_loss_while_sacrificial"] for row in replay_rows)
        historical_attributed = sum(row["night_loss_after_sacrificial"] for row in replay_rows)
        by_checkpoint[checkpoint] = {
            "replays": len(replay_rows),
            "sacrificial_events": len(event_rows),
            "turn_zero_event_rate": sum(r["turn_zero_assignment"] for r in event_rows) / max(len(event_rows), 1),
            "survival_40_rate": sum(r["alive_40"] for r in observed) / max(len(observed), 1),
            "mean_tile_retention_40": mean([r["tile_retention_40"] for r in observed]) if observed else 0.0,
            "mean_sacrificial_share": mean([r["mean_sacrificial_share"] for r in replay_rows]) if replay_rows else 0.0,
            "max_sacrificial_share": max([r["max_sacrificial_share"] for r in replay_rows], default=0.0),
            "over_budget_turns": sum(r["over_budget_turns"] for r in replay_rows),
            "night_city_tile_loss": night_loss,
            "night_loss_while_sacrificial_rate": attributed / max(night_loss, 1),
            "night_loss_after_sacrificial_rate": historical_attributed / max(night_loss, 1),
        }
    return {
        "replays": len(summaries),
        "sacrificial_events": len(events),
        "observable_40_events": len(observable_40),
        "reasons": dict(Counter(row["reason"] for row in events)),
        "by_checkpoint": by_checkpoint,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Role Assignment Audit",
        "",
        f"Replays: {report['replays']}",
        f"Sacrificial transitions: {report['sacrificial_events']}",
        "",
        "| Checkpoint | Replays | Events | Turn-0 | Survive 40 | Retain tiles 40 | Mean share | Max share | Over budget | Night loss after label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["by_checkpoint"].items():
        lines.append(
            f"| {name} | {row['replays']} | {row['sacrificial_events']} | "
            f"{row['turn_zero_event_rate']:.1%} | {row['survival_40_rate']:.1%} | "
            f"{row['mean_tile_retention_40']:.1%} | {row['mean_sacrificial_share']:.1%} | "
            f"{row['max_sacrificial_share']:.1%} | {row['over_budget_turns']} | "
            f"{row['night_loss_after_sacrificial_rate']:.1%} |"
        )
    lines.extend([
        "",
        "A 40-turn survivor is not automatically a false positive; it is a review candidate.",
        "The 10% budget allows at least one sacrificial city per turn.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_root.rglob("*.roles.json"))
    if not paths:
        raise FileNotFoundError(f"No *.roles.json under {args.input_root}")
    summaries, events, errors = [], [], []
    for path in paths:
        try:
            summary, event_rows = audit_one(path)
            summaries.append(summary)
            events.extend(event_rows)
        except Exception as exc:  # Keep a full batch auditable when one replay failed.
            errors.append({"sidecar": str(path), "error": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = aggregate(summaries, events)
    report["errors"] = errors
    write_csv(args.output_dir / "replay_summary.csv", summaries)
    write_csv(args.output_dir / "sacrificial_events.csv", events)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "AUDIT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
