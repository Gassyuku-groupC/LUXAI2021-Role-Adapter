#!/usr/bin/env python3
"""Summarize early-game Lux metrics from stateful checkpoint replays."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REPLAY_NAME = re.compile(r"_p(?P<side>[01])\.json$")


def team_counts(frame: dict, team: int) -> tuple[int, int]:
    city_tiles = sum(
        len(city.get("cityCells", ()))
        for city in frame.get("cities", {}).values()
        if int(city.get("team", -1)) == team
    )
    units = frame.get("teamStates", {}).get(str(team), {}).get("units", {}).values()
    workers = sum(int(unit.get("type", -1)) == 0 for unit in units)
    return city_tiles, workers


def frame_at(frames: list[dict], turn: int) -> dict:
    by_turn = {int(frame.get("turn", -1)): frame for frame in frames}
    if turn in by_turn:
        return by_turn[turn]
    return min(frames, key=lambda frame: abs(int(frame.get("turn", -1)) - turn))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for replay_path in sorted(args.root.rglob("map_12x12_*.json")):
        if replay_path.name.endswith((".commands.json", ".roles.json")):
            continue
        match = REPLAY_NAME.search(replay_path.name)
        if match is None:
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        frames = replay.get("stateful") or []
        if not frames:
            continue
        side = int(match.group("side"))
        turn30 = frame_at(frames, 30)
        turn40 = frame_at(frames, 40)
        city30, workers30 = team_counts(turn30, side)
        city40, _ = team_counts(turn40, side)
        ranks = replay.get("results", {}).get("ranks", [])
        rank = ranks[side].get("rank") if len(ranks) > side and isinstance(ranks[side], dict) else (
            ranks[side] if len(ranks) > side else None
        )
        win = int(rank is not None and int(rank) == 1)
        checkpoint = replay_path.parents[1].name
        rows.append(
            {
                "checkpoint": checkpoint,
                "side": side,
                "win": win,
                "city_turn30": city30,
                "workers_turn30": workers30,
                "night1_zero_city_loss": int(city40 >= city30),
            }
        )

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["checkpoint"], []).append(row)
    summary = []
    for checkpoint, games in sorted(grouped.items()):
        p0 = [game for game in games if game["side"] == 0]
        summary.append(
            {
                "checkpoint": checkpoint,
                "games": len(games),
                "p0_games": len(p0),
                "p0_win_rate": sum(game["win"] for game in p0) / len(p0) if p0 else 0.0,
                "mean_city_turn30": sum(game["city_turn30"] for game in games) / len(games),
                "mean_workers_turn30": sum(game["workers_turn30"] for game in games) / len(games),
                "night1_survival_rate": sum(game["night1_zero_city_loss"] for game in games) / len(games),
            }
        )

    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys() if summary else [])
            if summary:
                writer.writeheader()
                writer.writerows(summary)


if __name__ == "__main__":
    main()
