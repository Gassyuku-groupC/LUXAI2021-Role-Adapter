#!/usr/bin/env python3
"""Compare Group C wins and losses in the first-battle replay set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


CHECKPOINT_TURNS = (30, 40, 80, 120, 160, 200, 240, 280, 320, 360)


def state_metrics(state: Dict[str, Any], player: int) -> Dict[str, float]:
    cities = [city for city in state["cities"].values() if int(city["team"]) == player]
    team = state["teamStates"][str(player)]
    return {
        "city_tiles": sum(len(city["cityCells"]) for city in cities),
        "city_count": len(cities),
        "units": len(team["units"]),
        "research": int(team["researchPoints"]),
        "fuel": sum(float(city["fuel"]) for city in cities),
        "upkeep": sum(float(city["lightupkeep"]) for city in cities),
    }


def state_at(states: List[Dict[str, Any]], turn: int) -> Dict[str, Any]:
    return states[min(max(turn, 0), len(states) - 1)]


def first_persistent_deficit(values: List[int], window: int = 20) -> int:
    for index in range(max(len(values) - window + 1, 0)):
        if all(value < 0 for value in values[index:index + window]):
            return index
    return -1


def action_counts(replay: Dict[str, Any], player: int) -> Dict[str, int]:
    result = Counter()
    for turn, commands in enumerate(replay.get("allCommands", [])):
        phase = "early" if turn < 120 else "mid" if turn < 240 else "late"
        for item in commands:
            if int(item.get("agentID", -1)) != player:
                continue
            command = str(item.get("command", "")).split(" ", 1)[0]
            result[f"{phase}_{command}"] += 1
    return dict(result)


def analyze_game(
    row: Dict[str, str], stateful_path: Path, command_path: Path
) -> Dict[str, Any]:
    replay = json.loads(stateful_path.read_text(encoding="utf-8-sig"))
    command_replay = json.loads(command_path.read_text(encoding="utf-8-sig"))
    states = replay["stateful"]
    player = 0 if row["first"] == "C" else 1
    opponent = 1 - player
    ours = [state_metrics(state, player) for state in states]
    theirs = [state_metrics(state, opponent) for state in states]
    city_margins = [int(a["city_tiles"] - b["city_tiles"]) for a, b in zip(ours, theirs)]
    unit_margins = [int(a["units"] - b["units"]) for a, b in zip(ours, theirs)]

    night_losses = []
    risk_tiles = []
    for turn in range(len(states) - 1):
        if turn % 40 == 29:
            cities = [
                city for city in states[turn]["cities"].values()
                if int(city["team"]) == player
            ]
            risk_tiles.append(sum(
                len(city["cityCells"])
                for city in cities
                if float(city["fuel"]) < float(city["lightupkeep"]) * 10.0
            ))
        if (turn + 1) % 40 < 30:
            continue
        loss = int(ours[turn]["city_tiles"] - ours[turn + 1]["city_tiles"])
        if loss > 0:
            night_losses.append((turn + 1, loss))

    output = {
        "map_size": int(row["map_size"]),
        "first": row["first"],
        "second": row["second"],
        "opponent": row["second"] if row["first"] == "C" else row["first"],
        "side": player,
        "outcome": "draw" if row["draw"].upper() == "TRUE" else "win" if row["winner"] == "C" else "loss",
        "seed": replay.get("seed"),
        "file": str(stateful_path),
        "turns": int(states[-1]["turn"]),
        "first_city_deficit_turn": first_persistent_deficit(city_margins),
        "first_unit_deficit_turn": first_persistent_deficit(unit_margins),
        "night_loss_total": sum(loss for _, loss in night_losses),
        "max_night_loss": max((loss for _, loss in night_losses), default=0),
        "max_night_loss_turn": max(night_losses, key=lambda item: item[1])[0] if night_losses else -1,
        "max_pre_night_at_risk_tiles": max(risk_tiles, default=0),
        "mean_pre_night_at_risk_tiles": mean(risk_tiles) if risk_tiles else 0.0,
        "final_city_margin": city_margins[-1],
        "final_unit_margin": unit_margins[-1],
        "final_research_margin": int(ours[-1]["research"] - theirs[-1]["research"]),
    }
    for turn in CHECKPOINT_TURNS:
        index = min(turn, len(states) - 1)
        output[f"city_margin_t{turn}"] = city_margins[index]
        output[f"unit_margin_t{turn}"] = unit_margins[index]
        output[f"research_margin_t{turn}"] = int(ours[index]["research"] - theirs[index]["research"])
    output.update(action_counts(command_replay, player))
    return output


def average(rows: Iterable[Dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0)) for row in rows]
    return mean(values) if values else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["map_size"], row["outcome"])].append(row)
    output = []
    for (map_size, outcome), items in sorted(groups.items()):
        output.append({
            "map_size": map_size,
            "outcome": outcome,
            "games": len(items),
            "city_margin_t120": average(items, "city_margin_t120"),
            "city_margin_t240": average(items, "city_margin_t240"),
            "final_city_margin": average(items, "final_city_margin"),
            "unit_margin_t120": average(items, "unit_margin_t120"),
            "final_unit_margin": average(items, "final_unit_margin"),
            "night_loss_total": average(items, "night_loss_total"),
            "max_night_loss": average(items, "max_night_loss"),
            "pre_night_at_risk_tiles": average(items, "mean_pre_night_at_risk_tiles"),
            "early_bcity": average(items, "early_bcity"),
            "mid_bcity": average(items, "mid_bcity"),
            "late_bcity": average(items, "late_bcity"),
            "early_bw": average(items, "early_bw"),
            "early_r": average(items, "early_r"),
            "early_t": average(items, "early_t"),
        })
    return output


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> str:
    losses = [row for row in rows if row["outcome"] == "loss"]
    draws = [row for row in rows if row["outcome"] == "draw"]
    opponent_losses = Counter(row["opponent"] for row in losses)
    lines = [
        "# Group C Replay Loss Audit",
        "",
        f"Games: {len(rows)}; losses: {len(losses)}; draws: {len(draws)}.",
        f"Losses by opponent: {dict(opponent_losses)}.",
        "",
        "## Loss Games",
        "",
        "| Map | Side | Opponent | City@120 | City@240 | Final city | Final unit | Night loss | Worst night | First city deficit |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in losses:
        lines.append(
            f"| {row['map_size']} | P{row['side']} | {row['opponent']} | "
            f"{row['city_margin_t120']} | {row['city_margin_t240']} | {row['final_city_margin']} | "
            f"{row['final_unit_margin']} | {row['night_loss_total']} | {row['max_night_loss']} "
            f"@{row['max_night_loss_turn']} | {row['first_city_deficit_turn']} |"
        )
    lines.extend([
        "",
        "## Win/Loss Control Means",
        "",
        "| Map | Outcome | Games | City@120 | City@240 | Final city | Night loss | At-risk tiles | Early bcity | Early bw | Early transfer |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            f"| {row['map_size']} | {row['outcome']} | {row['games']} | "
            f"{row['city_margin_t120']:.1f} | {row['city_margin_t240']:.1f} | "
            f"{row['final_city_margin']:.1f} | {row['night_loss_total']:.1f} | "
            f"{row['pre_night_at_risk_tiles']:.1f} | {row['early_bcity']:.1f} | "
            f"{row['early_bw']:.1f} | {row['early_t']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--command-root", type=Path, required=True)
    parser.add_argument("--stateful-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for result in csv.DictReader(args.results.open(encoding="utf-8-sig")):
        if result["first"] != "C" and result["second"] != "C":
            continue
        name = f"{result['first']}_{result['second']}.json"
        stateful = args.stateful_root / f"{result['map_size']}x{result['map_size']}_{name}"
        command = args.command_root / f"{result['map_size']}x{result['map_size']}" / name
        rows.append(analyze_game(result, stateful, command))
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "games.csv", rows)
    write_csv(args.output_dir / "win_loss_control.csv", summary)
    (args.output_dir / "report.json").write_text(
        json.dumps({"games": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(rows, summary), encoding="utf-8")
    print(render_report(rows, summary))


if __name__ == "__main__":
    main()
