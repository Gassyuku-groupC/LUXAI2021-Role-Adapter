from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LeagueOpponent:
    name: str
    agent_path: str
    kind: str = "historical"
    games: int = 0
    wins: int = 0
    night_loss_mean: float = 0.0

    @property
    def learner_win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.5


class PFSPOpponentSampler:
    """Prioritize hard opponents while retaining support over the full league."""

    def __init__(
            self,
            opponents: Sequence[LeagueOpponent],
            hard_win_rate: float = 0.45,
            night_loss_weight: float = 0.25,
            exploration: float = 0.05,
            seed: int = 0,
    ):
        if not opponents:
            raise ValueError("PFSP requires at least one opponent")
        self.opponents = list(opponents)
        self.hard_win_rate = float(hard_win_rate)
        self.night_loss_weight = float(night_loss_weight)
        self.exploration = float(exploration)
        self.random = random.Random(seed)

    def weights(self) -> list[float]:
        raw = []
        for opponent in self.opponents:
            difficulty = max(self.hard_win_rate - opponent.learner_win_rate, 0.0)
            uncertainty = 1.0 / (opponent.games + 1.0) ** 0.5
            night_loss = max(opponent.night_loss_mean, 0.0)
            raw.append(self.exploration + difficulty + uncertainty + self.night_loss_weight * night_loss)
        total = sum(raw)
        return [value / total for value in raw]

    def sample(self) -> LeagueOpponent:
        return self.random.choices(self.opponents, weights=self.weights(), k=1)[0]

    def to_dict(self) -> dict:
        return {
            "hard_win_rate": self.hard_win_rate,
            "night_loss_weight": self.night_loss_weight,
            "exploration": self.exploration,
            "opponents": [asdict(opponent) for opponent in self.opponents],
            "weights": dict(zip((item.name for item in self.opponents), self.weights())),
        }

    @classmethod
    def from_json(cls, path: Path, seed: int = 0) -> "PFSPOpponentSampler":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            [LeagueOpponent(**item) for item in payload["opponents"]],
            hard_win_rate=payload.get("hard_win_rate", 0.45),
            night_loss_weight=payload.get("night_loss_weight", 0.25),
            exploration=payload.get("exploration", 0.05),
            seed=seed,
        )


def merge_opponents(items: Iterable[LeagueOpponent]) -> list[LeagueOpponent]:
    return list({item.name: item for item in items}.values())
