from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import CityTile, Unit
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from ..utility_constants import DN_CYCLE_LEN


class RuntimeGatePolicy:
    """Shared runtime gate used by Kaggle inference and training rollouts."""

    def __init__(self, flags: SimpleNamespace):
        self.flags = flags

    def active(self, game_state: Game) -> bool:
        if not getattr(self.flags, "runtime_gate_hard_diagnostic_mode", False):
            return False
        turn = game_state.turn
        map_size = game_state.map_width
        player = getattr(game_state, "id", 0)
        if map_size in (12, 16):
            return turn >= 120 and 25 <= turn % DN_CYCLE_LEN <= 29
        if map_size == 24 and player == 0 and turn < 80:
            return False
        return True

    @staticmethod
    def city_fuel_turns(city) -> float:
        if city is None:
            return float("inf")
        upkeep_getter = getattr(city, "get_light_upkeep", None)
        upkeep = upkeep_getter() if upkeep_getter is not None else city.light_upkeep
        return float(city.fuel) / max(float(upkeep), 1e-6)

    def city_fuel_buffer_summary(self, player) -> Dict[str, float]:
        fuel_turns = [self.city_fuel_turns(city) for city in player.cities.values()]
        if len(fuel_turns) == 0:
            return {
                "min": float("inf"),
                "p25": float("inf"),
                "total": float("inf"),
                "city_tiles": 0.0,
            }
        total_fuel = sum(float(city.fuel) for city in player.cities.values())
        total_upkeep = sum(
            float(getattr(city, "get_light_upkeep", lambda: city.light_upkeep)())
            for city in player.cities.values()
        )
        return {
            "min": min(fuel_turns),
            "p25": float(np.percentile(fuel_turns, 25)),
            "total": total_fuel / max(total_upkeep, 1e-6),
            "city_tiles": float(player.city_tile_count),
        }

    def large_map_gate_strict(self, game_state: Game, buffer: Dict[str, float]) -> bool:
        return (
            getattr(self.flags, "runtime_gate_v1_large_map_extra_strict", False)
            and game_state.map_width >= getattr(self.flags, "runtime_gate_v1_large_map_size", 32)
            and buffer["city_tiles"] >= getattr(self.flags, "runtime_gate_v1_large_map_min_city_tiles", 40)
        )

    def should_gate_build_worker(
            self,
            game_state: Game,
            player,
            city_tile: CityTile,
            buffer: Optional[Dict[str, float]] = None,
    ) -> bool:
        if not self.active(game_state):
            return False
        if buffer is None:
            buffer = self.city_fuel_buffer_summary(player)
        p25_limit = getattr(self.flags, "runtime_gate_v1_p25_city_fuel_turns_block_bw", 8.0)
        if self.large_map_gate_strict(game_state, buffer):
            p25_limit = max(
                p25_limit,
                getattr(self.flags, "runtime_gate_v1_large_map_p25_city_fuel_turns_block_bw", 12.0),
            )
        city = player.cities.get(city_tile.cityid)
        city_turns = self.city_fuel_turns(city)
        return (
            city_turns < getattr(self.flags, "runtime_gate_v1_min_city_fuel_turns_block_bw", 3.0)
            or buffer["p25"] < p25_limit
            or buffer["total"] < getattr(self.flags, "runtime_gate_v1_total_fuel_turns_block_bw", 5.0)
        )

    def should_gate_build_city(
            self,
            game_state: Game,
            player,
            unit: Unit,
            buffer: Optional[Dict[str, float]] = None,
    ) -> bool:
        if not self.active(game_state):
            return False
        if buffer is None:
            buffer = self.city_fuel_buffer_summary(player)
        p25_limit = getattr(self.flags, "runtime_gate_v1_p25_city_fuel_turns_block_bcity", 6.0)
        if self.large_map_gate_strict(game_state, buffer):
            p25_limit = max(
                p25_limit,
                getattr(self.flags, "runtime_gate_v1_large_map_p25_city_fuel_turns_block_bcity", 10.0),
            )
        cargo = float(unit.cargo.wood + unit.cargo.coal * 10 + unit.cargo.uranium * 40)
        return (
            cargo < GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]
            or buffer["min"] < getattr(self.flags, "runtime_gate_v1_min_city_fuel_turns_block_bcity", 2.5)
            or buffer["p25"] < p25_limit
            or buffer["total"] < getattr(self.flags, "runtime_gate_v1_total_fuel_turns_block_bcity", 4.0)
        )

    def mask_policy_logits_for_game(
            self,
            policy_logits: Dict[str, torch.Tensor],
            game_state: Game,
            env_index: int,
    ) -> bool:
        """Mask gated actions in batched model logits for one vector-env game.

        Returns True if at least one logit was masked.
        """
        if not self.active(game_state):
            return False
        changed = False
        build_worker_idx = ACTION_MEANINGS_TO_IDX["city_tile"].get("BUILD_WORKER")
        build_city_idx = ACTION_MEANINGS_TO_IDX["worker"].get("BUILD_CITY")
        for player in game_state.players:
            player_id = player.team
            buffer = self.city_fuel_buffer_summary(player)
            if build_worker_idx is not None and "city_tile" in policy_logits:
                for city in player.cities.values():
                    for city_tile in city.citytiles:
                        if city_tile.can_act() and self.should_gate_build_worker(game_state, player, city_tile, buffer):
                            policy_logits["city_tile"][
                                env_index, 0, player_id, city_tile.pos.x, city_tile.pos.y, build_worker_idx
                            ] = float("-inf")
                            changed = True
            if build_city_idx is not None and "worker" in policy_logits:
                for unit in player.units:
                    if (
                        unit.is_worker()
                        and unit.can_act()
                        and self.should_gate_build_city(game_state, player, unit, buffer)
                    ):
                        policy_logits["worker"][
                            env_index, 0, player_id, unit.pos.x, unit.pos.y, build_city_idx
                        ] = float("-inf")
                        changed = True
        return changed


def iter_lux_envs(env) -> Iterable:
    """Return underlying LuxEnv instances from the training/inference wrappers."""
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    if hasattr(current, "envs"):
        for wrapped in current.envs:
            yield wrapped.unwrapped
    elif hasattr(current, "game_state"):
        yield current


def apply_runtime_gate_to_actor_output(agent_output: Dict[str, Dict[str, torch.Tensor]], env, flags) -> Dict:
    """Apply the shared gate to monobeast actor logits before env.step()."""
    # Production intervention is performed by SidecarAgentWrapper as a soft
    # additive bias. Legacy hard masking is available only for diagnostics.
    if not getattr(flags, "runtime_gate_hard_diagnostic_mode", False):
        return agent_output
    if "policy_logits" not in agent_output:
        return agent_output

    policy_logits = {
        key: value.clone()
        for key, value in agent_output["policy_logits"].items()
    }
    gate = RuntimeGatePolicy(flags)
    changed = False
    for env_index, lux_env in enumerate(iter_lux_envs(env)):
        changed = gate.mask_policy_logits_for_game(policy_logits, lux_env.game_state, env_index) or changed
    if not changed:
        return agent_output

    updated = dict(agent_output)
    updated["policy_logits"] = policy_logits
    updated["actions"] = {
        key: value.argsort(dim=-1, descending=True)[..., :agent_output["actions"][key].shape[-1]]
        for key, value in policy_logits.items()
    }
    return updated
