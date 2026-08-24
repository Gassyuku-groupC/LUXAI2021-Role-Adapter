from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..lux.constants import Constants
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from .gate_policy import iter_lux_envs
from .role_assignment import (
    ATTACKER,
    BUILDER,
    FIREFIGHTER,
    FUEL_DEPOT,
    FUEL_STATION,
    HARVESTER,
    MANUFACTURING_POINT,
    RESEARCH_STATION,
    SACRIFICIAL_DECAY,
    RoleAssignmentConfig,
    RoleCityBiasParams,
    RoleState,
    assign_roles,
    city_tile_roles,
    direction_towards,
    nearest_city_center_by_role,
    nearest_critical_city_center,
    nearest_enemy_worker_position,
    nearest_fuel_position,
)


ROLE_BIAS_NAMES = RoleCityBiasParams.names()
ROLE_BIAS_INDEX = {name: index for index, name in enumerate(ROLE_BIAS_NAMES)}


class TrainableRoleBiasLayer(nn.Module):
    """Apply compact signed role codes to policy logits."""

    def __init__(self, initial: RoleCityBiasParams | None = None):
        super().__init__()
        initial = initial or RoleCityBiasParams()
        self.bias_params = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(getattr(initial, name))))
            for name in ROLE_BIAS_NAMES
        })

    def forward(
            self,
            policy_logits: Dict[str, torch.Tensor],
            role_codes: Dict[str, torch.Tensor],
            role_scale: torch.Tensor | None = None,
    ):
        output = {}
        for space, logits in policy_logits.items():
            codes = role_codes.get(space)
            if codes is None:
                output[space] = logits
                continue
            codes = codes.to(device=logits.device)
            delta = torch.zeros_like(logits)
            for index, name in enumerate(ROLE_BIAS_NAMES, start=1):
                value = self.bias_params[name].to(dtype=logits.dtype)
                delta = delta + (codes == index).to(logits.dtype) * value
                delta = delta - (codes == -index).to(logits.dtype) * value
            if role_scale is not None:
                delta = delta * role_scale.to(device=delta.device, dtype=delta.dtype)
            output[space] = logits + delta
        return output

    def values(self) -> Dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.bias_params.items()}


class RoleBiasCodeBuilder:
    """Build signed parameter-index tensors from live Lux game states."""

    def __init__(self, config: RoleAssignmentConfig):
        self.config = config
        self.states: Dict[tuple[int, int], RoleState] = {}

    def build(self, env, available_actions_mask: Dict[str, torch.Tensor]):
        codes = {
            space: torch.zeros_like(mask, dtype=torch.int8)
            for space, mask in available_actions_mask.items()
        }
        for env_index, lux_env in enumerate(iter_lux_envs(env)):
            game = lux_env.game_state
            for player in game.players:
                player_id = player.team
                key = (env_index, player_id)
                if game.turn == 0 or key not in self.states:
                    self.states[key] = RoleState()
                opponent = game.players[1 - player_id]
                snapshot = assign_roles(
                    game_state=game,
                    player=player,
                    opponent=opponent,
                    state=self.states[key],
                    config=self.config,
                )
                self._write_player(codes, env_index, game, player, opponent, snapshot)
        return codes

    def build_player_from_snapshot(
            self,
            game,
            player,
            opponent,
            snapshot,
            available_actions_mask: Dict[str, torch.Tensor],
    ):
        codes = {
            space: torch.zeros_like(mask, dtype=torch.int8)
            for space, mask in available_actions_mask.items()
        }
        self._write_player(codes, 0, game, player, opponent, snapshot)
        return codes

    @staticmethod
    def _code(name: str, sign: int = 1) -> int:
        return sign * (ROLE_BIAS_INDEX[name] + 1)

    def _set(self, codes, space, env_index, player, x, y, action, name, sign=1):
        index = ACTION_MEANINGS_TO_IDX[space].get(action)
        if index is not None:
            codes[space][env_index, 0, player, x, y, index] = self._code(name, sign)

    def _write_player(self, codes, env_index, game, player, opponent, snapshot):
        player_id = player.team
        for worker in player.units:
            if not worker.is_worker() or not worker.can_act():
                continue
            assignment = snapshot.unit_roles.get(worker.id)
            if assignment is None:
                continue
            x, y = worker.pos.astuple()
            if assignment.role == HARVESTER:
                if game.map.get_cell(x, y).has_resource():
                    self._set(codes, "worker", env_index, player_id, x, y, "NO-OP", "harvester_mine_bias")
                target = nearest_fuel_position(game, player, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._set(codes, "worker", env_index, player_id, x, y, f"MOVE_{direction}", "harvester_move_bias")
            elif assignment.role == BUILDER:
                if worker.can_build(game.map):
                    self._set(codes, "worker", env_index, player_id, x, y, "BUILD_CITY", "builder_build_city_bias")
                else:
                    target = nearest_city_center_by_role(player, snapshot, FUEL_STATION, worker.pos)
                    direction = direction_towards(worker.pos, target)
                    if direction is not None:
                        self._set(codes, "worker", env_index, player_id, x, y, f"MOVE_{direction}", "builder_move_to_fuel_station_bias")
            elif assignment.role == ATTACKER:
                target = nearest_enemy_worker_position(opponent, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._set(codes, "worker", env_index, player_id, x, y, f"MOVE_{direction}", "attacker_block_move_bias")
                if not self.config.preserve_build_city_logit:
                    self._set(codes, "worker", env_index, player_id, x, y, "BUILD_CITY", "attacker_build_city_penalty", -1)
            elif assignment.role == FIREFIGHTER:
                target = nearest_critical_city_center(player, snapshot, worker.pos)
                direction = direction_towards(worker.pos, target)
                if direction is not None:
                    self._set(codes, "worker", env_index, player_id, x, y, f"MOVE_{direction}", "firefighter_move_bias")
                self._write_transfers(codes, env_index, player, worker, target)
                if not self.config.preserve_build_city_logit:
                    self._set(codes, "worker", env_index, player_id, x, y, "BUILD_CITY", "firefighter_build_city_penalty", -1)

        for (x, y), role in city_tile_roles(player, snapshot).items():
            tile = game.map.get_cell(x, y).citytile
            if tile is None or not tile.can_act():
                continue
            if role == MANUFACTURING_POINT:
                self._set(codes, "city_tile", env_index, player_id, x, y, "BUILD_WORKER", "manufacturing_build_worker_bias")
            elif role == RESEARCH_STATION:
                self._set(codes, "city_tile", env_index, player_id, x, y, "RESEARCH", "research_station_research_bias")
            elif role == FUEL_DEPOT:
                self._set(codes, "city_tile", env_index, player_id, x, y, "RESEARCH", "depot_research_bias")
            elif role == FUEL_STATION:
                self._set(codes, "city_tile", env_index, player_id, x, y, "RESEARCH", "fuel_station_research_bias")
            elif role == SACRIFICIAL_DECAY:
                self._set(codes, "city_tile", env_index, player_id, x, y, "NO-OP", "sacrificial_noop_bias")

    def _write_transfers(self, codes, env_index, player, worker, target):
        if target is None:
            return
        directions = {
            Constants.DIRECTIONS.NORTH: (0, -1),
            Constants.DIRECTIONS.EAST: (1, 0),
            Constants.DIRECTIONS.SOUTH: (0, 1),
            Constants.DIRECTIONS.WEST: (-1, 0),
        }
        x, y = worker.pos.astuple()
        for resource in ("uranium", "coal", "wood"):
            if worker.cargo.get(resource) <= 0:
                continue
            for direction, (dx, dy) in directions.items():
                recipient = next((unit for unit in player.units if unit.pos.x == x + dx and unit.pos.y == y + dy), None)
                if recipient is None or recipient.pos.distance_to(target) > worker.pos.distance_to(target):
                    continue
                self._set(
                    codes, "worker", env_index, player.team, x, y,
                    f"TRANSFER_{resource}_{direction}", "firefighter_transfer_bias",
                )


def attach_role_bias_codes(env_output, env, builder: RoleBiasCodeBuilder):
    updated = dict(env_output)
    info = dict(env_output["info"])
    info["role_bias_codes"] = builder.build(env, info["available_actions_mask"])
    envs = list(iter_lux_envs(env))
    scales = [builder.config.bias_scale_for(item.game_state.map_width) for item in envs]
    info["role_bias_scale"] = torch.tensor(scales, dtype=torch.float32).view(-1, 1, 1, 1, 1, 1)
    updated["info"] = info
    return updated
