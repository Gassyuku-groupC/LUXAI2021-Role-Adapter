from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
from torch import nn

from ..lux.constants import Constants
from ..lux.game_objects import Unit
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from ..utility_constants import MAX_BOARD_SIZE
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
    RoleAssignmentSnapshot,
    RoleState,
    assign_roles,
    city_tile_roles,
)


ROLE_BIAS_PRIORITY = {
    FIREFIGHTER: 0,
    BUILDER: 1,
    HARVESTER: 2,
    ATTACKER: 3,
}


def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return pos[0] * board_dims[1] + pos[1]


class RoleCityAdapter(nn.Module):
    """Plug-in role/city adapter that can stay heuristic or learn bias strengths.

    The adapter owns role cooldown state and applies legal-mask-respecting logit
    biases after the frozen Actor produces base logits. When learnable_biases is
    enabled, the scalar bias strengths are nn.Parameters and can be optimized by
    a KL-APPO loop without changing the Actor checkpoint structure.
    """

    def __init__(self, config: RoleAssignmentConfig, learnable_biases: bool = False):
        super().__init__()
        self.config = config
        self.state = RoleState()
        self.snapshot: Optional[RoleAssignmentSnapshot] = None
        self._fuel_positions = []
        self._enemy_worker_positions = []
        self._city_centers_by_role = {}
        self._critical_city_centers = []
        self._runtime_bias_scale = 1.0
        self._unit_grid = np.empty((0, 0), dtype=np.int32)
        self._units = []
        self._directions = {}
        self._targets = {}
        self._skip_next_role_update = False
        self.last_update_degraded = False
        self.last_update_seconds = 0.0
        self._worker_updates = []
        self._city_updates = []
        self.learnable_biases = learnable_biases
        if learnable_biases:
            self.bias_params = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(float(value), dtype=torch.float32))
                for name, value in config.bias_params.to_mapping().items()
            })
        else:
            self.bias_params = None

    @classmethod
    def from_config(cls, config: RoleAssignmentConfig) -> "RoleCityAdapter":
        return cls(config=config, learnable_biases=config.learnable_biases)

    def bias_value(self, name: str, like: torch.Tensor) -> torch.Tensor:
        if self.bias_params is not None:
            return self.bias_params[name].to(device=like.device, dtype=like.dtype)
        return torch.tensor(
            float(getattr(self.config.bias_params, name)),
            device=like.device,
            dtype=like.dtype,
        )

    def update(
            self,
            *,
            game_state,
            player,
            opponent,
            risk_blocked_positions=(),
    ) -> Optional[RoleAssignmentSnapshot]:
        if not self.config.enabled:
            self.snapshot = None
            return None
        started = time.perf_counter()
        self._fuel_positions = self._collect_fuel_positions(game_state, player)
        self.last_update_degraded = bool(self._skip_next_role_update and self.snapshot is not None)
        if self.last_update_degraded:
            self._skip_next_role_update = False
        else:
            self.snapshot = assign_roles(
                game_state=game_state,
                player=player,
                opponent=opponent,
                state=self.state,
                config=self.config,
                risk_blocked_positions=risk_blocked_positions,
                fuel_positions=self._fuel_positions,
            )
        self._refresh_spatial_context(game_state, player, opponent)
        self.last_update_seconds = time.perf_counter() - started
        if self.last_update_seconds > float(self.config.update_time_budget_seconds):
            self._skip_next_role_update = True
        return self.snapshot

    def _refresh_spatial_context(self, game_state, player, opponent) -> None:
        self._units = list(player.units)
        self._unit_grid = np.full(
            (game_state.map_width, game_state.map_height), -1, dtype=np.int32
        )
        for index, unit in enumerate(self._units):
            self._unit_grid[unit.pos.x, unit.pos.y] = index
        self._enemy_worker_positions = [
            unit.pos for unit in opponent.units if unit.is_worker()
        ]
        self._city_centers_by_role = self._collect_city_centers_by_role(player)
        self._critical_city_centers = self.snapshot.critical_city_centers(player)
        self._directions = {}
        self._targets = {}
        self._cache_nearest("fuel", self._fuel_positions)
        self._cache_nearest("enemy", self._enemy_worker_positions)
        self._cache_nearest("critical", self._critical_city_centers)
        self._cache_nearest("fuel_station", self._city_centers_by_role.get(FUEL_STATION, ()))

    def deactivate(self) -> None:
        self.snapshot = None

    def apply(
            self,
            *,
            game_state,
            player,
            opponent,
            actionable_workers: Dict[int, List[Unit]],
            actionable_city_tiles: Dict[int, object],
            policy_logits: Dict[str, torch.Tensor],
            available_actions_mask: Dict[str, torch.Tensor],
            player_id: int,
    ) -> Dict[str, torch.Tensor]:
        if not self.config.enabled or not self.config.bias_enabled or self.snapshot is None:
            return policy_logits

        out = dict(policy_logits)
        runtime_fast_path = self.bias_params is None
        if runtime_fast_path:
            worker_logits = policy_logits["worker"]
            city_logits = policy_logits["city_tile"]
            self._worker_updates = []
            self._city_updates = []
        else:
            worker_logits = policy_logits["worker"].clone()
            city_logits = policy_logits["city_tile"].clone()
        worker_mask = available_actions_mask["worker"]
        city_mask = available_actions_mask["city_tile"]

        self._runtime_bias_scale = self.config.bias_scale_for(game_state.map_width)
        actionable = []
        for workers in actionable_workers.values():
            if not workers:
                continue
            worker = workers[0]
            assignment = self.snapshot.unit_roles.get(worker.id)
            if assignment is not None:
                actionable.append((ROLE_BIAS_PRIORITY.get(assignment.role, 9), worker.id, worker, assignment))
        actionable.sort(key=lambda item: (item[0], item[1]))
        worker_budget = self.config.max_biased_workers_for(game_state.map_width)
        safety_only = self.config.safety_only_for(game_state.map_width)

        for _, _, worker, assignment in actionable[:worker_budget]:
            if safety_only and assignment.role != FIREFIGHTER:
                continue
            x, y = worker.pos.astuple()
            if assignment.role == HARVESTER:
                cell = game_state.map.get_cell(x, y)
                if cell.has_resource():
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "NO-OP",
                                             "harvester_mine_bias")
                direction = self._directions.get(("fuel", worker.id))
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "harvester_move_bias")
            elif assignment.role == BUILDER:
                if worker.can_build(game_state.map):
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                             "builder_build_city_bias")
                else:
                    direction = self._directions.get(("fuel_station", worker.id))
                    if direction is not None:
                        self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                                 "builder_move_to_fuel_station_bias")
            elif assignment.role == ATTACKER:
                direction = self._directions.get(("enemy", worker.id))
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "attacker_block_move_bias")
                if not self.config.preserve_build_city_logit:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                             "attacker_build_city_penalty", sign=-1.0)
            elif assignment.role == FIREFIGHTER:
                target = self._targets.get(("critical", worker.id))
                direction = self._directions.get(("critical", worker.id))
                if direction is not None:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, f"MOVE_{direction}",
                                             "firefighter_move_bias")
                self._bias_firefighter_transfers(worker_logits, worker_mask, player_id, worker, target)
                if not self.config.preserve_build_city_logit:
                    self._bias_worker_action(worker_logits, worker_mask, player_id, x, y, "BUILD_CITY",
                                             "firefighter_build_city_penalty", sign=-1.0)

        research_complete = int(player.research_points) >= 200
        for pos, city_role in (() if safety_only else city_tile_roles(player, self.snapshot).items()):
            x, y = pos
            if pos_to_loc(pos) not in actionable_city_tiles:
                continue
            if city_role == MANUFACTURING_POINT:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "BUILD_WORKER",
                                       "manufacturing_build_worker_bias")
            elif city_role == RESEARCH_STATION and not research_complete:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH",
                                       "research_station_research_bias")
            elif city_role == FUEL_DEPOT and not research_complete:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH", "depot_research_bias")
            elif city_role == FUEL_STATION and not research_complete:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "RESEARCH",
                                       "fuel_station_research_bias")
            elif city_role == SACRIFICIAL_DECAY:
                self._bias_city_action(city_logits, city_mask, player_id, x, y, "NO-OP", "sacrificial_noop_bias")

        if runtime_fast_path:
            worker_candidate = policy_logits["worker"].clone()
            city_candidate = policy_logits["city_tile"].clone()
            self._apply_batched_updates(worker_candidate, self._worker_updates)
            self._apply_batched_updates(city_candidate, self._city_updates)
            out["worker"] = torch.where(
                worker_mask, worker_candidate, policy_logits["worker"]
            )
            out["city_tile"] = torch.where(
                city_mask, city_candidate, policy_logits["city_tile"]
            )
        else:
            out["worker"] = worker_logits
            out["city_tile"] = city_logits
        return out

    def _bias_worker_action(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            x: int,
            y: int,
            action: str,
            param_name: str,
            sign: float = 1.0,
    ) -> None:
        idx = ACTION_MEANINGS_TO_IDX["worker"][action]
        if self.bias_params is None:
            self._worker_updates.append((
                player_id, x, y, idx,
                float(getattr(self.config.bias_params, param_name)) * sign
                * self._runtime_bias_scale,
            ))
        elif mask[0, 0, player_id, x, y, idx]:
            logits[0, 0, player_id, x, y, idx] = (
                logits[0, 0, player_id, x, y, idx] +
                self.bias_value(param_name, logits) * sign
                * self._runtime_bias_scale
            )

    def _bias_city_action(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            x: int,
            y: int,
            action: str,
            param_name: str,
    ) -> None:
        idx = ACTION_MEANINGS_TO_IDX["city_tile"][action]
        if self.bias_params is None:
            self._city_updates.append((
                player_id, x, y, idx,
                float(getattr(self.config.bias_params, param_name))
                * self._runtime_bias_scale,
            ))
        elif mask[0, 0, player_id, x, y, idx]:
            logits[0, 0, player_id, x, y, idx] = (
                logits[0, 0, player_id, x, y, idx] +
                self.bias_value(param_name, logits)
                * self._runtime_bias_scale
            )

    def _bias_firefighter_transfers(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor,
            player_id: int,
            worker: Unit,
            target,
    ) -> None:
        if target is None:
            return
        x, y = worker.pos.astuple()
        directions = {
            Constants.DIRECTIONS.NORTH: (0, -1),
            Constants.DIRECTIONS.EAST: (1, 0),
            Constants.DIRECTIONS.SOUTH: (0, 1),
            Constants.DIRECTIONS.WEST: (-1, 0),
        }
        for resource in ("uranium", "coal", "wood"):
            if worker.cargo.get(resource) <= 0:
                continue
            for direction, (dx, dy) in directions.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self._unit_grid.shape[0] and 0 <= ny < self._unit_grid.shape[1]):
                    continue
                recipient_index = int(self._unit_grid[nx, ny])
                if recipient_index < 0:
                    continue
                recipient = self._units[recipient_index]
                # Lux 2021 transfer is unit-to-adjacent-unit only. This only biases relay transfers
                # toward a unit that is at least as close to the critical city as the carrier.
                if recipient.pos.distance_to(target) > worker.pos.distance_to(target):
                    continue
                action = f"TRANSFER_{resource}_{direction}"
                idx = ACTION_MEANINGS_TO_IDX["worker"][action]
                if self.bias_params is None:
                    self._worker_updates.append((
                        player_id, x, y, idx,
                        float(self.config.bias_params.firefighter_transfer_bias)
                        * self._runtime_bias_scale,
                    ))
                elif mask[0, 0, player_id, x, y, idx]:
                    logits[0, 0, player_id, x, y, idx] = (
                        logits[0, 0, player_id, x, y, idx] +
                        self.bias_value("firefighter_transfer_bias", logits)
                        * self._runtime_bias_scale
                    )

    def _cache_nearest(self, name: str, positions) -> None:
        if not self._units or not positions:
            return
        targets = sorted(positions, key=lambda pos: (pos.x, pos.y))
        unit_xy = np.asarray(
            [(unit.pos.x, unit.pos.y) for unit in self._units], dtype=np.int16
        )
        target_xy = np.asarray([(pos.x, pos.y) for pos in targets], dtype=np.int16)
        delta = target_xy[None, :, :] - unit_xy[:, None, :]
        distance = np.abs(delta).sum(axis=2)
        closest = np.argmin(distance, axis=1)
        best_delta = delta[np.arange(len(self._units)), closest]

        # Match Position.direction_to tie-breaking exactly: north, east, south, west.
        directions = np.full(len(self._units), "", dtype=object)
        directions[best_delta[:, 1] < 0] = Constants.DIRECTIONS.NORTH
        unresolved = directions == ""
        directions[unresolved & (best_delta[:, 0] > 0)] = Constants.DIRECTIONS.EAST
        unresolved = directions == ""
        directions[unresolved & (best_delta[:, 1] > 0)] = Constants.DIRECTIONS.SOUTH
        unresolved = directions == ""
        directions[unresolved & (best_delta[:, 0] < 0)] = Constants.DIRECTIONS.WEST

        for index, unit in enumerate(self._units):
            if directions[index]:
                self._directions[(name, unit.id)] = directions[index]
            self._targets[(name, unit.id)] = targets[int(closest[index])]

    @staticmethod
    def _apply_batched_updates(logits: torch.Tensor, updates) -> None:
        if not updates:
            return
        device = logits.device
        players, xs, ys, actions, values = zip(*updates)
        count = len(updates)
        zeros = torch.zeros(count, dtype=torch.long, device=device)
        logits.index_put_(
            (
                zeros,
                zeros,
                torch.as_tensor(players, dtype=torch.long, device=device),
                torch.as_tensor(xs, dtype=torch.long, device=device),
                torch.as_tensor(ys, dtype=torch.long, device=device),
                torch.as_tensor(actions, dtype=torch.long, device=device),
            ),
            torch.as_tensor(values, dtype=logits.dtype, device=device),
            accumulate=True,
        )

    @staticmethod
    def _collect_fuel_positions(game_state, player):
        positions = []
        for y in range(game_state.map_height):
            for x in range(game_state.map_width):
                cell = game_state.map.get_cell(x, y)
                if not cell.has_resource():
                    continue
                if cell.resource.type == "wood":
                    positions.append(cell.pos)
                elif cell.resource.type == "coal" and player.researched_coal():
                    positions.append(cell.pos)
                elif cell.resource.type == "uranium" and player.researched_uranium():
                    positions.append(cell.pos)
        return positions

    def _collect_city_centers_by_role(self, player):
        centers = {}
        for city_id, spec in self.snapshot.city_roles.items():
            city = player.cities.get(city_id)
            if city is None or not city.citytiles:
                continue
            x = round(sum(tile.pos.x for tile in city.citytiles) / len(city.citytiles))
            y = round(sum(tile.pos.y for tile in city.citytiles) / len(city.citytiles))
            centers.setdefault(spec.role, []).append(type(city.citytiles[0].pos)(x, y))
        return centers
