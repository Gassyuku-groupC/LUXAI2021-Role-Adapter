from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_map import Position
from ..lux.constants import Constants


HARVESTER = "Harvester"
BUILDER = "Builder"
ATTACKER = "Attacker"
FIREFIGHTER = "Firefighter"

FUEL_DEPOT = "FuelDepot"
FUEL_STATION = "FuelStation"
RESEARCH_STATION = "ResearchStation"
MANUFACTURING_POINT = "ManufacturingPoint"
SACRIFICIAL_DECAY = "SacrificialDecay"

UNIT_ROLE_NAMES = (HARVESTER, BUILDER, ATTACKER, FIREFIGHTER)
UNIT_ROLE_TO_CODE = {name: index for index, name in enumerate(UNIT_ROLE_NAMES)}


@dataclass(frozen=True)
class RoleCityBiasParams:
    harvester_move_bias: float = 0.35
    harvester_mine_bias: float = 0.25
    builder_build_city_bias: float = 1.00
    builder_move_to_fuel_station_bias: float = 0.45
    attacker_block_move_bias: float = 1.20
    attacker_build_city_penalty: float = 2.00
    firefighter_move_bias: float = 1.80
    firefighter_transfer_bias: float = 1.20
    firefighter_build_city_penalty: float = 2.50
    depot_research_bias: float = 0.50
    fuel_station_research_bias: float = 0.35
    research_station_research_bias: float = 0.90
    manufacturing_build_worker_bias: float = 1.10
    sacrificial_noop_bias: float = 0.75

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls, value: Optional[Mapping]) -> "RoleCityBiasParams":
        data = dict(value or {})
        known = {name: data[name] for name in cls.names() if name in data}
        return cls(**known)

    @classmethod
    def from_yaml(cls, path: Path) -> "RoleCityBiasParams":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "role_city_bias_params" in data:
            data = data["role_city_bias_params"] or {}
        return cls.from_mapping(data)

    def to_mapping(self) -> Dict[str, float]:
        return dict(asdict(self))

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"role_city_bias_params": self.to_mapping()},
                f,
                sort_keys=True,
            )


@dataclass(frozen=True)
class RoleAssignmentConfig:
    enabled: bool = False
    dry_run_logging: bool = True
    bias_enabled: bool = False
    learnable_biases: bool = False
    bias_params_path: Optional[str] = None
    cooldown_turns: int = 5
    firefighter_override_cooldown: bool = True
    critical_city_nights: float = 2.0
    abandon_city_nights: float = 0.25
    abandon_transport_share_threshold: float = 0.75
    abandon_min_turn: int = 30
    abandon_confirmation_turns: int = 3
    abandon_max_cities: int = 1
    abandon_max_fraction: float = 0.10
    abandon_min_city_count: int = 3
    abandon_min_fuel_distance: int = 6
    abandon_rescue_distance: int = 6
    builder_cargo_fill_ratio: float = 0.95
    attacker_enemy_worker_distance: int = 2
    max_log_units: int = 8
    max_log_cities: int = 8
    annotate_summary: bool = False
    bias_disabled_map_sizes: Sequence[int] = field(default_factory=tuple)
    bias_disabled_players_by_map: Mapping[int, Sequence[int]] = field(default_factory=dict)
    bias_scale_by_map_size: Mapping[int, float] = field(default_factory=dict)
    max_biased_workers_by_map_size: Mapping[int, int] = field(default_factory=dict)
    safety_only_map_sizes: Sequence[int] = field(default_factory=tuple)
    update_time_budget_seconds: float = 1.5
    preserve_build_city_logit: bool = False
    bias_params: RoleCityBiasParams = field(default_factory=RoleCityBiasParams)

    @classmethod
    def from_mapping(
            cls,
            value: Optional[Mapping],
            base_dir: Optional[Path] = None,
    ) -> "RoleAssignmentConfig":
        data = dict(value or {})
        bias_param_names = set(RoleCityBiasParams.names())
        flat_bias_values = {name: data.pop(name) for name in list(data) if name in bias_param_names}
        nested_bias_values = data.pop("bias_params", None) or data.pop("role_city_bias_params", None) or {}
        bias_path = data.get("bias_params_path")

        bias_data = {}
        bias_data.update(nested_bias_values)
        bias_data.update(flat_bias_values)
        bias_params = RoleCityBiasParams.from_mapping(bias_data)

        if bias_path:
            path = Path(str(bias_path))
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            file_params = RoleCityBiasParams.from_yaml(path)
            merged = file_params.to_mapping()
            merged.update(flat_bias_values)
            bias_params = RoleCityBiasParams.from_mapping(merged)

        known = {field.name for field in fields(cls)}
        config_data = {key: data[key] for key in data if key in known and key != "bias_params"}
        return cls(**config_data, bias_params=bias_params)

    def to_mapping(self) -> Dict:
        data = asdict(self)
        data["bias_params"] = self.bias_params.to_mapping()
        return data

    def bias_active_for(self, map_size: int, player_id: int) -> bool:
        if map_size in {int(size) for size in self.bias_disabled_map_sizes}:
            return False
        disabled_players = self.bias_disabled_players_by_map.get(map_size)
        if disabled_players is None:
            disabled_players = self.bias_disabled_players_by_map.get(str(map_size), ())
        return player_id not in {int(value) for value in disabled_players}

    def bias_scale_for(self, map_size: int) -> float:
        value = self.bias_scale_by_map_size.get(map_size)
        if value is None:
            value = self.bias_scale_by_map_size.get(str(map_size), 1.0)
        return max(float(value), 0.0)

    def max_biased_workers_for(self, map_size: int) -> int:
        value = self.max_biased_workers_by_map_size.get(map_size)
        if value is None:
            value = self.max_biased_workers_by_map_size.get(str(map_size), 10**9)
        return max(int(value), 0)

    def safety_only_for(self, map_size: int) -> bool:
        return map_size in {int(size) for size in self.safety_only_map_sizes}


@dataclass(frozen=True)
class UnitRoleAssignment:
    unit_id: str
    role: str
    desired_role: str
    changed: bool
    cooldown_remaining: int
    reason: str


@dataclass(frozen=True)
class CitySpecialization:
    city_id: str
    role: str
    nights_of_fuel: float
    fuel_distance: Optional[int]
    city_distance: Optional[float]
    abandon: bool
    reason: str


@dataclass
class RoleState:
    unit_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    role_codes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    cooldown_expiry: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    city_abandon_streaks: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleAssignmentSnapshot:
    unit_roles: Dict[str, UnitRoleAssignment]
    city_roles: Dict[str, CitySpecialization]
    critical_city_ids: Tuple[str, ...]
    abandoned_city_ids: Tuple[str, ...]

    def summary(self, max_units: int = 8, max_cities: int = 8) -> str:
        role_counts: Dict[str, int] = {}
        for assignment in self.unit_roles.values():
            role_counts[assignment.role] = role_counts.get(assignment.role, 0) + 1
        city_counts: Dict[str, int] = {}
        for specialization in self.city_roles.values():
            city_counts[specialization.role] = city_counts.get(specialization.role, 0) + 1
        units = ",".join(
            f"{a.unit_id}:{a.role[0]}({a.cooldown_remaining})"
            for a in list(self.unit_roles.values())[:max_units]
        )
        cities = ",".join(
            f"{c.city_id}:{c.role}"
            for c in list(self.city_roles.values())[:max_cities]
        )
        return (
            f"RoleAssignment units={role_counts} cities={city_counts} "
            f"critical={len(self.critical_city_ids)} abandoned={len(self.abandoned_city_ids)} "
            f"sample_units=[{units}] sample_cities=[{cities}]"
        )

    def critical_city_centers(self, player) -> List[Position]:
        return [
            _city_center(player.cities[city_id].citytiles)
            for city_id in self.critical_city_ids
            if city_id in player.cities and player.cities[city_id].citytiles
        ]


def assign_roles(
        *,
        game_state,
        player,
        opponent,
        state: RoleState,
        config: RoleAssignmentConfig,
        risk_blocked_positions: Optional[Iterable[Tuple[int, int]]] = None,
        fuel_positions: Optional[Sequence[Position]] = None,
) -> RoleAssignmentSnapshot:
    risk_blocked_positions = set(risk_blocked_positions or ())
    city_roles = classify_city_specializations(
        game_state,
        player,
        config,
        fuel_positions=fuel_positions,
        abandon_streaks=state.city_abandon_streaks,
    )
    abandoned = tuple(
        city_id for city_id, role in city_roles.items()
        if role.role == SACRIFICIAL_DECAY or role.abandon
    )
    critical = tuple(
        city_id for city_id, role in city_roles.items()
        if role.city_id not in abandoned and role.nights_of_fuel <= config.critical_city_nights
    )

    firefighter_units = _select_firefighter_units(player.units, player.cities, critical)
    enemy_worker_near_positions = _enemy_worker_near_positions(
        opponent.units, config.attacker_enemy_worker_distance
    )
    turn = int(game_state.turn)
    units = list(player.units)
    unit_ids = np.asarray([unit.id for unit in units], dtype=object)
    previous_index = {unit_id: index for index, unit_id in enumerate(state.unit_ids.tolist())}
    previous_codes = np.full(len(units), -1, dtype=np.int8)
    previous_expiry = np.full(len(units), -1, dtype=np.int16)
    for index, unit_id in enumerate(unit_ids):
        old_index = previous_index.get(unit_id)
        if old_index is not None:
            previous_codes[index] = state.role_codes[old_index]
            previous_expiry[index] = state.cooldown_expiry[old_index]

    in_cooldown = previous_expiry > turn
    firefighter_set = set(firefighter_units)
    override_mask = np.asarray([
        config.firefighter_override_cooldown and unit.id in firefighter_set
        for unit in units
    ], dtype=bool)
    desired_codes = previous_codes.copy()
    reasons = ["cooldown_reuse" for _ in units]
    for index, unit in enumerate(units):
        if previous_codes[index] >= 0 and in_cooldown[index] and not override_mask[index]:
            continue
        desired_role, reasons[index] = _desired_unit_role(
            game_state=game_state,
            unit=unit,
            player=player,
            opponent=opponent,
            firefighter_unit_ids=firefighter_set,
            enemy_worker_near_positions=enemy_worker_near_positions,
            risk_blocked_positions=risk_blocked_positions,
            config=config,
        )
        desired_codes[index] = UNIT_ROLE_TO_CODE[desired_role]

    new_unit_mask = previous_codes < 0
    role_changed_mask = desired_codes != previous_codes
    change_mask = new_unit_mask | (role_changed_mask & (~in_cooldown | override_mask))
    role_codes = np.where(change_mask, desired_codes, previous_codes).astype(np.int8, copy=False)
    cooldown_expiry = np.where(
        change_mask,
        turn + int(config.cooldown_turns),
        previous_expiry,
    ).astype(np.int16, copy=False)
    cooldown_remaining = np.maximum(cooldown_expiry.astype(np.int32) - turn, 0)

    state.unit_ids = unit_ids
    state.role_codes = role_codes
    state.cooldown_expiry = cooldown_expiry

    unit_roles: Dict[str, UnitRoleAssignment] = {}
    for index, unit in enumerate(units):
        role = UNIT_ROLE_NAMES[int(role_codes[index])]
        desired_role = UNIT_ROLE_NAMES[int(desired_codes[index])]
        unit_roles[unit.id] = UnitRoleAssignment(
            unit_id=unit.id,
            role=role,
            desired_role=desired_role,
            changed=bool(change_mask[index]),
            cooldown_remaining=int(cooldown_remaining[index]),
            reason=reasons[index],
        )

    return RoleAssignmentSnapshot(
        unit_roles=unit_roles,
        city_roles=city_roles,
        critical_city_ids=critical,
        abandoned_city_ids=abandoned,
    )


def nearest_fuel_position(game_state, player, pos: Position) -> Optional[Position]:
    fuel_positions = _fuel_resource_positions(game_state, player)
    if not fuel_positions:
        return None
    return min(fuel_positions, key=lambda fuel_pos: (pos.distance_to(fuel_pos), fuel_pos.x, fuel_pos.y))


def nearest_city_center_by_role(
        player,
        snapshot: RoleAssignmentSnapshot,
        role: str,
        pos: Position,
) -> Optional[Position]:
    candidates: List[Position] = []
    for city_id, spec in snapshot.city_roles.items():
        if spec.role != role or city_id not in player.cities or not player.cities[city_id].citytiles:
            continue
        candidates.append(_city_center(player.cities[city_id].citytiles))
    if not candidates:
        return None
    return min(candidates, key=lambda center: (pos.distance_to(center), center.x, center.y))


def nearest_enemy_worker_position(opponent, pos: Position) -> Optional[Position]:
    candidates = [unit.pos for unit in opponent.units if unit.is_worker()]
    if not candidates:
        return None
    return min(candidates, key=lambda enemy_pos: (pos.distance_to(enemy_pos), enemy_pos.x, enemy_pos.y))


def nearest_critical_city_center(player, snapshot: RoleAssignmentSnapshot, pos: Position) -> Optional[Position]:
    candidates = snapshot.critical_city_centers(player)
    if not candidates:
        return None
    return min(candidates, key=lambda center: (pos.distance_to(center), center.x, center.y))


def direction_towards(source: Position, target: Optional[Position]) -> Optional[str]:
    if target is None:
        return None
    direction = source.direction_to(target)
    if direction == Constants.DIRECTIONS.CENTER:
        return None
    return direction


def city_tile_roles(player, snapshot: RoleAssignmentSnapshot) -> Dict[Tuple[int, int], str]:
    roles: Dict[Tuple[int, int], str] = {}
    for city_id, spec in snapshot.city_roles.items():
        city = player.cities.get(city_id)
        if city is None:
            continue
        for city_tile in city.citytiles:
            roles[city_tile.pos.astuple()] = spec.role
    return roles


def classify_city_specializations(
        game_state,
        player,
        config: RoleAssignmentConfig,
        fuel_positions: Optional[Sequence[Position]] = None,
        abandon_streaks: Optional[MutableMapping[str, int]] = None,
) -> Dict[str, CitySpecialization]:
    if fuel_positions is None:
        fuel_positions = _fuel_resource_positions(game_state, player)
    city_centers = {
        city_id: _city_center(city.citytiles)
        for city_id, city in player.cities.items()
        if city.citytiles
    }
    city_ids_with_centers = list(city_centers)
    center_array = np.asarray(
        [city_centers[city_id].astuple() for city_id in city_ids_with_centers],
        dtype=np.int16,
    )
    fuel_array = np.asarray([position.astuple() for position in fuel_positions], dtype=np.int16)
    fuel_distance_by_city = {}
    city_distance_by_city = {}
    if len(center_array):
        if len(fuel_array):
            fuel_distances = np.abs(center_array[:, None, :] - fuel_array[None, :, :]).sum(axis=2)
            nearest_fuel = fuel_distances.min(axis=1)
            fuel_distance_by_city = {
                city_id: int(nearest_fuel[index])
                for index, city_id in enumerate(city_ids_with_centers)
            }
        city_distances = np.abs(center_array[:, None, :] - center_array[None, :, :]).sum(axis=2)
        mean_city_distances = (
            city_distances.sum(axis=1) / max(len(city_ids_with_centers) - 1, 1)
        )
        city_distance_by_city = {
            city_id: float(mean_city_distances[index])
            for index, city_id in enumerate(city_ids_with_centers)
        }
    raw: Dict[str, CitySpecialization] = {}
    candidates = []
    for city_id, city in player.cities.items():
        center = city_centers.get(city_id)
        fuel_distance = fuel_distance_by_city.get(city_id)
        city_distance = city_distance_by_city.get(city_id)
        nights = _city_nights_of_fuel(city)
        candidate = _should_abandon_city(
            game_state=game_state,
            city=city,
            center=center,
            player=player,
            fuel_distance=fuel_distance,
            nights=nights,
            config=config,
        )
        if abandon_streaks is not None:
            abandon_streaks[city_id] = abandon_streaks.get(city_id, 0) + 1 if candidate else 0
            confirmed = abandon_streaks[city_id] >= max(config.abandon_confirmation_turns, 1)
        else:
            confirmed = candidate and config.abandon_confirmation_turns <= 1
        if confirmed:
            candidates.append(city_id)
        raw[city_id] = CitySpecialization(
            city_id=city_id,
            role=RESEARCH_STATION,
            nights_of_fuel=nights,
            fuel_distance=fuel_distance,
            city_distance=city_distance,
            abandon=False,
            reason="abandon_candidate_confirming" if candidate else "default_research",
        )

    if abandon_streaks is not None:
        active_ids = set(player.cities)
        for stale_id in list(abandon_streaks):
            if stale_id not in active_ids:
                del abandon_streaks[stale_id]

    city_count = len(raw)
    allowed_by_fraction = max(int(np.ceil(city_count * max(config.abandon_max_fraction, 0.0))), 0)
    abandon_limit = min(
        max(config.abandon_max_cities, 0),
        allowed_by_fraction,
        max(city_count - 1, 0),
    )
    if city_count < config.abandon_min_city_count:
        abandon_limit = 0
    selected = sorted(
        candidates,
        key=lambda city_id: (
            raw[city_id].nights_of_fuel,
            -(raw[city_id].fuel_distance or 10**6),
            city_id,
        ),
    )[:abandon_limit]
    for city_id in selected:
        raw[city_id] = _replace_city_role(
            raw[city_id],
            SACRIFICIAL_DECAY,
            "abandon_confirmed_strict_budget",
            abandon=True,
        )

    active_city_ids = [city_id for city_id, spec in raw.items() if not spec.abandon]
    if not active_city_ids:
        return raw

    fuel_station_id = min(
        active_city_ids,
        key=lambda cid: (raw[cid].fuel_distance if raw[cid].fuel_distance is not None else 10**6, cid),
    )
    depot_scores = _normalized_depot_scores(raw, active_city_ids)
    depot_id = min(active_city_ids, key=lambda cid: (depot_scores[cid], cid))
    manufacturing_id = _select_manufacturing_city_id(player, raw, active_city_ids)

    return {
        city_id: _replace_city_role(
            spec,
            _city_role_for_id(city_id, depot_id, fuel_station_id, manufacturing_id),
            _city_reason_for_id(city_id, depot_id, fuel_station_id, manufacturing_id, spec),
        )
        for city_id, spec in raw.items()
    }


def _replace_city_role(
        spec: CitySpecialization,
        role: str,
        reason: str,
        abandon: Optional[bool] = None,
) -> CitySpecialization:
    if spec.abandon and abandon is None:
        return spec
    return CitySpecialization(
        city_id=spec.city_id,
        role=role,
        nights_of_fuel=spec.nights_of_fuel,
        fuel_distance=spec.fuel_distance,
        city_distance=spec.city_distance,
        abandon=spec.abandon if abandon is None else abandon,
        reason=reason,
    )


def _city_role_for_id(city_id: str, depot_id: str, fuel_station_id: str, manufacturing_id: Optional[str]) -> str:
    if city_id == manufacturing_id:
        return MANUFACTURING_POINT
    if city_id == depot_id:
        return FUEL_DEPOT
    if city_id == fuel_station_id:
        return FUEL_STATION
    return RESEARCH_STATION


def _city_reason_for_id(
        city_id: str,
        depot_id: str,
        fuel_station_id: str,
        manufacturing_id: Optional[str],
        spec: CitySpecialization,
) -> str:
    if city_id == manufacturing_id:
        return "unit_cap_available_safe_fuel"
    if city_id == depot_id:
        return "combined_city_and_fuel_centrality"
    if city_id == fuel_station_id:
        return "nearest_fuel_access"
    return spec.reason


def _desired_unit_role(
        *,
        game_state,
        unit,
        player,
        opponent,
        firefighter_unit_ids: Sequence[str],
        enemy_worker_near_positions: Iterable[Tuple[int, int]],
        risk_blocked_positions: Iterable[Tuple[int, int]],
        config: RoleAssignmentConfig,
) -> Tuple[str, str]:
    if unit.id in firefighter_unit_ids:
        return FIREFIGHTER, "nearest_fuel_carrier_to_non_abandoned_critical_city"
    cargo_total = _cargo_total(unit)
    capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] if unit.is_worker() else \
        GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
    cargo_ratio = cargo_total / max(capacity, 1)
    if (
            unit.is_worker() and
            cargo_ratio >= config.builder_cargo_fill_ratio and
            unit.pos.astuple() not in set(risk_blocked_positions) and
            unit.can_build(game_state.map)
    ):
        return BUILDER, "full_worker_on_legal_non_risk_build_tile"
    if unit.is_worker() and unit.pos.astuple() in enemy_worker_near_positions:
        if player.city_tile_count >= opponent.city_tile_count or len(player.units) >= len(opponent.units):
            return ATTACKER, "enemy_worker_nearby_with_even_or_better_position"
    return HARVESTER, "default_resource_flow"


def _select_firefighter_units(units, cities, critical_city_ids: Sequence[str]) -> Tuple[str, ...]:
    critical_targets = [
        _city_center(cities[city_id].citytiles)
        for city_id in critical_city_ids
        if city_id in cities and cities[city_id].citytiles
    ]
    if not critical_targets:
        return ()
    candidates = [unit for unit in units if _cargo_total(unit) > 0]
    if not candidates:
        return ()
    unit_positions = np.asarray([unit.pos.astuple() for unit in candidates], dtype=np.int16)
    target_positions = np.asarray([target.astuple() for target in critical_targets], dtype=np.int16)
    distances = np.abs(unit_positions[:, None, :] - target_positions[None, :, :]).sum(axis=2)
    nearest = distances.min(axis=1)
    order = sorted(range(len(candidates)), key=lambda index: (int(nearest[index]), candidates[index].id))
    return tuple(candidates[index].id for index in order[:2])


def _select_manufacturing_city_id(player, specs: Mapping[str, CitySpecialization], active_city_ids: Sequence[str]):
    units_to_build = max(player.city_tile_count - len(player.units), 0)
    if units_to_build <= 0:
        return None
    candidates = [
        city_id for city_id in active_city_ids
        if specs[city_id].nights_of_fuel > 2.0
    ]
    if not candidates:
        candidates = list(active_city_ids)
    return max(
        candidates,
        key=lambda cid: (
            specs[cid].nights_of_fuel,
            -(specs[cid].fuel_distance if specs[cid].fuel_distance is not None else 10**6),
            cid,
        ),
    )


def _normalized_depot_scores(
        specs: Mapping[str, CitySpecialization],
        active_city_ids: Sequence[str],
) -> Dict[str, float]:
    fuel_values = [
        specs[city_id].fuel_distance
        for city_id in active_city_ids
        if specs[city_id].fuel_distance is not None
    ]
    city_values = [
        specs[city_id].city_distance
        for city_id in active_city_ids
        if specs[city_id].city_distance is not None
    ]
    fuel_max = max(fuel_values, default=0)
    fuel_min = min(fuel_values, default=0)
    city_max = max(city_values, default=0)
    city_min = min(city_values, default=0)
    return {
        city_id: (
            _normalize_distance(specs[city_id].fuel_distance, fuel_min, fuel_max) +
            _normalize_distance(specs[city_id].city_distance, city_min, city_max)
        )
        for city_id in active_city_ids
    }


def _normalize_distance(value: Optional[float], min_value: float, max_value: float) -> float:
    if value is None:
        return 1.0
    spread = max_value - min_value
    if spread <= 0:
        return 0.0
    return (value - min_value) / spread


def _should_abandon_city(
        *, game_state, city, center, player, fuel_distance, nights: float,
        config: RoleAssignmentConfig,
) -> bool:
    turn = int(game_state.turn)
    if turn < config.abandon_min_turn or turn % 40 < 30:
        return False
    if nights > config.abandon_city_nights:
        return False
    fuel_poor = fuel_distance is None or fuel_distance > max(
        config.abandon_min_fuel_distance,
        len(city.citytiles) + 3,
    )
    if not fuel_poor:
        return False
    cargo_carriers = [unit for unit in player.units if _cargo_total(unit) > 0]
    if not cargo_carriers:
        return True
    if center is not None and any(
            unit.pos.distance_to(center) <= config.abandon_rescue_distance
            for unit in cargo_carriers
    ):
        return False
    needed = min(len(cargo_carriers), max(len(city.citytiles), 1))
    transport_share = needed / max(len(player.units), 1)
    return transport_share > config.abandon_transport_share_threshold and fuel_poor


def _fuel_resource_positions(game_state, player) -> List[Position]:
    positions: List[Position] = []
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


def _city_center(citytiles) -> Position:
    x = round(sum(city_tile.pos.x for city_tile in citytiles) / len(citytiles))
    y = round(sum(city_tile.pos.y for city_tile in citytiles) / len(citytiles))
    return Position(x, y)


def _nearest_distance(pos: Optional[Position], targets: Sequence[Position]) -> Optional[int]:
    if pos is None or not targets:
        return None
    return min(pos.distance_to(target) for target in targets)


def _mean_distance_to_other_cities(
        city_id: str,
        center: Optional[Position],
        city_centers: Mapping[str, Position],
) -> Optional[float]:
    if center is None:
        return None
    others = [pos for other_id, pos in city_centers.items() if other_id != city_id]
    if not others:
        return 0.0
    return sum(center.distance_to(pos) for pos in others) / len(others)


def _city_nights_of_fuel(city) -> float:
    if city.light_upkeep <= 0:
        return float("inf")
    return city.fuel / city.light_upkeep


def _cargo_total(unit) -> int:
    return int(unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium)


def _enemy_worker_near_positions(opponent_units, max_distance: int):
    positions = set()
    for unit in opponent_units:
        if not unit.is_worker():
            continue
        x, y = unit.pos.astuple()
        for dx in range(-max_distance, max_distance + 1):
            remaining = max_distance - abs(dx)
            for dy in range(-remaining, remaining + 1):
                positions.add((x + dx, y + dy))
    return positions
