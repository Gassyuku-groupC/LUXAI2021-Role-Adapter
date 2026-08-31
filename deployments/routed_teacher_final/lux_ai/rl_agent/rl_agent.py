import numpy as np
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import *
import yaml

from . import data_augmentation
from .gate_policy import RuntimeGatePolicy
from .role_assignment import RoleAssignmentConfig
from .role_city_adapter import RoleCityAdapter
from .trainable_role_bias import RoleBiasCodeBuilder
from ..lux_gym import create_reward_space, LuxEnv, wrappers
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..utils import DEBUG_MESSAGE, RUNTIME_DEBUG_MESSAGE, LOCAL_EVAL
from ..utility_constants import MAX_RESEARCH, DN_CYCLE_LEN, MAX_BOARD_SIZE
from ..nns import create_model, models
from ..utils import flags_to_namespace, Stopwatch

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import CityTile, Unit
from ..lux import annotate

MODEL_CONFIG_PATH = Path(__file__).parent / "config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "candidate_weights.pt"
AGENT = None
STRATEGY_SCORER_DIR = Path(__file__).parent / "strategy_scorers"
STRATEGY_SCORER_FEATURES = [
    "map_size", "turn", "turns_remaining", "night_cycle", "cycle_turn", "pre_night", "is_night",
    "turns_to_night", "team", "eval_side_numeric", "city_tiles", "cities", "largest_city_size",
    "mean_city_size", "resource_near_cities", "isolated_cities_r3", "units", "workers", "carts",
    "unit_cap_margin", "worker_citytile_ratio", "research", "fuel", "upkeep", "fuel_turns_total",
    "min_city_fuel_turns", "p25_city_fuel_turns", "median_city_fuel_turns", "mean_city_fuel_turns",
    "low_fuel_city_lt3", "low_fuel_city_lt5", "low_fuel_city_lt10", "unit_cargo_fuel",
    "wood_remaining", "coal_remaining", "uranium_remaining", "action_count", "move_actions",
    "transfer_actions", "pillage_actions", "research_actions", "bw_actions", "bc_actions",
    "bcity_actions", "bcity_isolated_actions", "bcity_adjacent_actions", "bcity_resource_near_actions",
    "bcity_adjacent_low_fuel_lt5_actions", "bw_low_fuel_lt3_actions", "bw_low_fuel_lt5_actions",
    "bw_low_fuel_lt10_actions", "city_tiles_delta_10", "city_tiles_growth_10", "workers_delta_10",
    "workers_growth_10", "upkeep_delta_10", "upkeep_growth_10", "fuel_delta_10",
    "fuel_turns_total_delta_10", "fuel_turns_total_drop_10", "p25_city_fuel_turns_delta_10",
    "p25_city_fuel_turns_drop_10", "min_city_fuel_turns_delta_10", "research_delta_10",
    "research_growth_10", "city_tiles_delta_next", "units_delta_next", "research_delta_next",
]

def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return pos[0] * board_dims[1] + pos[1]


def resolve_model_artifacts(agent_config: Mapping[str, Any], conf: Any) -> Tuple[Path, Path]:
    map_size = conf.get("width") if isinstance(conf, Mapping) else getattr(conf, "width", None)
    routes = agent_config.get("model_route_by_map_size", {}) or {}
    route = routes.get(str(map_size), routes.get(map_size, {})) or {}
    base_dir = RL_AGENT_CONFIG_PATH.parent.resolve()

    def package_path(value: str, default: Path) -> Path:
        path = (base_dir / value).resolve() if value else default.resolve()
        if path.parent != base_dir:
            raise ValueError(f"Model route must remain inside agent package: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing routed model artifact: {path}")
        return path

    return (
        package_path(str(route.get("model_config", "")), MODEL_CONFIG_PATH),
        package_path(str(route.get("checkpoint", "")), DEFAULT_CHECKPOINT_PATH),
    )


class RLAgent:
    def __init__(self, obs, conf):
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            agent_config = yaml.safe_load(f) or {}
        model_config_path, checkpoint_path = resolve_model_artifacts(agent_config, conf)
        with open(model_config_path, 'r') as f:
            self.model_flags = flags_to_namespace(yaml.safe_load(f))
            self.agent_flags = SimpleNamespace(**agent_config)
        self.role_assignment_config = RoleAssignmentConfig.from_mapping(
            agent_config.get("role_assignment"),
            base_dir=RL_AGENT_CONFIG_PATH.parent,
        )
        self.role_city_adapter = RoleCityAdapter.from_config(self.role_assignment_config)
        self.role_code_builder = RoleBiasCodeBuilder(self.role_assignment_config)
        self.role_local_adapter_enabled = bool(
            getattr(self.model_flags, "role_local_adapter_enabled", False)
        )
        role_trace_base = os.environ.get("LUX_ROLE_TRACE_PATH")
        self.role_trace_handle = None
        if role_trace_base:
            trace_path = Path(f"{role_trace_base}.p{int(obs.player)}.jsonl")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.role_trace_handle = trace_path.open("w", encoding="utf-8", buffering=1)
        torch_num_threads = int(getattr(self.agent_flags, "runtime_torch_num_threads", 1))
        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
            try:
                torch.set_num_interop_threads(torch_num_threads)
            except RuntimeError:
                pass
        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(obs.player, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"
        self.device = torch.device(device_id)

        # Build the env used to convert observations for the model
        env = LuxEnv(
            act_space=self.model_flags.act_space(),
            obs_space=self.model_flags.obs_space(),
            configuration=conf,
            run_game_automatically=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = wrappers.PadFixedShapeEnv(env)
        env = wrappers.VecEnv([env])
        # We'll move the data onto the target device if necessary after preprocessing
        env = wrappers.PytorchEnv(env, torch.device("cpu"))
        env = wrappers.DictEnv(env)
        self.env = env
        self.env.reset(observation_updates=obs["updates"], force=True)
        self.action_placeholder = {
            key: torch.zeros(space.shape)
            for key, space in self.unwrapped_env.action_space.get_action_space().spaces.items()
        }

        # Load the model
        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()
        self.strategy_scorers = self.load_strategy_scorers()

        # Load the data augmenters
        self.data_augmentations = []
        da_factories = list(self.agent_flags.data_augmentations)
        fast_no_aug_map_size = getattr(self.agent_flags, "runtime_no_augmentations_above_map_size", 0)
        if fast_no_aug_map_size and self.game_state.map_width > fast_no_aug_map_size:
            da_factories = []
        for da_factory in da_factories:
            da = data_augmentation.__dict__[da_factory](game_state=self.game_state)
            if not isinstance(da, data_augmentation.DataAugmenter):
                raise ValueError(f"Unrecognized data augmentation '{da}' created by: {da_factory}")
            self.data_augmentations.append(da)

        # Various utility properties
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.runtime_gate_policy = RuntimeGatePolicy(self.agent_flags)
        self.my_city_tile_mat = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        # NB: loc = pos[0] * n_cols + pos[1]
        self.loc_to_actionable_city_tiles = {}
        self.loc_to_actionable_workers = {}
        self.loc_to_actionable_carts = {}

        # Logging
        self.stopwatch = Stopwatch()

    def city_fuel_turns(self, city) -> float:
        return self.runtime_gate_policy.city_fuel_turns(city)

    def city_fuel_buffer_summary(self) -> Dict[str, float]:
        return self.runtime_gate_policy.city_fuel_buffer_summary(self.me)

    def runtime_gate_active(self) -> bool:
        return self.runtime_gate_policy.active(self.game_state)

    def large_map_gate_strict(self, buffer: Dict[str, float]) -> bool:
        return self.runtime_gate_policy.large_map_gate_strict(self.game_state, buffer)

    def should_gate_build_worker(self, city_tile: CityTile, buffer: Dict[str, float]) -> bool:
        return self.runtime_gate_policy.should_gate_build_worker(
            self.game_state,
            self.me,
            city_tile,
            buffer,
        )

    def should_gate_build_city(self, unit: Unit, buffer: Dict[str, float]) -> bool:
        return self.runtime_gate_policy.should_gate_build_city(
            self.game_state,
            self.me,
            unit,
            buffer,
        )

    def load_strategy_scorers(self) -> Dict[str, Any]:
        if not getattr(self.agent_flags, "runtime_scorer_gate_enabled", False):
            return {}
        max_map_size = getattr(self.agent_flags, "runtime_scorer_gate_max_map_size", 0)
        if max_map_size and self.game_state.map_width > max_map_size:
            return {}
        try:
            import joblib
            scorers = {}
            for name in ("risk_big_loss_20", "error_failed_big_loss"):
                package = joblib.load(STRATEGY_SCORER_DIR / f"{name}_lgbm.joblib")
                scorers[name] = package.get("model", package) if isinstance(package, dict) else package
            return scorers
        except Exception as exc:
            DEBUG_MESSAGE(f"Runtime scorer gate disabled; failed to load scorers: {exc}")
            return {}

    def strategy_scorer_gate_active(self) -> bool:
        if not self.strategy_scorers:
            return False
        max_map_size = getattr(self.agent_flags, "runtime_scorer_gate_max_map_size", 0)
        if max_map_size and self.game_state.map_width > max_map_size:
            return False
        turn = self.game_state.turn
        if turn < self.map_gate_value("runtime_scorer_gate_disable_before_turn", 80):
            return False
        cycle_turn = turn % DN_CYCLE_LEN
        turns_to_night = max(30 - cycle_turn, 0) if cycle_turn < 30 else 0
        is_night = cycle_turn >= 30
        return (
            turn >= self.map_gate_value("runtime_scorer_gate_late_turn", 120)
            or (is_night and turn >= self.map_gate_value("runtime_scorer_gate_night_after_turn", 80))
            or turns_to_night <= getattr(self.agent_flags, "runtime_scorer_gate_pre_night_turns", 3)
        )

    def map_gate_value(self, name: str, default: Any) -> Any:
        mapping = getattr(self.agent_flags, f"{name}_by_map_size", None)
        if isinstance(mapping, dict):
            value = mapping.get(str(self.game_state.map_width), mapping.get(self.game_state.map_width, None))
            if value is not None:
                return value
        return getattr(self.agent_flags, name, default)

    def in_runtime_hold_mode(self) -> bool:
        hold_turn = self.map_gate_value("runtime_scorer_gate_hold_turn", 340)
        if self.game_state.turn < hold_turn:
            return False
        fuel = self.city_fuel_buffer_summary()
        min_city_tiles = self.map_gate_value("runtime_scorer_gate_hold_min_city_tiles", 10)
        min_p25_fuel = self.map_gate_value("runtime_scorer_gate_hold_min_p25_fuel", 5.0)
        return self.me.city_tile_count >= min_city_tiles and fuel["p25"] >= min_p25_fuel

    def map_resource_remaining(self) -> Tuple[float, float, float]:
        wood = coal = uranium = 0.0
        for row in self.game_state.map.map:
            for cell in row:
                if cell.resource is None:
                    continue
                if cell.resource.type == "wood":
                    wood += float(cell.resource.amount)
                elif cell.resource.type == "coal":
                    coal += float(cell.resource.amount)
                elif cell.resource.type == "uranium":
                    uranium += float(cell.resource.amount)
        return wood, coal, uranium

    def strategy_feature_vector(self) -> List[float]:
        turn = self.game_state.turn
        cycle_turn = turn % DN_CYCLE_LEN
        is_night = 1.0 if cycle_turn >= 30 else 0.0
        turns_to_night = 0.0 if is_night else float(max(30 - cycle_turn, 0))
        city_sizes = [len(city.citytiles) for city in self.me.cities.values()]
        fuel_buffer = self.city_fuel_buffer_summary()
        fuel = sum(float(city.fuel) for city in self.me.cities.values())
        upkeep = sum(float(city.get_light_upkeep()) for city in self.me.cities.values())
        workers = sum(1 for unit in self.me.units if unit.is_worker())
        carts = sum(1 for unit in self.me.units if unit.is_cart())
        cargo_fuel = sum(float(unit.cargo.wood + unit.cargo.coal * 10 + unit.cargo.uranium * 40) for unit in self.me.units)
        wood_remaining, coal_remaining, uranium_remaining = self.map_resource_remaining()
        values = {
            "map_size": float(self.game_state.map_width),
            "turn": float(turn),
            "turns_remaining": float(GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] - turn),
            "night_cycle": float(turn // DN_CYCLE_LEN),
            "cycle_turn": float(cycle_turn),
            "pre_night": 1.0 if 0 < turns_to_night <= 3 else 0.0,
            "is_night": is_night,
            "turns_to_night": turns_to_night,
            "team": float(self.me.team),
            "eval_side_numeric": float(self.me.team),
            "city_tiles": float(self.me.city_tile_count),
            "cities": float(len(self.me.cities)),
            "largest_city_size": float(max(city_sizes) if city_sizes else 0),
            "mean_city_size": float(np.mean(city_sizes) if city_sizes else 0.0),
            "resource_near_cities": 0.0,
            "isolated_cities_r3": 0.0,
            "units": float(len(self.me.units)),
            "workers": float(workers),
            "carts": float(carts),
            "unit_cap_margin": float(self.me.city_tile_count - len(self.me.units)),
            "worker_citytile_ratio": float(workers / max(self.me.city_tile_count, 1)),
            "research": float(self.me.research_points),
            "fuel": fuel,
            "upkeep": upkeep,
            "fuel_turns_total": fuel_buffer["total"],
            "min_city_fuel_turns": fuel_buffer["min"],
            "p25_city_fuel_turns": fuel_buffer["p25"],
            "median_city_fuel_turns": fuel_buffer["median"],
            "mean_city_fuel_turns": fuel_buffer["mean"],
            "low_fuel_city_lt3": float(sum(1 for city in self.me.cities.values() if self.city_fuel_turns(city) < 3)),
            "low_fuel_city_lt5": float(sum(1 for city in self.me.cities.values() if self.city_fuel_turns(city) < 5)),
            "low_fuel_city_lt10": float(sum(1 for city in self.me.cities.values() if self.city_fuel_turns(city) < 10)),
            "unit_cargo_fuel": cargo_fuel,
            "wood_remaining": wood_remaining,
            "coal_remaining": coal_remaining,
            "uranium_remaining": uranium_remaining,
        }
        return [float(values.get(feature, 0.0)) for feature in STRATEGY_SCORER_FEATURES]

    def strategy_risk_scores(self) -> Dict[str, float]:
        if not self.strategy_scorer_gate_active():
            return {"risk_big_loss_20": 0.0, "error_failed_big_loss": 0.0}
        features = np.array([self.strategy_feature_vector()], dtype=float)
        scores = {}
        for name, model in self.strategy_scorers.items():
            try:
                scores[name] = float(model.predict_proba(features)[0, 1])
            except Exception as exc:
                DEBUG_MESSAGE(f"Runtime scorer prediction failed for {name}: {exc}")
                scores[name] = 0.0
        return scores

    def should_gate_scorer_action(self, action_meaning: str, scores: Dict[str, float]) -> bool:
        gated_actions = getattr(self.agent_flags, "runtime_scorer_gate_actions", ["BUILD_WORKER", "BUILD_CITY"])
        if action_meaning not in set(gated_actions):
            return False
        if self.in_runtime_hold_mode() and action_meaning == "BUILD_WORKER":
            return True
        return (
            scores.get("risk_big_loss_20", 0.0) >= self.map_gate_value("runtime_scorer_gate_risk_big_loss_threshold", 0.40)
            or scores.get("error_failed_big_loss", 0.0) >= self.map_gate_value("runtime_scorer_gate_error_failed_threshold", 0.35)
        )

    def __call__(self, obs, conf, raw_model_output: bool = False):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        self.preprocess(obs, conf)
        role_active = self.role_assignment_config.bias_active_for(
            int(self.game_state.map_width), int(obs.player)
        )
        if role_active or self.role_trace_handle is not None:
            role_snapshot = self.role_city_adapter.update(
                game_state=self.game_state,
                player=self.me,
                opponent=self.opp,
            )
        else:
            self.role_city_adapter.deactivate()
            role_snapshot = None
        self.write_role_trace(role_snapshot, role_active, int(obs.player))
        if role_snapshot is not None and self.role_assignment_config.dry_run_logging:
            DEBUG_MESSAGE(role_snapshot.summary(
                max_units=self.role_assignment_config.max_log_units,
                max_cities=self.role_assignment_config.max_log_cities,
            ))
        env_output = self.get_env_output()
        role_bias_codes = None
        if self.role_local_adapter_enabled and role_snapshot is not None and role_active:
            role_bias_codes = self.role_code_builder.build_player_from_snapshot(
                self.game_state,
                self.me,
                self.opp,
                role_snapshot,
                env_output["info"]["available_actions_mask"],
            )
        relevant_env_output_augmented = {
            "obs": self.augment_data(env_output["obs"], is_policy=False),
            "info": {
                "input_mask": self.augment_data(env_output["info"]["input_mask"].unsqueeze(1),
                                                is_policy=False).squeeze(1),
                "available_actions_mask": self.augment_data(env_output["info"]["available_actions_mask"],
                                                            is_policy=True),
            },
        }
        if role_bias_codes is not None:
            relevant_env_output_augmented["info"]["role_bias_codes"] = self.augment_data(
                role_bias_codes, is_policy=True
            )
            role_scale = self.role_assignment_config.bias_scale_for(
                int(self.game_state.map_width)
            )
            augmentation_count = len(self.data_augmentations) + 1
            relevant_env_output_augmented["info"]["role_bias_scale"] = torch.full(
                (augmentation_count, 1, 1, 1, 1, 1),
                float(role_scale),
                dtype=torch.float32,
                device=self.device,
            )

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)
            agent_output = {
                "policy_logits": self.aggregate_augmented_predictions(agent_output_augmented["policy_logits"]),
                "baseline": agent_output_augmented["baseline"].mean(dim=0, keepdim=True).cpu()
            }
            if not self.role_local_adapter_enabled:
                agent_output["policy_logits"] = self.role_city_adapter.apply(
                    game_state=self.game_state,
                    player=self.me,
                    opponent=self.opp,
                    actionable_workers=self.loc_to_actionable_workers,
                    actionable_city_tiles=self.loc_to_actionable_city_tiles,
                    policy_logits=agent_output["policy_logits"],
                    available_actions_mask=env_output["info"]["available_actions_mask"],
                    player_id=obs.player,
                )
            agent_output["actions"] = {
                key: models.DictActor.logits_to_actions(
                    torch.flatten(val, start_dim=0, end_dim=-2),
                    sample=False,
                    actions_per_square=None
                ).view(*val.shape[:-1], -1)
                for key, val in agent_output["policy_logits"].items()
            }
        # Used for debugging and visualization
        if raw_model_output:
            return agent_output

        self.stopwatch.stop().start("Collision detection")
        if self.agent_flags.use_collision_detection:
            actions = self.resolve_collision_detection(obs, agent_output)
        else:
            actions, _ = self.unwrapped_env.process_actions({
                key: value.squeeze(0).numpy() for key, value in agent_output["actions"].items()
            })
            actions = actions[obs.player]
        self.stopwatch.stop()

        if LOCAL_EVAL:
            # Add transfer annotations locally
            actions.extend(self.get_transfer_annotations(actions))

        value = agent_output["baseline"].squeeze().numpy()[obs.player]
        value_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        actions.append(annotate.sidetext(value_msg))
        DEBUG_MESSAGE(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return actions

    def write_role_trace(self, snapshot, bias_active: bool, player_id: int) -> None:
        if self.role_trace_handle is None or snapshot is None:
            return
        units_by_id = {unit.id: unit for unit in self.me.units}
        unit_records = []
        for unit_id, assignment in snapshot.unit_roles.items():
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            unit_records.append({
                "id": unit_id,
                "x": int(unit.pos.x),
                "y": int(unit.pos.y),
                "role": assignment.role,
                "desired_role": assignment.desired_role,
                "changed": bool(assignment.changed),
                "cooldown": int(assignment.cooldown_remaining),
                "reason": assignment.reason,
            })
        city_records = []
        for city_id, specialization in snapshot.city_roles.items():
            city = self.me.cities.get(city_id)
            if city is None:
                continue
            city_records.append({
                "id": city_id,
                "role": specialization.role,
                "reason": specialization.reason,
                "nights_of_fuel": float(specialization.nights_of_fuel),
                "abandon": bool(specialization.abandon),
                "tiles": [
                    {"x": int(tile.pos.x), "y": int(tile.pos.y)}
                    for tile in city.citytiles
                ],
            })
        record = {
            "turn": int(self.game_state.turn),
            "player": player_id,
            "map_size": int(self.game_state.map_width),
            "bias_active": bool(bias_active),
            "bias_scale": float(self.role_assignment_config.bias_scale_for(self.game_state.map_width)),
            "update_seconds": float(self.role_city_adapter.last_update_seconds),
            "update_degraded": bool(self.role_city_adapter.last_update_degraded),
            "units": unit_records,
            "cities": city_records,
        }
        self.role_trace_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def preprocess(self, obs, conf) -> NoReturn:
        # Do not call manual_step on the first turn, or you will be off-by-1 turn the entire game
        if obs["step"] > 0:
            self.unwrapped_env.manual_step(obs["updates"])

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        self.my_city_tile_mat[:] = False
        self.loc_to_actionable_city_tiles: Dict[int, CityTile] = {}
        self.loc_to_actionable_workers: Dict[int, List[Unit]] = {}
        self.loc_to_actionable_carts: Dict[int, List[Unit]] = {}
        for unit in self.me.units:
            if unit.can_act():
                if unit.is_worker():
                    dictionary = self.loc_to_actionable_workers
                elif unit.is_cart():
                    dictionary = self.loc_to_actionable_carts
                else:
                    DEBUG_MESSAGE(f"Unrecognized unit type: {unit}")
                    continue
                dictionary.setdefault(pos_to_loc(unit.pos.astuple()), []).append(unit)
        for city_tile in self.me.city_tiles:
            self.my_city_tile_mat[city_tile.pos.x, city_tile.pos.y] = True
            if city_tile.can_act():
                self.loc_to_actionable_city_tiles[pos_to_loc(city_tile.pos.astuple())] = city_tile

        # Remove data augmentations if there are fewer overage seconds than 2x the number of data augmentations
        while max(obs["remainingOverageTime"], 0.) < len(self.data_augmentations) * 2:
            DEBUG_MESSAGE(f"Removing data augmentation: {self.data_augmentations[-1]}")
            del self.data_augmentations[-1]

    def get_env_output(self) -> Dict:
        return self.env.step(self.action_placeholder)

    def augment_data(
            self,
            data: Union[torch.Tensor, Dict[str, torch.Tensor]],
            is_policy: bool
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies and concatenates all augmented observations into a single tensor/dict of tensors and moves the tensor
        to the correct device for inference.
        """
        if isinstance(data, dict):
            augmented_data = [data] + [augmentation.apply(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return {
                key: torch.cat([d[key] for d in augmented_data], dim=0).to(device=self.device)
                for key in data.keys()
            }
        else:
            augmented_data = [data] + [augmentation.op(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return torch.cat(augmented_data, dim=0).to(device=self.device)

    def aggregate_augmented_predictions(self, policy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Moves the predictions to the cpu, applies the inverse of all augmentations,
        and then returns the mean prediction for each available action.
        """
        policy = {key: val.cpu() for key, val in policy.items()}
        if len(self.data_augmentations) == 0:
            return policy

        policy_reoriented = [{key: val[0].unsqueeze(0) for key, val in policy.items()}]
        for i, augmentation in enumerate(self.data_augmentations):
            augmented_policy = {key: val[i + 1].unsqueeze(0) for key, val in policy.items()}
            policy_reoriented.append(augmentation.apply(augmented_policy, inverse=True, is_policy=True))
        return {
            key: torch.cat([d[key] for d in policy_reoriented], dim=0).mean(dim=0, keepdim=True)
            for key in policy.keys()
        }

    def resolve_collision_detection(self, obs, agent_output) -> List[str]:
        # Get log_probs for all of my actions
        flat_log_probs = {
            key: torch.flatten(
                F.log_softmax(val.squeeze(0).squeeze(0), dim=-1),
                start_dim=-3,
                end_dim=-2
            )
            for key, val in agent_output["policy_logits"].items()
        }
        my_flat_log_probs = {
            key: val[obs.player] for key, val in flat_log_probs.items()
        }
        my_flat_actions = {
            key: torch.flatten(
                val.squeeze(0).squeeze(0)[obs.player],
                start_dim=-3,
                end_dim=-2
            )
            for key, val in agent_output["actions"].items()
        }
        # Use actions with highest prob/log_prob as highest priority
        city_tile_priorities = torch.argsort(my_flat_log_probs["city_tile"].max(dim=-1)[0], dim=-1, descending=True)

        # First handle city tile actions, ensuring the unit cap and research cap is not exceeded
        units_to_build = max(self.me.city_tile_count - len(self.me.units), 0)
        research_remaining = max(MAX_RESEARCH - self.me.research_points, 0)
        fuel_buffer = self.city_fuel_buffer_summary()
        strategy_scores = self.strategy_risk_scores()
        for loc in city_tile_priorities:
            loc = loc.item()
            actions = my_flat_actions["city_tile"][loc]
            city_tile = self.loc_to_actionable_city_tiles.get(loc, None)
            if city_tile is not None:
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS["city_tile"][act]
                    # Check that it is allowed to build carts
                    if action_meaning == "BUILD_CART" and not self.agent_flags.can_build_carts:
                        illegal_action = True
                    elif (
                            action_meaning == "BUILD_WORKER"
                            and self.should_gate_build_worker(city_tile, fuel_buffer)
                    ) or self.should_gate_scorer_action(action_meaning, strategy_scores):
                        illegal_action = True
                    # Check that the city will not build more units than the unit cap
                    elif action_meaning.startswith("BUILD_"):
                        if units_to_build > 0:
                            units_to_build -= 1
                        else:
                            illegal_action = True
                    # Check that the city will not research more than the research cap
                    elif action_meaning == "RESEARCH":
                        if research_remaining > 0:
                            research_remaining -= 1
                        else:
                            illegal_action = True
                    # Ban no-ops after the first night until research is complete
                    # This might prevent games like this from happening:
                    # https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-episode-26458475
                    elif (
                            action_meaning == "NO-OP" and
                            self.game_state.turn >= DN_CYCLE_LEN and
                            research_remaining > 0 and
                            self.agent_flags.must_research
                    ):
                        illegal_action = True
                    # Ban all non-unit-creating actions on the final step
                    if self.game_state.turn >= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] - 1:
                        if action_meaning == "BUILD_CART":
                            illegal_action = False
                        else:
                            illegal_action = True
                    if illegal_action:
                        my_flat_log_probs["city_tile"][loc, act] = float("-inf")
                    else:
                        break

        # Then handle unit actions, ensuring that no units try to move to the same square
        occupied_squares = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        max_loc_val = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]
        combined_unit_log_probs = torch.cat(
            [my_flat_log_probs["worker"].max(dim=-1)[0], my_flat_log_probs["cart"].max(dim=-1)[0]],
            dim=-1
        )
        unit_priorities = torch.argsort(combined_unit_log_probs, dim=-1, descending=True)
        for loc in unit_priorities:
            loc = loc.item()
            if loc >= max_loc_val:
                unit_type = "cart"
                actionable_dict = self.loc_to_actionable_carts
            else:
                unit_type = "worker"
                actionable_dict = self.loc_to_actionable_workers
            loc = loc % max_loc_val
            actions = my_flat_actions[unit_type][loc]
            actionable_list = actionable_dict.get(loc, None)
            if actionable_list is not None:
                acted_count = 0
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS[unit_type][act]
                    unit = actionable_list[acted_count]
                    if self.should_gate_scorer_action(action_meaning, strategy_scores):
                        illegal_action = True
                        new_pos = unit.pos
                    elif action_meaning.startswith("MOVE_"):
                        direction = action_meaning.split("_")[1]
                        new_pos = unit.pos.translate(direction, 1)
                    else:
                        new_pos = unit.pos

                    if not illegal_action:
                        if (
                                unit_type == "worker"
                                and action_meaning == "BUILD_CITY"
                                and self.should_gate_build_city(unit, fuel_buffer)
                        ):
                            illegal_action = True
                        # Check that the new position is a legal square
                        elif (
                                new_pos.x < 0 or new_pos.x >= self.game_state.map_width or
                                new_pos.y < 0 or new_pos.y >= self.game_state.map_height
                        ):
                            illegal_action = True
                        # Check that the new position does not conflict with another unit's new position
                        elif occupied_squares[new_pos.x, new_pos.y] and not self.my_city_tile_mat[new_pos.x, new_pos.y]:
                            illegal_action = True
                        else:
                            occupied_squares[new_pos.x, new_pos.y] = True

                    if illegal_action:
                        my_flat_log_probs[unit_type][loc, act] = float("-inf")
                    else:
                        acted_count += 1

                    if acted_count >= len(actionable_list):
                        break

        # Finally, get new actions from the modified log_probs
        actions_tensors = {
            key: val.view(1, *val.shape[:-2], *MAX_BOARD_SIZE, -1).argsort(dim=-1, descending=True)
            for key, val in flat_log_probs.items()
        }
        actions, _ = self.unwrapped_env.process_actions({
            key: value.numpy() for key, value in actions_tensors.items()
        })
        actions = actions[obs.player]
        return actions

    def get_transfer_annotations(self, actions: List[str]) -> List[str]:
        annotations = []
        for act in actions:
            act_split = act.split(" ")
            if act_split[0] == "t":
                unit_from = self.me.get_unit_by_id(act_split[1])
                unit_to = self.me.get_unit_by_id(act_split[2])
                if unit_from is None or unit_to is None:
                    DEBUG_MESSAGE(f"Unrecognized transfer: {act}")
                    continue
                annotations.append(annotate.line(unit_from.pos.x, unit_from.pos.y, unit_to.pos.x, unit_to.pos.y))
                annotations.append(annotate.x(unit_to.pos.x, unit_to.pos.y))
        return annotations

    @property
    def unwrapped_env(self) -> LuxEnv:
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        return self.unwrapped_env.game_state

    # Helper functions for debugging
    def set_to_turn_and_call(self, turn: int, *args, **kwargs):
        self.game_state.turn = max(turn - 1, 0)
        return self(*args, **kwargs)


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
