from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from scipy.stats import rankdata
from typing import Dict, NamedTuple, NoReturn, Tuple

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import Player


def count_city_tiles(game_state: Game) -> np.ndarray:
    return np.array([player.city_tile_count for player in game_state.players])


def count_units(game_state: Game) -> np.ndarray:
    return np.array([len(player.units) for player in game_state.players])


def count_total_fuel(game_state: Game) -> np.ndarray:
    return np.array([
        sum([city.fuel for city in player.cities.values()])
        for player in game_state.players
    ])


def count_research_points(game_state: Game) -> np.ndarray:
    return np.array([player.research_points for player in game_state.players])


def should_early_stop(game_state: Game) -> bool:
    ct_count = count_city_tiles(game_state)
    unit_count = count_units(game_state)
    ct_pct = ct_count / max(ct_count.sum(), 1)
    unit_pct = unit_count / max(unit_count.sum(), 1)
    return ((ct_count == 0).any() or
            (unit_count == 0).any() or
            (ct_pct >= 0.75).any() or
            (unit_pct >= 0.75).any())


class RewardSpec(NamedTuple):
    reward_min: float
    reward_max: float
    zero_sum: bool
    only_once: bool


# All reward spaces defined below

class BaseRewardSpace(ABC):
    """
    A class used for defining a reward space and/or done state for either the full game or a sub-task
    """
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}


# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        pass


class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def __init__(self, early_stop: bool = False, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        ct_count = player.city_tile_count
        unit_count = len(player.units)
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return ct_count * 10000 + unit_count


class CityTileReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        return tuple(count_city_tiles(game_state) / 1024.)


class LogScaleOutcomeReward(FullGameRewardSpace):
    """Outcome-dominant reward with bounded logarithmic scale progress."""

    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.0,
            reward_max=1.0,
            zero_sum=False,
            only_once=False,
        )

    def __init__(
            self,
            outcome_weight: float = 0.80,
            city_weight: float = 0.12,
            unit_weight: float = 0.08,
            early_stop: bool = False,
            **kwargs,
    ):
        # Parent reward configs may be merged by Hydra. They are intentionally
        # irrelevant to this bounded reward and must not alter its semantics.
        super().__init__()
        if outcome_weight < 0.80:
            raise ValueError("outcome_weight must be at least 0.80")
        if min(city_weight, unit_weight) < 0.0:
            raise ValueError("Scale reward weights must be non-negative")
        if outcome_weight + city_weight + unit_weight > 1.0 + 1e-9:
            raise ValueError("Reward weights must sum to at most 1.0")
        self.outcome_weight = float(outcome_weight)
        self.city_weight = float(city_weight)
        self.unit_weight = float(unit_weight)
        self.early_stop = bool(early_stop)
        self._normalizer = float(np.log1p(1024.0))
        self._city_log = np.zeros(2, dtype=float)
        self._unit_log = np.zeros(2, dtype=float)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        city_log = np.log1p(count_city_tiles(game_state).astype(float))
        unit_log = np.log1p(count_units(game_state).astype(float))
        reward = (
            self.city_weight * (city_log - self._city_log) / self._normalizer
            + self.unit_weight * (unit_log - self._unit_log) / self._normalizer
        )
        if done:
            scores = np.array(
                [GameResultReward.compute_player_reward(player) for player in game_state.players],
                dtype=float,
            )
            outcome = (rankdata(scores) - 1.0) * 2.0 - 1.0
            reward = reward + self.outcome_weight * outcome
            self._city_log.fill(0.0)
            self._unit_log.fill(0.0)
        else:
            self._city_log = city_log
            self._unit_log = unit_log
        return tuple(reward)


class StatefulMultiReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.,
            early_stop: bool = False,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 10.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.1,
            "fuel": 0.005,
            # Penalize workers each step that their cargo remains full
            # "full_workers": -0.01,
            "full_workers": 0.,
            # A reward given each step
            "step": 0.,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(StatefulMultiReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        reward_items_dict = {
            "city": new_city_count - self.city_count,
            "unit": new_unit_count - self.unit_count,
            "research": new_research_points - self.research_points,
            # Don't penalize losing fuel at night
            "fuel": np.maximum(new_total_fuel - self.total_fuel, 0),
            "full_workers": np.array([
                sum(unit.get_cargo_space_left() > 0 for unit in player.units if unit.is_worker())
                for player in game_state.players
            ]),
            "step": np.ones(2, dtype=float)
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [self.weight_rewards(reward_items_dict[key] * w) for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        return tuple(reward / 500. / max(self.positive_weight, self.negative_weight))

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


class ZeroSumStatefulMultiReward(StatefulMultiReward):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        reward = np.array(super(ZeroSumStatefulMultiReward, self).compute_rewards(game_state, done))
        return tuple(reward - reward.mean())


class WinScaleCatastrophicGuardReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.25,
            early_stop: bool = False,
            reward_scale: float = 500.,
            result: float = 80.,
            final_city_margin: float = 1.8,
            final_unit_margin: float = 0.45,
            city_gain: float = 0.10,
            unit_gain: float = 0.04,
            catastrophic_city_loss: float = 12.,
            catastrophic_loss_threshold: float = 4.,
            losing_loss_multiplier: float = 2.0,
            scale_shortfall: float = 0.20,
            scale_target_start_turn: int = 120,
            scale_target_mid_turn: int = 200,
            scale_target_late_turn: int = 300,
            scale_target_early_density: float = 0.04,
            scale_target_mid_density: float = 0.10,
            scale_target_late_density: float = 0.16,
            safe_scale_bonus: float = 0.18,
            safe_scale_min_buffer: float = 6.,
            safe_scale_p25_buffer: float = 10.,
            unsafe_large_city: float = 4.,
            unsafe_large_city_start_turn: int = 240,
            unsafe_large_city_late_turn: int = 320,
            unsafe_large_city_min_tiles: int = 18,
            unsafe_large_city_target_turns: float = 12.,
            unsafe_large_city_late_target_turns: float = 22.,
            buffer_drop_with_growth: float = 8.,
            buffer_growth_drop_threshold: float = 4.,
            buffer_growth_drop_clip: float = 16.,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        assert reward_scale > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop
        self.reward_scale = reward_scale
        self.result = result
        self.final_city_margin = final_city_margin
        self.final_unit_margin = final_unit_margin
        self.city_gain = city_gain
        self.unit_gain = unit_gain
        self.catastrophic_city_loss = catastrophic_city_loss
        self.catastrophic_loss_threshold = catastrophic_loss_threshold
        self.losing_loss_multiplier = losing_loss_multiplier
        self.scale_shortfall = scale_shortfall
        self.scale_target_start_turn = scale_target_start_turn
        self.scale_target_mid_turn = scale_target_mid_turn
        self.scale_target_late_turn = scale_target_late_turn
        self.scale_target_early_density = scale_target_early_density
        self.scale_target_mid_density = scale_target_mid_density
        self.scale_target_late_density = scale_target_late_density
        self.safe_scale_bonus = safe_scale_bonus
        self.safe_scale_min_buffer = safe_scale_min_buffer
        self.safe_scale_p25_buffer = safe_scale_p25_buffer
        self.unsafe_large_city = unsafe_large_city
        self.unsafe_large_city_start_turn = unsafe_large_city_start_turn
        self.unsafe_large_city_late_turn = unsafe_large_city_late_turn
        self.unsafe_large_city_min_tiles = unsafe_large_city_min_tiles
        self.unsafe_large_city_target_turns = unsafe_large_city_target_turns
        self.unsafe_large_city_late_target_turns = unsafe_large_city_late_target_turns
        self.buffer_drop_with_growth = buffer_drop_with_growth
        self.buffer_growth_drop_threshold = buffer_growth_drop_threshold
        self.buffer_growth_drop_clip = buffer_growth_drop_clip

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.total_fuel_turns = np.empty_like(self.city_count)
        super(WinScaleCatastrophicGuardReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_total_fuel_turns = SurvivalResearchReward.count_total_fuel_turns(game_state)
        min_buffer = SurvivalResearchReward.count_fuel_buffer(self, game_state)
        p25_buffer = SurvivalResearchReward.count_fuel_buffer_percentile(game_state, 0.25)

        city_diff = new_city_count - self.city_count
        unit_diff = new_unit_count - self.unit_count
        city_loss = np.maximum(-city_diff, 0.)
        catastrophic_loss = np.maximum(city_loss - self.catastrophic_loss_threshold, 0.)
        scale_target = self.count_scale_target(game_state)
        scale_progress = np.minimum(new_city_count / max(scale_target, 1.), 1.) if scale_target > 0. else np.zeros(2)
        scale_shortfall = -np.maximum(scale_target - new_city_count, 0.) / max(scale_target, 1.) if scale_target > 0. else np.zeros(2)
        safe_scale_health = (
            SurvivalResearchReward.count_buffer_health(min_buffer, self.safe_scale_min_buffer)
            * SurvivalResearchReward.count_buffer_health(p25_buffer, self.safe_scale_p25_buffer)
        )
        unsafe_large_city = -self.count_unsafe_large_city_pressure(game_state)
        buffer_drop_with_growth = -np.maximum(
            self.total_fuel_turns - new_total_fuel_turns - self.buffer_growth_drop_threshold,
            0.,
        ) / max(self.buffer_growth_drop_clip, 1e-6) * np.maximum(city_diff, 0.)

        reward = (
            np.maximum(city_diff, 0.) * self.city_gain
            + np.maximum(unit_diff, 0.) * self.unit_gain
            - catastrophic_loss * self.catastrophic_city_loss
            + scale_shortfall * self.scale_shortfall
            + scale_progress * safe_scale_health * self.safe_scale_bonus
            + unsafe_large_city * self.unsafe_large_city
            + buffer_drop_with_growth * self.buffer_drop_with_growth
        )

        if done:
            player_scores = np.array([GameResultReward.compute_player_reward(p) for p in game_state.players], dtype=float)
            result_reward = (rankdata(player_scores) - 1.) * 2. - 1.
            city_margin = new_city_count - new_city_count[::-1]
            unit_margin = new_unit_count - new_unit_count[::-1]
            losing = (result_reward < 0.).astype(float)
            reward = reward + (
                result_reward * self.result
                + np.tanh(city_margin / 20.) * self.final_city_margin
                + np.tanh(unit_margin / 20.) * self.final_unit_margin
                - catastrophic_loss * self.catastrophic_city_loss * self.losing_loss_multiplier * losing
            )
            self._reset()
        else:
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.total_fuel_turns = new_total_fuel_turns

        return tuple(self.weight_rewards(reward) / self.reward_scale / max(self.positive_weight, self.negative_weight))

    def count_scale_target(self, game_state: Game) -> float:
        if game_state.turn < self.scale_target_start_turn:
            return 0.
        area = game_state.map_width * game_state.map_height
        if game_state.turn < self.scale_target_mid_turn:
            density = self.scale_target_early_density
        elif game_state.turn < self.scale_target_late_turn:
            density = self.scale_target_mid_density
        else:
            density = self.scale_target_late_density
        return max(1., area * density)

    def count_unsafe_large_city_pressure(self, game_state: Game) -> np.ndarray:
        if game_state.turn < self.unsafe_large_city_start_turn:
            return np.zeros((2,), dtype=float)
        target_turns = (
            self.unsafe_large_city_late_target_turns
            if game_state.turn >= self.unsafe_large_city_late_turn
            else self.unsafe_large_city_target_turns
        )
        pressure = []
        for player in game_state.players:
            player_pressure = 0.
            for city in player.cities.values():
                tiles = len(city.citytiles)
                if tiles < self.unsafe_large_city_min_tiles:
                    continue
                fuel_turns = city.fuel / max(city.light_upkeep, 1.)
                shortfall = max(target_turns - fuel_turns, 0.) / max(target_turns, 1e-6)
                player_pressure += shortfall * tiles / max(self.unsafe_large_city_min_tiles, 1)
            pressure.append(player_pressure)
        return np.array(pressure, dtype=float)

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(reward > 0., self.positive_weight * reward, reward)
        reward = np.where(reward < 0., self.negative_weight * reward, reward)
        return reward

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.total_fuel_turns = np.zeros_like(self.total_fuel_turns)


class SurvivalResearchReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.5,
            early_stop: bool = False,
            reward_scale: float = 500.,
            pre_night_window: int = 10,
            min_buffer_nights: float = 1.,
            endgame_start_turn: int = 320,
            fuel_deficit_clip: float = 10.,
            midgame_start_turn: int = 70,
            midgame_end_turn: int = 160,
            midgame_ramp_turns: int = 10,
            research_coal_deadline_turn: int = 80,
            research_uranium_deadline_turn: int = 200,
            unsafe_new_city_start_turn: int = 60,
            unsafe_new_city_end_turn: int = 120,
            unsafe_new_city_ramp_turns: int = 10,
            research_progress_end_turn: int = 120,
            research_late_scale: float = 1.,
            strategy_buffer_start_turn: int = 120,
            strategy_buffer_mid_turn: int = 160,
            strategy_buffer_late_turn: int = 240,
            min_buffer_target_early: float = 8.,
            min_buffer_target_mid: float = 10.,
            p25_buffer_target_mid: float = 15.,
            p25_buffer_target_late: float = 20.,
            risk_adjusted_size_start_turn: int = 0,
            risk_adjusted_size_buffer_blend: float = 0.5,
            scale_target_start_turn: int = 120,
            scale_target_mid_turn: int = 160,
            scale_target_late_turn: int = 240,
            scale_target_early_density: float = 0.04,
            scale_target_mid_density: float = 0.10,
            scale_target_late_density: float = 0.18,
            scale_target_end_density: float = 0.22,
            unsafe_large_city_start_turn: int = 240,
            unsafe_large_city_late_turn: int = 320,
            unsafe_large_city_min_tiles: int = 20,
            unsafe_large_city_target_turns: float = 12.,
            unsafe_large_city_late_target_turns: float = 20.,
            critical_survival_start_turn: int = 280,
            critical_survival_late_turn: int = 320,
            critical_survival_target_turns: float = 14.,
            critical_survival_late_target_turns: float = 24.,
            critical_survival_deficit_clip: float = 40.,
            buffer_growth_drop_threshold: float = 5.,
            buffer_growth_drop_clip: float = 20.,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        assert reward_scale > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop
        self.reward_scale = reward_scale
        self.pre_night_window = pre_night_window
        self.min_buffer_nights = min_buffer_nights
        self.endgame_start_turn = endgame_start_turn
        self.fuel_deficit_clip = fuel_deficit_clip
        self.midgame_start_turn = midgame_start_turn
        self.midgame_end_turn = midgame_end_turn
        self.midgame_ramp_turns = midgame_ramp_turns
        self.research_coal_deadline_turn = research_coal_deadline_turn
        self.research_uranium_deadline_turn = research_uranium_deadline_turn
        self.unsafe_new_city_start_turn = unsafe_new_city_start_turn
        self.unsafe_new_city_end_turn = unsafe_new_city_end_turn
        self.unsafe_new_city_ramp_turns = unsafe_new_city_ramp_turns
        self.research_progress_end_turn = research_progress_end_turn
        self.research_late_scale = research_late_scale
        self.strategy_buffer_start_turn = strategy_buffer_start_turn
        self.strategy_buffer_mid_turn = strategy_buffer_mid_turn
        self.strategy_buffer_late_turn = strategy_buffer_late_turn
        self.min_buffer_target_early = min_buffer_target_early
        self.min_buffer_target_mid = min_buffer_target_mid
        self.p25_buffer_target_mid = p25_buffer_target_mid
        self.p25_buffer_target_late = p25_buffer_target_late
        self.risk_adjusted_size_start_turn = risk_adjusted_size_start_turn
        self.risk_adjusted_size_buffer_blend = risk_adjusted_size_buffer_blend
        self.scale_target_start_turn = scale_target_start_turn
        self.scale_target_mid_turn = scale_target_mid_turn
        self.scale_target_late_turn = scale_target_late_turn
        self.scale_target_early_density = scale_target_early_density
        self.scale_target_mid_density = scale_target_mid_density
        self.scale_target_late_density = scale_target_late_density
        self.scale_target_end_density = scale_target_end_density
        self.unsafe_large_city_start_turn = unsafe_large_city_start_turn
        self.unsafe_large_city_late_turn = unsafe_large_city_late_turn
        self.unsafe_large_city_min_tiles = unsafe_large_city_min_tiles
        self.unsafe_large_city_target_turns = unsafe_large_city_target_turns
        self.unsafe_large_city_late_target_turns = unsafe_large_city_late_target_turns
        self.critical_survival_start_turn = critical_survival_start_turn
        self.critical_survival_late_turn = critical_survival_late_turn
        self.critical_survival_target_turns = critical_survival_target_turns
        self.critical_survival_late_target_turns = critical_survival_late_target_turns
        self.critical_survival_deficit_clip = critical_survival_deficit_clip
        self.buffer_growth_drop_threshold = buffer_growth_drop_threshold
        self.buffer_growth_drop_clip = buffer_growth_drop_clip
        self.min_buffer_turns = (
            min_buffer_nights * GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        )

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)
        self.total_fuel_turns = np.empty_like(self.city_count)
        self.fuel_buffer = np.empty_like(self.city_count)
        self.weighted_buffer_deficit = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 10.,
            "city_gain": 1.5,
            "city_loss": 6.,
            "unit": 0.25,
            "research": 0.08,
            "small_map_research": 0.,
            "small_map_coal_deadline": 0.,
            "small_map_uranium_deadline": 0.,
            "coal_milestone": 10.,
            "uranium_milestone": 20.,
            "fuel_gain": 0.004,
            "buffer_gain": 0.03,
            "buffer_deficit": 0.06,
            "unsafe_expansion": 4.,
            "unsafe_new_city": 0.,
            "midgame_unsafe_new_city": 0.,
            "endgame_buffer_deficit": 0.02,
            "late_expansion": 8.,
            "midgame_buffer_health": 0.,
            "strategy_min_buffer": 0.,
            "strategy_p25_buffer": 0.,
            "risk_adjusted_city_gain": 0.,
            "risk_adjusted_unit_gain": 0.,
            "scale_target_bonus": 0.,
            "scale_target_shortfall": 0.,
            "safe_scale_target_bonus": 0.,
            "unsafe_large_city": 0.,
            "critical_survival_deficit": 0.,
            "critical_low_buffer_expansion": 0.,
            "buffer_drop_with_growth": 0.,
            "survival_step": 0.02,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(SurvivalResearchReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)
        new_total_fuel_turns = self.count_total_fuel_turns(game_state)
        new_fuel_buffer = self.count_fuel_buffer(game_state)
        new_p25_fuel_buffer = self.count_fuel_buffer_percentile(game_state, 0.25)
        next_night_target_turns = self.count_next_night_target_turns(game_state.turn)
        new_weighted_buffer_deficit = self.count_weighted_buffer_deficit(
            game_state,
            next_night_target_turns,
            self.fuel_deficit_clip,
        )
        remaining_night_turns = self.count_remaining_night_turns(game_state.turn)
        endgame_buffer_deficit = self.count_weighted_buffer_deficit(
            game_state,
            remaining_night_turns,
            self.fuel_deficit_clip,
        )
        full_night_turns = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        full_night_deficit = self.count_weighted_buffer_deficit(
            game_state,
            full_night_turns,
            self.fuel_deficit_clip,
        )

        city_diff = new_city_count - self.city_count
        research_diff = new_research_points - self.research_points
        late_research_scale = (
            1.
            if game_state.turn < self.research_progress_end_turn
            else self.research_late_scale
        )
        small_map = float(max(game_state.map_width, game_state.map_height) <= 16)
        below_uranium = (self.research_points < 200).astype(float)
        endgame_active = float(game_state.turn >= self.endgame_start_turn)
        midgame_weight = self.smooth_window_weight(
            game_state.turn,
            self.midgame_start_turn,
            self.midgame_end_turn,
            self.midgame_ramp_turns,
        )
        active = ((new_city_count + new_unit_count) > 0).astype(float)
        has_city = (new_city_count > 0).astype(float)
        crossed_coal_deadline = float(
            self.previous_turn < self.research_coal_deadline_turn <= game_state.turn
        )
        crossed_uranium_deadline = float(
            self.previous_turn < self.research_uranium_deadline_turn <= game_state.turn
        )
        coal_shortfall = np.maximum(50. - new_research_points, 0.) / 50.
        uranium_shortfall = np.maximum(200. - new_research_points, 0.) / 200.
        current_city_tiles = [
            {
                city_tile.pos.astuple(): city.cityid
                for city in player.cities.values()
                for city_tile in city.citytiles
            }
            for player in game_state.players
        ]
        if self.city_tiles is None:
            new_city_deficit = np.zeros((2,), dtype=float)
        else:
            target_night_turns = self.count_new_city_target_turns(game_state.turn)
            new_city_deficit = []
            for player_id, player in enumerate(game_state.players):
                new_positions = (
                    current_city_tiles[player_id].keys()
                    - self.city_tiles[player_id].keys()
                )
                affected_city_tile_counts = {}
                for position in new_positions:
                    city_id = current_city_tiles[player_id][position]
                    affected_city_tile_counts[city_id] = (
                        affected_city_tile_counts.get(city_id, 0) + 1
                    )
                deficit = sum(
                    max(
                        target_night_turns
                        - player.cities[city_id].fuel
                        / max(player.cities[city_id].light_upkeep, 1.),
                        0.,
                    ) / max(target_night_turns, 1.) * added_tile_count
                    for city_id, added_tile_count in affected_city_tile_counts.items()
                )
                new_city_deficit.append(deficit)
            new_city_deficit = np.array(new_city_deficit)
        unsafe_new_city_midgame_weight = self.smooth_window_weight(
            game_state.turn,
            self.unsafe_new_city_start_turn,
            self.unsafe_new_city_end_turn,
            self.unsafe_new_city_ramp_turns,
        )
        min_buffer_target = self.count_strategy_min_buffer_target(game_state.turn)
        p25_buffer_target = self.count_strategy_p25_buffer_target(game_state.turn)
        strategy_min_buffer_health = self.count_buffer_health(
            new_fuel_buffer,
            min_buffer_target,
        )
        strategy_p25_buffer_health = self.count_buffer_health(
            new_p25_fuel_buffer,
            p25_buffer_target,
        )
        risk_adjusted_size_health = self.count_risk_adjusted_size_health(
            new_fuel_buffer,
            min_buffer_target,
            new_p25_fuel_buffer,
            p25_buffer_target,
        )
        risk_adjusted_size_active = float(game_state.turn >= self.risk_adjusted_size_start_turn)
        scale_target = self.count_scale_target(game_state)
        scale_target_active = float(scale_target > 0.)
        scale_target_progress = np.minimum(
            new_city_count / max(scale_target, 1.),
            1.,
        ) * scale_target_active
        scale_target_shortfall = -np.maximum(
            scale_target - new_city_count,
            0.,
        ) / max(scale_target, 1.) * scale_target_active
        safe_scale_target_progress = scale_target_progress * risk_adjusted_size_health
        unsafe_large_city = -self.count_unsafe_large_city_pressure(game_state)
        critical_survival_target = self.count_critical_survival_target(game_state.turn)
        critical_survival_active = float(critical_survival_target > 0.)
        critical_weighted_deficit = self.count_weighted_buffer_deficit(
            game_state,
            critical_survival_target,
            self.critical_survival_deficit_clip,
        )
        critical_survival_deficit = (
            -critical_weighted_deficit
            / max(critical_survival_target, 1e-6)
            * critical_survival_active
        )
        critical_low_buffer_expansion = -np.maximum(city_diff, 0.) * (
            critical_weighted_deficit > 0.
        ).astype(float) * critical_survival_active
        buffer_drop_with_growth = -np.maximum(
            self.total_fuel_turns - new_total_fuel_turns - self.buffer_growth_drop_threshold,
            0.,
        ) / max(self.buffer_growth_drop_clip, 1e-6) * np.maximum(city_diff, 0.)

        reward_items_dict = {
            "city_gain": np.maximum(city_diff, 0.),
            "city_loss": -np.maximum(-city_diff, 0.),
            "unit": new_unit_count - self.unit_count,
            "research": np.maximum(research_diff, 0.) * late_research_scale,
            "small_map_research": (
                np.maximum(research_diff, 0.) * small_map * below_uranium * late_research_scale
            ),
            "small_map_coal_deadline": (
                -coal_shortfall * small_map * crossed_coal_deadline
            ),
            "small_map_uranium_deadline": (
                -uranium_shortfall * small_map * crossed_uranium_deadline
            ),
            "coal_milestone": ((new_research_points >= 50) & (self.research_points < 50)).astype(float),
            "uranium_milestone": ((new_research_points >= 200) & (self.research_points < 200)).astype(float),
            "fuel_gain": np.maximum(new_total_fuel - self.total_fuel, 0.),
            "buffer_gain": np.maximum(
                self.weighted_buffer_deficit - new_weighted_buffer_deficit,
                0.,
            ),
            "buffer_deficit": -new_weighted_buffer_deficit,
            "unsafe_expansion": -np.maximum(city_diff, 0.) * (
                new_weighted_buffer_deficit > 0.
            ).astype(float),
            "unsafe_new_city": -new_city_deficit,
            "midgame_unsafe_new_city": (
                -new_city_deficit * unsafe_new_city_midgame_weight
            ),
            "endgame_buffer_deficit": -endgame_buffer_deficit * endgame_active,
            "late_expansion": -np.maximum(city_diff, 0.) * endgame_active * (
                endgame_buffer_deficit > 0.
            ).astype(float),
            "midgame_buffer_health": (
                1. - full_night_deficit / max(self.fuel_deficit_clip, 1e-6)
            ) * midgame_weight * has_city,
            "strategy_min_buffer": strategy_min_buffer_health * has_city,
            "strategy_p25_buffer": strategy_p25_buffer_health * has_city,
            "risk_adjusted_city_gain": (
                np.maximum(city_diff, 0.) * risk_adjusted_size_health * risk_adjusted_size_active
            ),
            "risk_adjusted_unit_gain": (
                np.maximum(new_unit_count - self.unit_count, 0.) * risk_adjusted_size_health * risk_adjusted_size_active
            ),
            "scale_target_bonus": safe_scale_target_progress,
            "scale_target_shortfall": scale_target_shortfall,
            "safe_scale_target_bonus": safe_scale_target_progress,
            "unsafe_large_city": unsafe_large_city,
            "critical_survival_deficit": critical_survival_deficit,
            "critical_low_buffer_expansion": critical_low_buffer_expansion,
            "buffer_drop_with_growth": buffer_drop_with_growth,
            "survival_step": active,
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel
            self.total_fuel_turns = new_total_fuel_turns
            self.fuel_buffer = new_fuel_buffer
            self.weighted_buffer_deficit = new_weighted_buffer_deficit
            self.city_tiles = current_city_tiles
            self.previous_turn = game_state.turn
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [self.weight_rewards(reward_items_dict[key] * w) for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        return tuple(reward / self.reward_scale / max(self.positive_weight, self.negative_weight))

    def count_fuel_buffer(self, game_state: Game) -> np.ndarray:
        buffers = []
        for player in game_state.players:
            city_buffers = [
                city.fuel / max(city.light_upkeep, 1.)
                for city in player.cities.values()
            ]
            buffers.append(min(city_buffers) if city_buffers else 0.)
        return np.array(buffers)

    @staticmethod
    def count_total_fuel_turns(game_state: Game) -> np.ndarray:
        fuel_turns = []
        for player in game_state.players:
            total_fuel = sum(city.fuel for city in player.cities.values())
            total_upkeep = sum(city.light_upkeep for city in player.cities.values())
            fuel_turns.append(total_fuel / max(total_upkeep, 1.))
        return np.array(fuel_turns)

    @staticmethod
    def count_fuel_buffer_percentile(game_state: Game, q: float) -> np.ndarray:
        buffers = []
        for player in game_state.players:
            city_buffers = sorted(
                city.fuel / max(city.light_upkeep, 1.)
                for city in player.cities.values()
            )
            if not city_buffers:
                buffers.append(0.)
                continue
            idx = (len(city_buffers) - 1) * q
            lo = int(np.floor(idx))
            hi = int(np.ceil(idx))
            if lo == hi:
                buffers.append(city_buffers[lo])
            else:
                buffers.append(city_buffers[lo] * (hi - idx) + city_buffers[hi] * (idx - lo))
        return np.array(buffers)

    @staticmethod
    def count_buffer_health(buffer: np.ndarray, target: float) -> np.ndarray:
        if target <= 0.:
            return np.zeros_like(buffer)
        return 1. - np.exp(-np.maximum(buffer, 0.) / max(target, 1e-6))

    def count_risk_adjusted_size_health(
            self,
            min_buffer: np.ndarray,
            min_target: float,
            p25_buffer: np.ndarray,
            p25_target: float,
    ) -> np.ndarray:
        if min_target <= 0. and p25_target <= 0.:
            return np.ones_like(min_buffer)
        min_health = self.count_buffer_health(min_buffer, min_target) if min_target > 0. else np.ones_like(min_buffer)
        p25_health = self.count_buffer_health(p25_buffer, p25_target) if p25_target > 0. else min_health
        blend = np.clip(self.risk_adjusted_size_buffer_blend, 0., 1.)
        return (1. - blend) * min_health + blend * p25_health

    @staticmethod
    def count_weighted_buffer_deficit(
            game_state: Game,
            target_turns: float,
            deficit_clip: float = float("inf"),
    ) -> np.ndarray:
        deficits = []
        for player in game_state.players:
            total_tiles = sum(len(city.citytiles) for city in player.cities.values())
            weighted_deficit = sum(
                max(target_turns - city.fuel / max(city.light_upkeep, 1.), 0.)
                * len(city.citytiles)
                for city in player.cities.values()
            )
            player_deficit = weighted_deficit / max(total_tiles, 1)
            deficits.append(min(player_deficit, deficit_clip))
        return np.array(deficits)

    @staticmethod
    def count_next_night_target_turns(turn: int) -> float:
        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        cycle_turn = turn % (day_length + night_length)
        if cycle_turn < day_length:
            return night_length * cycle_turn / day_length
        return night_length - (cycle_turn - day_length)

    @staticmethod
    def count_remaining_night_turns(turn: int) -> int:
        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        cycle_length = day_length + night_length
        max_turns = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        return sum(
            step % cycle_length >= day_length
            for step in range(turn, max_turns)
        )

    @staticmethod
    def count_new_city_target_turns(turn: int) -> int:
        day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        cycle_turn = turn % (day_length + night_length)
        if cycle_turn < day_length:
            return night_length
        return night_length - (cycle_turn - day_length)

    def count_strategy_min_buffer_target(self, turn: int) -> float:
        if turn < self.strategy_buffer_start_turn:
            return 0.
        if turn < self.strategy_buffer_mid_turn:
            return self.min_buffer_target_early
        return self.min_buffer_target_mid

    def count_strategy_p25_buffer_target(self, turn: int) -> float:
        if turn < self.strategy_buffer_mid_turn:
            return 0.
        if turn < self.strategy_buffer_late_turn:
            return self.p25_buffer_target_mid
        return self.p25_buffer_target_late

    def count_scale_target(self, game_state: Game) -> float:
        if game_state.turn < self.scale_target_start_turn:
            return 0.
        area = game_state.map_width * game_state.map_height
        if game_state.turn < self.scale_target_mid_turn:
            density = self.scale_target_early_density
        elif game_state.turn < self.scale_target_late_turn:
            density = self.scale_target_mid_density
        elif game_state.turn < self.endgame_start_turn:
            density = self.scale_target_late_density
        else:
            density = self.scale_target_end_density
        return max(1., area * density)

    def count_unsafe_large_city_pressure(self, game_state: Game) -> np.ndarray:
        if game_state.turn < self.unsafe_large_city_start_turn:
            return np.zeros((2,), dtype=float)
        if game_state.turn >= self.unsafe_large_city_late_turn:
            target_turns = self.unsafe_large_city_late_target_turns
        else:
            target_turns = self.unsafe_large_city_target_turns
        min_tiles = max(self.unsafe_large_city_min_tiles, 1)
        pressure = []
        for player in game_state.players:
            player_pressure = 0.
            for city in player.cities.values():
                tiles = len(city.citytiles)
                if tiles < min_tiles:
                    continue
                fuel_turns = city.fuel / max(city.light_upkeep, 1.)
                shortfall = max(target_turns - fuel_turns, 0.) / max(target_turns, 1e-6)
                size_factor = tiles / min_tiles
                player_pressure += shortfall * size_factor
            pressure.append(player_pressure)
        return np.array(pressure, dtype=float)

    def count_critical_survival_target(self, turn: int) -> float:
        if turn < self.critical_survival_start_turn:
            return 0.
        if turn >= self.critical_survival_late_turn:
            return self.critical_survival_late_target_turns
        return self.critical_survival_target_turns

    @staticmethod
    def smooth_window_weight(turn: int, start: int, end: int, ramp: int) -> float:
        if turn < start - ramp or turn > end + ramp:
            return 0.
        if turn < start:
            return (turn - (start - ramp)) / max(ramp, 1)
        if turn <= end:
            return 1.
        return ((end + ramp) - turn) / max(ramp, 1)

    def should_check_fuel_buffer(self, game_state: Game) -> float:
        cycle_len = (
            GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] +
            GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        )
        cycle_turn = game_state.turn % cycle_len
        day_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        if cycle_turn >= day_len:
            return 1.
        return float(day_len - cycle_turn <= self.pre_night_window)

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)
        self.total_fuel_turns = np.zeros_like(self.total_fuel_turns)
        self.fuel_buffer = np.zeros_like(self.fuel_buffer)
        self.weighted_buffer_deficit = np.zeros_like(self.weighted_buffer_deficit)
        self.city_tiles = None
        self.previous_turn = -1


class PunishingExponentialReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            **kwargs
    ):
        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 0.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.01,
            "fuel": 0.001,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(PunishingExponentialReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        city_diff = new_city_count - self.city_count
        unit_diff = new_unit_count - self.unit_count
        reward_items_dict = {
            "city": new_city_count,
            "unit": new_unit_count,
            "research": new_research_points,
            "fuel": new_total_fuel,
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [reward_items_dict[key] * w for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        lost_unit_or_city = (city_diff < 0) | (unit_diff < 0)
        reward = np.where(
            lost_unit_or_city,
            -0.1,
            reward / 1_000.
        )

        return tuple(reward), done or lost_unit_or_city.any()

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        raise NotImplementedError

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


# Subtask reward spaces defined below
# NB: Subtasks that are "different enough" should be defined separately since each subtask gets its own embedding
# See obs_spaces.SUBTASK_ENCODING

# TODO: Somehow include target locations for subtasks?
class Subtask(BaseRewardSpace, ABC):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        """
        Don't override reward_spec or you risk breaking classes like multi_subtask.MultiSubtask
        """
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=True
        )

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        goal_reached = self.completed_task(game_state)
        return tuple(goal_reached.astype(float)), goal_reached.any() or done

    @abstractmethod
    def completed_task(self, game_state: Game) -> np.ndarray:
        pass

    def get_subtask_encoding(self, subtask_encoding: dict) -> int:
        return subtask_encoding[type(self)]


class CollectNWood(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"], **kwargs):
        super(CollectNWood, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.wood for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNCoal(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 2, **kwargs):
        super(CollectNCoal, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.coal for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNUranium(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 5, **kwargs):
        super(CollectNUranium, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.uranium for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class MakeNCityTiles(Subtask):
    def __init__(self, n_city_tiles: int = 2, **kwargs):
        super(MakeNCityTiles, self).__init__(**kwargs)
        assert n_city_tiles > 1, "Players start with 1 city tile already"
        self.n_city_tiles = n_city_tiles

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_city_tiles(game_state) >= self.n_city_tiles


class MakeNContiguousCityTiles(MakeNCityTiles):
    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            # Extra -1 is included to avoid taking max of empty sequence
            max([len(city.citytiles) for city in player.cities.values()] + [0])
            for player in game_state.players
        ]) >= self.n_city_tiles


class CollectNTotalFuel(Subtask):
    def __init__(self, n_total_fuel: int = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] *
                                           GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"], **kwargs):
        super(CollectNTotalFuel, self).__init__(**kwargs)
        self.n_total_fuel = n_total_fuel

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_total_fuel(game_state) >= self.n_total_fuel


class SurviveNNights(Subtask):
    def __init__(self, n_nights: int = 1, **kwargs):
        super(SurviveNNights, self).__init__(**kwargs)
        cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        self.target_step = n_nights * cycle_len
        assert self.target_step <= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        self.city_count = np.empty((2,), dtype=int)
        self.unit_count = np.empty_like(self.city_count)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        failed_task = self.failed_task(game_state)
        completed_task = self.completed_task(game_state)
        if failed_task.any():
            rewards = np.where(
                failed_task,
                0.,
                0.5 + 0.5 * completed_task.astype(float)
            )
        else:
            rewards = completed_task.astype(float)
        done = failed_task.any() or completed_task.any() or done
        if done:
            self._reset()
        return tuple(rewards), done

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            game_state.turn >= self.target_step
        ]).repeat(2)

    def failed_task(self, game_state: Game) -> np.ndarray:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)

        failed = np.logical_or(
            new_city_count < self.city_count,
            new_unit_count < self.unit_count
        )
        self.city_count = new_city_count
        self.unit_count = new_unit_count
        return failed

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)


class GetNResearchPoints(Subtask):
    def __init__(
            self,
            n_research_points: int = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"],
            **kwargs
    ):
        super(GetNResearchPoints, self).__init__(**kwargs)
        self.n_research_points = n_research_points

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([player.research_points for player in game_state.players]) >= self.n_research_points
