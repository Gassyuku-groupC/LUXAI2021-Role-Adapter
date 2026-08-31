from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lux_ai.lux.constants import Constants
from lux_ai.lux.game_map import GameMap
from lux_ai.lux.game_objects import City, Player, Unit
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX
from lux_ai.rl_agent.role_assignment import (
    ATTACKER,
    BUILDER,
    FIREFIGHTER,
    HARVESTER,
    RoleAssignmentConfig,
    RoleCityBiasParams,
    SACRIFICIAL_DECAY,
)
from lux_ai.rl_agent.role_city_adapter import ROLE_BIAS_PRIORITY, RoleCityAdapter, pos_to_loc
from lux_ai.rl_agent.trainable_role_bias import ROLE_BIAS_INDEX, TrainableRoleBiasLayer


class RoleCityAdapterTests(unittest.TestCase):
    def make_game(self, width=12, height=12, turn=0):
        return SimpleNamespace(
            map=GameMap(width, height),
            map_width=width,
            map_height=height,
            turn=turn,
        )

    def add_city(self, player, city_id, x, y, fuel=1000, upkeep=10):
        city = City(player.team, city_id, fuel=fuel, light_upkeep=upkeep)
        city._add_city_tile(x, y, cooldown=0)
        player.cities[city_id] = city
        player.city_tile_count += 1
        return city

    def add_worker(self, player, unit_id, x, y, wood=0, coal=0, uranium=0):
        unit = Unit(
            player.team,
            Constants.UNIT_TYPES.WORKER,
            unit_id,
            x,
            y,
            0,
            wood,
            coal,
            uranium,
        )
        player.units.append(unit)
        return unit

    def make_policy(self):
        logits = {
            "worker": torch.zeros(1, 1, 2, 32, 32, len(ACTION_MEANINGS["worker"])),
            "city_tile": torch.zeros(1, 1, 2, 32, 32, len(ACTION_MEANINGS["city_tile"])),
        }
        masks = {key: torch.ones_like(value, dtype=torch.bool) for key, value in logits.items()}
        return logits, masks

    def test_disabled_adapter_is_exactly_transparent(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        worker = self.add_worker(player, "u0", 1, 1)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=False, bias_enabled=True)
        )
        adapter.update(game_state=game, player=player, opponent=opponent)
        logits, masks = self.make_policy()

        output = adapter.apply(
            game_state=game,
            player=player,
            opponent=opponent,
            actionable_workers={pos_to_loc(worker.pos.astuple()): [worker]},
            actionable_city_tiles={},
            policy_logits=logits,
            available_actions_mask=masks,
            player_id=0,
        )

        self.assertIs(output, logits)

    def test_bias_activation_respects_map_and_player_fallbacks(self):
        config = RoleAssignmentConfig(
            enabled=True,
            bias_enabled=True,
            bias_disabled_map_sizes=(),
            bias_disabled_players_by_map={},
            bias_scale_by_map_size={12: 0.35, 16: 1.0, 24: 0.35, 32: 0.25},
            max_biased_workers_by_map_size={12: 32, 16: 64, 24: 64, 32: 64},
            safety_only_map_sizes=(12, 24, 32),
        )

        self.assertTrue(config.bias_active_for(12, 0))
        self.assertTrue(config.bias_active_for(16, 0))
        self.assertTrue(config.bias_active_for(24, 0))
        self.assertTrue(config.bias_active_for(24, 1))
        self.assertTrue(config.bias_active_for(32, 1))
        self.assertEqual(config.bias_scale_for(24), 0.35)
        self.assertEqual(config.max_biased_workers_for(32), 64)
        self.assertTrue(config.safety_only_for(24))
        self.assertFalse(config.safety_only_for(16))

    def test_attacker_is_last_role_bias_priority(self):
        self.assertEqual(
            sorted(ROLE_BIAS_PRIORITY, key=ROLE_BIAS_PRIORITY.get),
            [FIREFIGHTER, BUILDER, HARVESTER, ATTACKER],
        )

    def test_build_city_protection_round_trips_from_config(self):
        config = RoleAssignmentConfig.from_mapping({"preserve_build_city_logit": True})
        self.assertTrue(config.preserve_build_city_logit)
        self.assertTrue(config.to_mapping()["preserve_build_city_logit"])

    def test_cooldown_uses_compact_arrays_and_skips_reassignment(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        self.add_worker(player, "u0", 1, 1)
        adapter = RoleCityAdapter.from_config(RoleAssignmentConfig(enabled=True))
        first = adapter.update(game_state=game, player=player, opponent=opponent)

        game.turn = 11
        self.add_worker(opponent, "e0", 2, 1)
        second = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertIsInstance(adapter.state.role_codes, np.ndarray)
        self.assertEqual(second.unit_roles["u0"].role, first.unit_roles["u0"].role)
        self.assertEqual(second.unit_roles["u0"].reason, "cooldown_reuse")

    def test_sacrificial_city_never_triggers_at_turn_zero(self):
        game = self.make_game(turn=0)
        player, opponent = Player(0), Player(1)
        self.add_city(player, "c0", 1, 1, fuel=0, upkeep=10)
        adapter = RoleCityAdapter.from_config(RoleAssignmentConfig(enabled=True))

        snapshot = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertNotEqual(snapshot.city_roles["c0"].role, SACRIFICIAL_DECAY)
        self.assertFalse(snapshot.city_roles["c0"].abandon)

    def test_sacrificial_city_requires_confirmation_and_obeys_cap(self):
        game = self.make_game(turn=30)
        player, opponent = Player(0), Player(1)
        for index in range(12):
            self.add_city(player, f"c{index}", index, 0, fuel=0, upkeep=10)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=True, abandon_confirmation_turns=3)
        )

        first = adapter.update(game_state=game, player=player, opponent=opponent)
        game.turn = 31
        second = adapter.update(game_state=game, player=player, opponent=opponent)
        game.turn = 32
        third = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertEqual(len(first.abandoned_city_ids), 0)
        self.assertEqual(len(second.abandoned_city_ids), 0)
        self.assertEqual(len(third.abandoned_city_ids), 1)

    def test_nearby_fuel_prevents_sacrificial_assignment(self):
        game = self.make_game(turn=30)
        player, opponent = Player(0), Player(1)
        for index in range(3):
            self.add_city(player, f"c{index}", index, 0, fuel=0, upkeep=10)
        game.map._setResource("wood", 0, 1, 500)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=True, abandon_confirmation_turns=1)
        )

        snapshot = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertNotEqual(snapshot.city_roles["c0"].role, SACRIFICIAL_DECAY)

    def test_city_assignment_never_emits_removed_fuel_station_role(self):
        game = self.make_game(turn=30)
        player, opponent = Player(0), Player(1)
        for index in range(4):
            self.add_city(player, f"c{index}", index * 2, 0, fuel=1000, upkeep=10)
        game.map._setResource("wood", 0, 1, 500)
        adapter = RoleCityAdapter.from_config(RoleAssignmentConfig(enabled=True))

        snapshot = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertNotIn("FuelStation", {state.role for state in snapshot.city_roles.values()})

    def test_update_budget_reuses_previous_snapshot_for_one_turn(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        self.add_worker(player, "u0", 1, 1)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=True, update_time_budget_seconds=0.0)
        )
        first = adapter.update(game_state=game, player=player, opponent=opponent)
        game.turn = 11
        second = adapter.update(game_state=game, player=player, opponent=opponent)

        self.assertIs(second, first)
        self.assertTrue(adapter.last_update_degraded)

    def test_learnable_bias_receives_gradient(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        city = self.add_city(player, "c0", 1, 0)
        worker = self.add_worker(player, "u0", 1, 1)
        self.add_worker(opponent, "e0", 2, 1)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=True, bias_enabled=True, learnable_biases=True)
        )
        adapter.update(game_state=game, player=player, opponent=opponent)
        logits, masks = self.make_policy()

        output = adapter.apply(
            game_state=game,
            player=player,
            opponent=opponent,
            actionable_workers={pos_to_loc(worker.pos.astuple()): [worker]},
            actionable_city_tiles={pos_to_loc(city.citytiles[0].pos.astuple()): city.citytiles[0]},
            policy_logits=logits,
            available_actions_mask=masks,
            player_id=0,
        )
        move_e = ACTION_MEANINGS_TO_IDX["worker"]["MOVE_e"]
        output["worker"][0, 0, 0, 1, 1, move_e].backward()

        gradient = adapter.bias_params["attacker_block_move_bias"].grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.item(), 0.0)

    def test_bias_never_changes_illegal_logits(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        worker = self.add_worker(player, "u0", 1, 1)
        self.add_worker(opponent, "e0", 2, 1)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(enabled=True, bias_enabled=True)
        )
        adapter.update(game_state=game, player=player, opponent=opponent)
        logits, masks = self.make_policy()
        worker_mask = masks["worker"]
        worker_mask[0, 0, 0, 1, 1, :] = False
        move_e = ACTION_MEANINGS_TO_IDX["worker"]["MOVE_e"]
        worker_mask[0, 0, 0, 1, 1, move_e] = True
        masked_logits = {
            key: value.masked_fill(~masks[key], float("-inf"))
            for key, value in logits.items()
        }

        output = adapter.apply(
            game_state=game,
            player=player,
            opponent=opponent,
            actionable_workers={pos_to_loc(worker.pos.astuple()): [worker]},
            actionable_city_tiles={},
            policy_logits=masked_logits,
            available_actions_mask=masks,
            player_id=0,
        )

        illegal = ~worker_mask
        self.assertTrue(torch.isneginf(output["worker"][illegal]).all())

    def test_safety_only_map_does_not_bias_harvester(self):
        game = self.make_game(turn=10)
        player, opponent = Player(0), Player(1)
        worker = self.add_worker(player, "u0", 1, 1)
        game.map._setResource("wood", 1, 1, 500)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(
                enabled=True,
                bias_enabled=True,
                safety_only_map_sizes=(12,),
            )
        )
        adapter.update(game_state=game, player=player, opponent=opponent)
        logits, masks = self.make_policy()

        output = adapter.apply(
            game_state=game,
            player=player,
            opponent=opponent,
            actionable_workers={pos_to_loc(worker.pos.astuple()): [worker]},
            actionable_city_tiles={},
            policy_logits=logits,
            available_actions_mask=masks,
            player_id=0,
        )

        self.assertEqual(torch.count_nonzero(output["worker"]).item(), 0)

    def test_firefighter_transfer_only_biases_adjacent_relay_toward_city(self):
        game = self.make_game(turn=20)
        player, opponent = Player(0), Player(1)
        city = self.add_city(player, "critical", 0, 0, fuel=5, upkeep=10)
        game.map._setResource("wood", 0, 1, 500)
        carrier = self.add_worker(player, "u0", 3, 0, wood=50)
        relay = self.add_worker(player, "u1", 2, 0)
        self.add_worker(player, "u2", 4, 0)
        adapter = RoleCityAdapter.from_config(
            RoleAssignmentConfig(
                enabled=True,
                bias_enabled=True,
                bias_params=RoleCityBiasParams(firefighter_move_bias=0.0),
            )
        )
        adapter.update(game_state=game, player=player, opponent=opponent)
        logits, masks = self.make_policy()

        output = adapter.apply(
            game_state=game,
            player=player,
            opponent=opponent,
            actionable_workers={pos_to_loc(carrier.pos.astuple()): [carrier]},
            actionable_city_tiles={},
            policy_logits=logits,
            available_actions_mask=masks,
            player_id=0,
        )
        transfer_w = ACTION_MEANINGS_TO_IDX["worker"]["TRANSFER_wood_w"]
        transfer_e = ACTION_MEANINGS_TO_IDX["worker"]["TRANSFER_wood_e"]

        self.assertLess(relay.pos.distance_to(city.citytiles[0].pos), carrier.pos.distance_to(city.citytiles[0].pos))
        self.assertGreater(output["worker"][0, 0, 0, 3, 0, transfer_w].item(), 0.0)
        self.assertEqual(output["worker"][0, 0, 0, 3, 0, transfer_e].item(), 0.0)

    def test_trainable_role_layer_updates_only_encoded_logits(self):
        layer = TrainableRoleBiasLayer()
        logits, _ = self.make_policy()
        codes = {key: torch.zeros_like(value, dtype=torch.int8) for key, value in logits.items()}
        build = ACTION_MEANINGS_TO_IDX["worker"]["BUILD_CITY"]
        code = ROLE_BIAS_INDEX["builder_build_city_bias"] + 1
        codes["worker"][0, 0, 0, 1, 1, build] = code
        codes["worker"][0, 0, 0, 2, 2, build] = -code

        output = layer(logits, codes)
        self.assertEqual(len(layer.bias_params), 14)
        self.assertAlmostEqual(output["worker"][0, 0, 0, 1, 1, build].item(), 1.0)
        self.assertAlmostEqual(output["worker"][0, 0, 0, 2, 2, build].item(), -1.0)
        self.assertEqual(torch.count_nonzero(output["city_tile"]).item(), 0)

        loss = output["worker"][0, 0, 0, 1, 1, build]
        loss.backward()
        self.assertAlmostEqual(layer.bias_params["builder_build_city_bias"].grad.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
