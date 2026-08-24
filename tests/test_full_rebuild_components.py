from types import SimpleNamespace
import unittest

import torch
from torch import nn

from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX
from lux_ai.lux_gym.reward_spaces import LogScaleOutcomeReward
from lux_ai.rl_agent.learned_intervention_gate import SidecarLogitDeltaGate
from lux_ai.rl_agent.role_conditioned_local_adapter import RoleConditionedLocalAdapter
from lux_ai.rl_agent.sidecar_agent_wrapper import SidecarAgentWrapper
from lux_ai.rl_agent.trainable_role_bias import ROLE_BIAS_NAMES, TrainableRoleBiasLayer
from lux_ai.torchbeast.monobeast import (
    combine_policy_entropy,
    combine_policy_logits_to_log_probs,
    compute_appo_policy_loss,
    compute_teacher_bc_loss,
    compute_teacher_kl_loss,
    configure_role_joint_head_training,
)
from lux_ai.torchbeast.pfsp import LeagueOpponent, PFSPOpponentSampler


class FullRebuildComponentsTest(unittest.TestCase):
    def make_logits(self):
        result = {}
        for space, meanings in ACTION_MEANINGS.items():
            result[space] = torch.zeros(1, 4, 2, 8, 8, len(meanings))
        result["worker"][..., 0] = float("-inf")
        return result

    def test_map_phase_rules(self):
        self.assertFalse(SidecarAgentWrapper.map_phase_active(12, 119, 0))
        self.assertTrue(SidecarAgentWrapper.map_phase_active(12, 145, 0))
        self.assertFalse(SidecarAgentWrapper.map_phase_active(16, 150, 1))
        self.assertFalse(SidecarAgentWrapper.map_phase_active(24, 79, 0))
        self.assertTrue(SidecarAgentWrapper.map_phase_active(24, 79, 1))
        self.assertTrue(SidecarAgentWrapper.map_phase_active(32, 0, 0))

    def test_safe_expansion_whitelist_and_soft_bias(self):
        gate = SidecarLogitDeltaGate()
        logits = self.make_logits()
        risk = torch.full((1, 2, 8, 8), 4.0)
        unsafe = torch.full_like(risk, -4.0)
        final, _ = gate(logits, risk, unsafe, risk_threshold=0.85, logit_bias_lambda=4.0)
        build_city = ACTION_MEANINGS_TO_IDX["worker"]["BUILD_CITY"]
        self.assertTrue((final["worker"][..., build_city] < 0).all())
        self.assertTrue(torch.equal(torch.isfinite(logits["worker"]), torch.isfinite(final["worker"])))

        safe = torch.full_like(risk, 4.0)
        whitelisted, _ = gate(logits, risk, safe, risk_threshold=0.85)
        self.assertTrue(torch.equal(logits["worker"], whitelisted["worker"]))

    def test_role_local_adapter_step0_is_exact_and_preserves_mask(self):
        adapter = RoleConditionedLocalAdapter(8, hidden_channels=4, max_delta=0.25)
        features = torch.randn(2, 8, 4, 4)
        logits = {
            "worker": torch.randn(1, 1, 2, 4, 4, len(ACTION_MEANINGS["worker"]))
        }
        logits["worker"][..., 0] = float("-inf")
        codes = {"worker": torch.ones_like(logits["worker"], dtype=torch.int8)}
        output, deltas = adapter(features, logits, codes)

        self.assertTrue(adapter.zero_projection_is_exact())
        self.assertTrue(torch.equal(output["worker"], logits["worker"]))
        self.assertEqual(torch.count_nonzero(deltas["worker"]).item(), 0)
        self.assertTrue(torch.equal(torch.isfinite(output["worker"]), torch.isfinite(logits["worker"])))

    def test_role_local_adapter_gradients_do_not_reach_actor_features(self):
        adapter = RoleConditionedLocalAdapter(8, hidden_channels=4, max_delta=0.25)
        with torch.no_grad():
            adapter.output_heads["worker"].weight.fill_(0.1)
        features = torch.randn(2, 8, 4, 4, requires_grad=True)
        logits = {
            "worker": torch.zeros(1, 1, 2, 4, 4, len(ACTION_MEANINGS["worker"]))
        }
        codes = {"worker": torch.ones_like(logits["worker"], dtype=torch.int8)}
        output, _ = adapter(features, logits, codes)
        output["worker"].sum().backward()

        self.assertIsNone(features.grad)
        self.assertIsNotNone(adapter.output_heads["worker"].weight.grad)

    def test_role_joint_head_training_keeps_backbone_frozen(self):
        class BaseAgent(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = nn.Linear(2, 2)
                self.actor_base = nn.Linear(2, 2)
                self.actor = nn.Linear(2, 2)
                self.baseline_base = nn.Linear(2, 2)
                self.baseline = nn.Linear(2, 1)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_agent = BaseAgent()
                self.role_bias_layer = nn.Linear(1, 1)
                self.role_local_adapter = nn.Linear(2, 2)
                self.policy_heads_enabled = False

            def set_policy_head_training(self, enabled):
                self.policy_heads_enabled = enabled

        model = Wrapper()
        configure_role_joint_head_training(model, training=True)

        self.assertFalse(any(p.requires_grad for p in model.base_agent.base_model.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.base_agent.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.base_agent.baseline.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.role_local_adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.role_bias_layer.parameters()))
        self.assertTrue(model.policy_heads_enabled)

    def test_role_bias_scale(self):
        layer = TrainableRoleBiasLayer()
        logits = {"worker": torch.zeros(1, 1, 2, 1, 1, len(ACTION_MEANINGS["worker"]))}
        code = ROLE_BIAS_NAMES.index("builder_build_city_bias") + 1
        codes = {"worker": torch.zeros_like(logits["worker"], dtype=torch.int8)}
        build_city = ACTION_MEANINGS_TO_IDX["worker"]["BUILD_CITY"]
        codes["worker"][..., build_city] = code
        output = layer(logits, codes, torch.tensor(0.25))
        expected = layer.bias_params["builder_build_city_bias"].detach() * 0.25
        self.assertTrue(torch.allclose(output["worker"][..., build_city], expected.expand_as(output["worker"][..., build_city])))

    def test_appo_clipping_and_reference_kl(self):
        target = torch.tensor([[[torch.log(torch.tensor(2.0))]]])
        behavior = torch.zeros_like(target)
        advantage = torch.ones_like(target)
        loss, clip_fraction, _ = compute_appo_policy_loss(target, behavior, advantage, 0.2, "mean")
        self.assertAlmostEqual(loss.item(), -1.2, places=6)
        self.assertEqual(clip_fraction.item(), 1.0)
        logits = torch.tensor([[[[[[1.0, -1.0]]]]]])
        mask = torch.ones(logits.shape[:-1], dtype=torch.bool)
        self.assertAlmostEqual(compute_teacher_kl_loss(logits, logits, mask).item(), 0.0, places=6)

    def test_appo_excludes_nonfinite_log_probability_pairs(self):
        target = torch.tensor([[[0.0, float("-inf")]]], requires_grad=True)
        behavior = torch.tensor([[[0.0, float("-inf")]]])
        advantage = torch.ones_like(target)
        valid = torch.tensor([[[True, False]]])

        loss, clip_fraction, approximate_kl = compute_appo_policy_loss(
            target, behavior, advantage, 0.2, "mean", valid_mask=valid
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(clip_fraction))
        self.assertTrue(torch.isfinite(approximate_kl))
        self.assertTrue(torch.isfinite(target.grad).all())

    def test_all_masked_policy_rows_have_finite_gradients(self):
        shape = (2, 1, 1, 2, 2, 2, 3)
        source = torch.randn(shape, requires_grad=True)
        legal_rows = torch.zeros(shape[:-1], dtype=torch.bool)
        legal_rows[..., 0, 0] = True
        logits = source.masked_fill(~legal_rows.unsqueeze(-1), float("-inf"))
        actions = torch.zeros((*shape[:-1], 1), dtype=torch.long)
        actions_taken = torch.zeros((*shape[:-1], 1), dtype=torch.bool)
        actions_taken[..., 0, 0, 0] = True
        any_actions_taken = actions_taken.any(dim=-1)

        loss = combine_policy_logits_to_log_probs(logits, actions, actions_taken).sum()
        loss = loss + combine_policy_entropy(logits, any_actions_taken).sum()
        loss = loss + compute_teacher_bc_loss(
            logits,
            logits.detach(),
            any_actions_taken,
        )[0].sum()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_pfsp_prioritizes_hard_and_night_loss(self):
        easy = LeagueOpponent("easy", "easy.py", games=20, wins=18)
        hard = LeagueOpponent("hard", "hard.py", games=20, wins=4, night_loss_mean=2.0)
        weights = PFSPOpponentSampler([easy, hard]).weights()
        self.assertGreater(weights[1], weights[0])

    def test_log_reward_is_finite_at_zero(self):
        reward = LogScaleOutcomeReward()
        players = [SimpleNamespace(city_tile_count=0, units=[]), SimpleNamespace(city_tile_count=0, units=[])]
        game = SimpleNamespace(players=players)
        values = reward.compute_rewards(game, False)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in values))


if __name__ == "__main__":
    unittest.main()
