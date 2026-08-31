import math
import unittest

import torch

from scripts.build_critical_state_preferences import contains_build_city, first_persistent
from scripts.train_critical_role_preferences import preference_terms


def outputs(logits):
    return {"policy_logits": {"worker": logits}}


class CriticalRolePreferenceTest(unittest.TestCase):
    def test_first_persistent_requires_full_window(self):
        self.assertEqual(first_persistent([False, True, True, False, True, True, True]), 4)
        self.assertEqual(first_persistent([True, True], width=3), -1)

    def test_build_city_rejected_detection(self):
        self.assertTrue(contains_build_city(["m u_1 n", "bcity u_2"]))
        self.assertFalse(contains_build_city(["m u_1 n", "t u_2 u_3 wood 10"]))

    def test_zero_relative_margin_is_log_two_and_weak_is_quarter_weight(self):
        logits = torch.tensor([[[[[2.0, 1.0, 0.0]]]]])
        preferred = torch.tensor([[[[[True, False, False]]]]])
        rejected = torch.tensor([[[[[False, True, False]]]]])
        batch = {
            "weights": torch.tensor([1.0]),
            "sample_type": torch.tensor([1]),
            "preferred": {"worker": preferred},
            "rejected": {"worker": rejected},
            "inputs": {"info": {"role_bias_codes": {"worker": torch.ones_like(preferred)}}},
        }
        terms = preference_terms(outputs(logits), outputs(logits.clone()), batch, beta=2.0, focal_gamma=2.0)
        self.assertAlmostEqual(float(terms["dpo_loss"]), math.log(2.0), places=6)
        self.assertAlmostEqual(float(terms["dpo_weight"]), 0.25, places=6)
        self.assertEqual(float(terms["strict_count"]), 0.0)
        self.assertEqual(float(terms["weak_count"]), 1.0)

    def test_inactive_role_has_no_preference_loss(self):
        logits = torch.zeros((1, 1, 1, 1, 3))
        preferred = torch.tensor([[[[[True, False, False]]]]])
        rejected = torch.tensor([[[[[False, True, False]]]]])
        batch = {
            "weights": torch.tensor([1.0]),
            "sample_type": torch.tensor([0]),
            "preferred": {"worker": preferred},
            "rejected": {"worker": rejected},
            "inputs": {"info": {"role_bias_codes": {"worker": torch.zeros_like(preferred)}}},
        }
        terms = preference_terms(outputs(logits), outputs(logits), batch, beta=2.0, focal_gamma=2.0)
        self.assertEqual(float(terms["dpo_loss"]), 0.0)
        self.assertEqual(float(terms["paired_count"]), 0.0)


if __name__ == "__main__":
    unittest.main()
