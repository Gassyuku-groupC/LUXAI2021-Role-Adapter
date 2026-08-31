from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX, MAX_OVERLAPPING_ACTIONS


class LearnedInterventionGate(nn.Module):
    """Spatial gate that moves probability away from narrowly risky actions."""

    GATED_ACTIONS = {
        "city_tile": "BUILD_WORKER",
        "worker": "BUILD_CITY",
    }
    ALTERNATIVE_ACTIONS = {
        "city_tile": ("NO-OP", "RESEARCH"),
        "worker": None,
    }

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 32,
            initial_probability: float = 1e-3,
            hard_threshold: float = 0.80,
            temperature: float = 1.0,
    ):
        super().__init__()
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be in (0, 1).")
        if not 0.0 < hard_threshold < 1.0:
            raise ValueError("hard_threshold must be in (0, 1).")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        self.hard_threshold = float(hard_threshold)
        self.temperature = float(temperature)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gate_heads = nn.ModuleDict({
            action_space: nn.Conv2d(hidden_channels, 1, kernel_size=1)
            for action_space in self.GATED_ACTIONS
        })
        initial_bias = math.log(initial_probability / (1.0 - initial_probability))
        for head in self.gate_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, initial_bias)

    def gate_logits(self, actor_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared(actor_features)
        batch_players, _, height, width = features.shape
        if batch_players % 2 != 0:
            raise ValueError(f"Expected a two-player feature batch, got {batch_players}.")
        batch_size = batch_players // 2
        return {
            action_space: head(features).view(batch_size, 1, 2, height, width)
            for action_space, head in self.gate_heads.items()
        }

    def _mix_one_policy(
            self,
            base_logits: torch.Tensor,
            gate_logits: torch.Tensor,
            risky_action_index: int,
            allowed_alternative_indices: Tuple[int, ...] | None,
            hard: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        legal = torch.isfinite(base_logits)
        base_probs = F.softmax(base_logits, dim=-1)

        if allowed_alternative_indices is None:
            alternative_mask = legal.clone()
        else:
            alternative_mask = torch.zeros_like(legal)
            alternative_mask[..., list(allowed_alternative_indices)] = legal[
                ..., list(allowed_alternative_indices)
            ]
        alternative_mask[..., risky_action_index] = False
        alternative_probs = base_probs * alternative_mask.to(base_probs.dtype)
        alternative_total = alternative_probs.sum(dim=-1, keepdim=True)
        has_alternative = alternative_total > 1e-12
        alternative_probs = alternative_probs / alternative_total.clamp(min=1e-12)

        gate_probability = torch.sigmoid(gate_logits / self.temperature).unsqueeze(-1)
        risky_is_legal = legal[..., risky_action_index:risky_action_index + 1]
        gate_probability = gate_probability * risky_is_legal * has_alternative
        if hard:
            gate_probability = (gate_probability >= self.hard_threshold).to(base_probs.dtype)

        mixed_probs = (1.0 - gate_probability) * base_probs + gate_probability * alternative_probs
        mixed_logits = torch.where(
            legal,
            mixed_probs.clamp(min=1e-12).log(),
            torch.full_like(mixed_probs, float("-inf")),
        )
        return mixed_logits, gate_probability.squeeze(-1)

    def forward(
            self,
            actor_features: torch.Tensor,
            base_policy_logits: Dict[str, torch.Tensor],
            sample: bool,
            actions_per_square: int = MAX_OVERLAPPING_ACTIONS,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        logits_by_space = self.gate_logits(actor_features)
        policy_logits = dict(base_policy_logits)
        gate_probabilities: Dict[str, torch.Tensor] = {}
        for action_space, action_name in self.GATED_ACTIONS.items():
            if action_space not in policy_logits:
                continue
            action_index = ACTION_MEANINGS_TO_IDX[action_space][action_name]
            alternative_names = self.ALTERNATIVE_ACTIONS[action_space]
            alternative_indices = (
                None if alternative_names is None else tuple(
                    ACTION_MEANINGS_TO_IDX[action_space][name] for name in alternative_names
                )
            )
            policy_logits[action_space], gate_probabilities[action_space] = self._mix_one_policy(
                policy_logits[action_space],
                logits_by_space[action_space],
                action_index,
                alternative_indices,
                hard=not sample,
            )

        from ..nns.models import DictActor
        actions = {}
        for action_space, logits in policy_logits.items():
            n_actions = logits.shape[-1]
            flat_actions = DictActor.logits_to_actions(
                logits.view(-1, n_actions),
                sample=sample,
                actions_per_square=actions_per_square,
            )
            actions[action_space] = flat_actions.view(*logits.shape[:-1], -1)
        return policy_logits, actions, gate_probabilities


class SidecarLogitDeltaGate(nn.Module):
    """Zero-initialized additive gate driven by base logits and a risk map.

    The gate only changes the configured risky action logit. Illegal actions
    remain illegal because the original finite-logit mask is reapplied after
    adding the delta.
    """

    GATED_ACTIONS = {
        "city_tile": "BUILD_WORKER",
        "worker": "BUILD_CITY",
    }

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.gate_trunks = nn.ModuleDict()
        self.delta_projections = nn.ModuleDict()
        for action_space in self.GATED_ACTIONS:
            self.gate_trunks[action_space] = nn.Sequential(
                nn.Conv2d(2, int(hidden_channels), kernel_size=3, padding=1),
                nn.ReLU(),
            )
            projection = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.delta_projections[action_space] = projection

    def forward(
            self,
            base_policy_logits: Dict[str, torch.Tensor],
            risk_logits: torch.Tensor,
            safe_expansion_logits: torch.Tensor | None = None,
            risk_threshold: float | torch.Tensor | None = None,
            safe_expansion_threshold: float = 0.80,
            logit_bias_lambda: float = 4.0,
            active_mask: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if risk_logits.dim() != 4 or risk_logits.shape[1] != 2:
            raise ValueError(f"Expected B,2,H,W risk logits, got {tuple(risk_logits.shape)}")
        risk_probability = torch.sigmoid(risk_logits.detach())
        safe_probability = (
            torch.sigmoid(safe_expansion_logits.detach())
            if safe_expansion_logits is not None
            else torch.zeros_like(risk_probability)
        )
        final_logits = dict(base_policy_logits)
        deltas: Dict[str, torch.Tensor] = {}

        for action_space, action_name in self.GATED_ACTIONS.items():
            if action_space not in base_policy_logits:
                continue
            base_logits = base_policy_logits[action_space]
            action_index = ACTION_MEANINGS_TO_IDX[action_space][action_name]
            safe_logits = torch.nan_to_num(base_logits, nan=-30.0, neginf=-30.0, posinf=30.0)
            risky_probability = F.softmax(safe_logits, dim=-1)[..., action_index]
            # Average overlapping action planes, retaining B,2,H,W topology.
            while risky_probability.dim() > 4:
                risky_probability = risky_probability.mean(dim=1)
            gate_input = torch.stack((risk_probability, risky_probability.detach()), dim=2)
            batch_size, players, channels, height, width = gate_input.shape
            gate_input = gate_input.reshape(batch_size * players, channels, height, width)
            hidden = self.gate_trunks[action_space](gate_input)
            scalar_delta = self.delta_projections[action_space](hidden).view(
                batch_size,
                1,
                players,
                height,
                width,
                1,
            )
            if risk_threshold is not None and action_space == "worker":
                threshold = torch.as_tensor(
                    risk_threshold,
                    dtype=risk_probability.dtype,
                    device=risk_probability.device,
                )
                calibrated_active = risk_probability >= threshold
                calibrated_active &= safe_probability <= float(safe_expansion_threshold)
                if active_mask is not None:
                    calibrated_active &= active_mask.to(device=calibrated_active.device, dtype=torch.bool)
                calibrated_bias = -float(logit_bias_lambda) * risk_probability
                calibrated_bias = torch.where(
                    calibrated_active,
                    calibrated_bias,
                    torch.zeros_like(calibrated_bias),
                )
                scalar_delta = scalar_delta + calibrated_bias.unsqueeze(1).unsqueeze(-1)
            action_selector = torch.zeros(
                (1, 1, 1, 1, 1, base_logits.shape[-1]),
                dtype=base_logits.dtype,
                device=base_logits.device,
            )
            action_selector[..., action_index] = 1.0
            delta = scalar_delta * action_selector
            if delta.shape != base_logits.shape:
                delta = delta.expand_as(base_logits)
            legal = torch.isfinite(base_logits)
            adjusted = torch.where(legal, base_logits + delta, base_logits)
            final_logits[action_space] = adjusted
            deltas[action_space] = delta

        return final_logits, deltas

    def zero_projection_is_exact(self) -> bool:
        return all(
            torch.count_nonzero(projection.weight).item() == 0
            and torch.count_nonzero(projection.bias).item() == 0
            for projection in self.delta_projections.values()
        )
