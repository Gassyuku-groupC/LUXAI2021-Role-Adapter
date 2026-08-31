from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..nns import models


class LogitResidualHead(nn.Module):
    """Small residual policy head operating on frozen policy logits.

    The module is intentionally tiny: for each action space it applies a shared
    linear map over the action dimension at every board position. All weights are
    zero-initialized so an untrained residual exactly preserves the base policy.
    """

    def __init__(self, action_sizes: Dict[str, int], gamma: float = 0.15, max_delta: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.max_delta = float(max_delta)
        self.heads = nn.ModuleDict({
            key: nn.Linear(int(n_actions), int(n_actions), bias=True)
            for key, n_actions in action_sizes.items()
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def delta_logits(self, policy_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, logits in policy_logits.items():
            safe_logits = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
            out[key] = self.heads[key](safe_logits).clamp(-self.max_delta, self.max_delta)
        return out

    def forward(self, policy_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        deltas = self.delta_logits(policy_logits)
        out = {}
        for key, logits in policy_logits.items():
            finite_mask = torch.isfinite(logits)
            safe_logits = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
            final = safe_logits + self.gamma * deltas[key]
            out[key] = torch.where(finite_mask, final, logits)
        return out


def action_sizes_from_logits(policy_logits: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {key: int(value.shape[-1]) for key, value in policy_logits.items()}


class SpatialResidualHead(nn.Module):
    """Small zero-initialized residual head over actor feature maps.

    The residual keeps the full board topology: input is the frozen actor feature
    map of shape ``B*2, C, H, W`` and each action space receives a local conv
    delta with the same shape as the frozen policy logits.
    """

    def __init__(
        self,
        in_channels: int,
        action_sizes: Dict[str, int],
        action_plane_shapes: Dict[str, Tuple[int, ...]],
        gamma: float = 0.05,
        max_delta: float = 0.75,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        player_scales: Tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.action_sizes = {key: int(value) for key, value in action_sizes.items()}
        self.action_plane_shapes = {key: tuple(int(v) for v in value) for key, value in action_plane_shapes.items()}
        self.gamma = float(gamma)
        self.max_delta = float(max_delta)
        self.player_scales = (float(player_scales[0]), float(player_scales[1]))
        padding = kernel_size // 2
        self.trunk = nn.Sequential(
            nn.Conv2d(self.in_channels, int(hidden_channels), kernel_size, padding=padding, groups=1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), 1),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            key: nn.Conv2d(
                int(hidden_channels),
                int(self.action_sizes[key] * max(1, _prod(self.action_plane_shapes[key]))),
                1,
            )
            for key in self.action_sizes
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def delta_logits(self, actor_features: torch.Tensor, policy_logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = self.trunk(actor_features)
        out = {}
        b2, _, h, w = actor_features.shape
        if b2 % 2 != 0:
            raise ValueError(f"Expected actor feature batch to be B*2, got {b2}")
        batch = b2 // 2
        for key, logits in policy_logits.items():
            n_actions = self.action_sizes[key]
            plane_shape = self.action_plane_shapes[key]
            raw = self.heads[key](features)
            raw = raw.view(batch, 2, n_actions, *plane_shape, h, w)
            raw = raw.permute(0, 3, 1, 4, 5, 2).contiguous()
            if raw.shape != logits.shape:
                raise ValueError(f"Residual delta for {key} has shape {tuple(raw.shape)}, expected {tuple(logits.shape)}")
            out[key] = raw.clamp(-self.max_delta, self.max_delta)
        return out

    def forward(self, actor_features: torch.Tensor, policy_logits: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        deltas = self.delta_logits(actor_features, policy_logits)
        out = {}
        for key, logits in policy_logits.items():
            finite_mask = torch.isfinite(logits)
            safe = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
            player_scale = torch.tensor(self.player_scales, dtype=safe.dtype, device=safe.device).view(1, 1, 2, 1, 1, 1)
            final = safe + self.gamma * player_scale * deltas[key]
            out[key] = torch.where(finite_mask, final, logits)
        return out, deltas


def _prod(values: Tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def spatial_residual_forward(model: nn.Module, model_input: dict, residual: SpatialResidualHead, sample: bool = False) -> dict:
    """Run frozen best and spatial residual in one no-extra-backbone pass."""

    x, input_mask, available_actions_mask, subtask_embeddings = model.dict_input_layer(model_input)
    base_out, input_mask = model.base_model((x, input_mask))
    if subtask_embeddings is not None:
        subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)
    actor_features = model.actor_base(base_out)
    base_logits, _ = model.actor(
        actor_features,
        available_actions_mask=available_actions_mask,
        sample=sample,
    )
    final_logits, deltas = residual(actor_features, base_logits)
    actions = {
        key: models.DictActor.logits_to_actions(
            logits.view(-1, logits.shape[-1]),
            sample=sample,
            actions_per_square=None,
        ).view(*logits.shape[:-1], -1)
        for key, logits in final_logits.items()
    }
    baseline = model.baseline(model.baseline_base(base_out), input_mask, subtask_embeddings)
    return {
        "actions": actions,
        "policy_logits": final_logits,
        "base_policy_logits": base_logits,
        "delta_logits": deltas,
        "baseline": baseline,
    }
