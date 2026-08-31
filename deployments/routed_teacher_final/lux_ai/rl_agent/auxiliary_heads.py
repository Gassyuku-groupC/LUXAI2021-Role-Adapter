from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class AuxiliaryRiskHead(nn.Module):
    """Global risk prediction head over frozen actor feature maps.

    The head predicts player-level diagnostic labels, not actions. It is meant
    to teach and validate risk representations before any policy integration.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), 1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), 1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.loss10 = nn.Linear(int(hidden_channels), 1)
        self.loss20 = nn.Linear(int(hidden_channels), 1)
        self.unsafe_large_city = nn.Linear(int(hidden_channels), 1)

    def forward(self, actor_features: torch.Tensor, input_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        features = self.trunk(actor_features)
        pooled = self.masked_mean(features, input_mask)
        pooled = self.dropout(pooled)
        batch = pooled.shape[0] // 2
        return {
            "loss10_logit": self.loss10(pooled).view(batch, 2),
            "loss20_logit": self.loss20(pooled).view(batch, 2),
            "unsafe_large_city_logit": self.unsafe_large_city(pooled).view(batch, 2),
        }

    @staticmethod
    def masked_mean(features: torch.Tensor, input_mask: torch.Tensor | None) -> torch.Tensor:
        if input_mask is None:
            return features.flatten(start_dim=-2).mean(dim=-1)
        mask = input_mask
        if mask.dim() == 5 and mask.shape[2] == 2:
            mask = mask.squeeze(1).reshape(mask.shape[0] * 2, 1, mask.shape[-2], mask.shape[-1])
        elif mask.dim() == 5 and mask.shape[1] == 2:
            mask = mask.reshape(mask.shape[0] * 2, 1, mask.shape[-2], mask.shape[-1])
        elif mask.dim() == 4:
            if mask.shape[0] * 2 == features.shape[0]:
                mask = torch.repeat_interleave(mask, 2, dim=0)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            if mask.shape[0] * 2 == features.shape[0]:
                mask = torch.repeat_interleave(mask, 2, dim=0)
        else:
            return features.flatten(start_dim=-2).mean(dim=-1)
        if mask.shape[0] != features.shape[0]:
            return features.flatten(start_dim=-2).mean(dim=-1)
        if mask.shape[-2:] != features.shape[-2:]:
            return features.flatten(start_dim=-2).mean(dim=-1)
        mask = mask.to(dtype=features.dtype, device=features.device)
        weighted = features * mask
        denom = mask.flatten(start_dim=-2).sum(dim=-1).clamp(min=1.0)
        return weighted.flatten(start_dim=-2).sum(dim=-1) / denom


def auxiliary_feature_forward(model: nn.Module, model_input: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen actor features and input mask for auxiliary heads."""

    x, input_mask, _available_actions_mask, subtask_embeddings = model.dict_input_layer(model_input)
    del subtask_embeddings
    base_out, input_mask = model.base_model((x, input_mask))
    actor_features = model.actor_base(base_out)
    return actor_features, input_mask
