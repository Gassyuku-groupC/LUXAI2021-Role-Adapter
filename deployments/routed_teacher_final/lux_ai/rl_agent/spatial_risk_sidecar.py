from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class SpatialRiskAttentionSidecar(nn.Module):
    """Detached pooled-KV attention branch for tile-level risk prediction.

    Lux observations are padded to the maximum 32x32 board. The branch crops
    each player feature map to its logical board using ``input_mask`` before
    attention, so Query length remains H*W while Key/Value length is fixed at
    ``pool_size ** 2``. Results are written back into the padded layout to stay
    aligned with the frozen actor logits.
    """

    def __init__(
            self,
            in_channels: int,
            attention_dim: int = 64,
            num_heads: int = 4,
            pool_size: int = 8,
            dropout: float = 0.0,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.in_channels = int(in_channels)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.pool_size = int(pool_size)

        self.input_projection = nn.Sequential(
            nn.Conv2d(self.in_channels, self.attention_dim, kernel_size=1),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(self.attention_dim)
        self.kv_norm = nn.LayerNorm(self.attention_dim)
        self.kv_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.risk_projection = nn.Conv2d(self.attention_dim, 1, kernel_size=1)
        self.safe_expansion_projection = nn.Conv2d(self.attention_dim, 1, kernel_size=1)

    def forward(
            self,
            actor_features: torch.Tensor,
            input_mask: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        if actor_features.dim() != 4:
            raise ValueError(f"Expected B*2,C,H,W actor features, got {tuple(actor_features.shape)}")
        if actor_features.shape[0] % 2 != 0:
            raise ValueError(f"Expected a two-player feature batch, got {actor_features.shape[0]}")

        # This detach is intentional even when the caller already uses no_grad.
        frozen_features = actor_features.detach()
        projected = self.input_projection(frozen_features)
        player_mask = self.normalize_player_mask(input_mask, projected)
        context = torch.zeros_like(projected)
        logical_shapes = []

        for player_index in range(projected.shape[0]):
            height, width = self.logical_board_shape(player_mask[player_index, 0])
            logical_shapes.append((height, width))
            feature_map = projected[player_index:player_index + 1, :, :height, :width]
            query = feature_map.flatten(2).transpose(1, 2)
            pooled = self.kv_pool(feature_map).flatten(2).transpose(1, 2)
            attention_out, _ = self.attention(
                self.query_norm(query),
                self.kv_norm(pooled),
                self.kv_norm(pooled),
                need_weights=False,
            )
            attended = attention_out.transpose(1, 2).reshape(
                1,
                self.attention_dim,
                height,
                width,
            )
            fused = self.spatial_fusion(feature_map + attended)
            context[player_index:player_index + 1, :, :height, :width] = fused

        context = context * player_mask.to(dtype=context.dtype)
        risk_logits_flat = self.risk_projection(context)
        safe_expansion_logits_flat = self.safe_expansion_projection(context)
        invalid = ~player_mask.to(dtype=torch.bool)
        risk_logits_flat = risk_logits_flat.masked_fill(invalid, -30.0)
        safe_expansion_logits_flat = safe_expansion_logits_flat.masked_fill(invalid, -30.0)
        batch_size = actor_features.shape[0] // 2
        risk_logits = risk_logits_flat.view(
            batch_size,
            2,
            actor_features.shape[-2],
            actor_features.shape[-1],
        )
        safe_expansion_logits = safe_expansion_logits_flat.view_as(risk_logits)
        return {
            "risk_logits": risk_logits,
            "risk_probabilities": torch.sigmoid(risk_logits),
            "safe_expansion_logits": safe_expansion_logits,
            "safe_expansion_probabilities": torch.sigmoid(safe_expansion_logits),
            "context_features": context,
            "player_mask": player_mask,
            "logical_shapes": torch.tensor(
                logical_shapes,
                dtype=torch.long,
                device=actor_features.device,
            ).view(batch_size, 2, 2),
        }

    @staticmethod
    def normalize_player_mask(
            input_mask: torch.Tensor | None,
            features: torch.Tensor,
    ) -> torch.Tensor:
        if input_mask is None:
            return torch.ones(
                (features.shape[0], 1, features.shape[-2], features.shape[-1]),
                dtype=torch.bool,
                device=features.device,
            )
        mask = input_mask
        if mask.dim() == 5 and mask.shape[2] == 2:
            mask = mask.squeeze(1).reshape(-1, 1, mask.shape[-2], mask.shape[-1])
        elif mask.dim() == 5 and mask.shape[1] == 2:
            mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
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
            raise ValueError(f"Unsupported input_mask shape: {tuple(mask.shape)}")
        if mask.shape[0] != features.shape[0] or mask.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"input_mask {tuple(mask.shape)} does not align with features {tuple(features.shape)}"
            )
        return mask.to(device=features.device, dtype=torch.bool)

    @staticmethod
    def logical_board_shape(mask: torch.Tensor) -> Tuple[int, int]:
        occupied = torch.nonzero(mask, as_tuple=False)
        if occupied.numel() == 0:
            raise ValueError("input_mask contains no valid board cells")
        height = int(occupied[:, 0].max().item()) + 1
        width = int(occupied[:, 1].max().item()) + 1
        return height, width
