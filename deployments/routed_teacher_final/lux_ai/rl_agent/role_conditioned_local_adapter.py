from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..lux_gym.act_spaces import ACTION_MEANINGS, MAX_OVERLAPPING_ACTIONS
from ..nns.models import DictActor
from .trainable_role_bias import ROLE_BIAS_NAMES


class RoleConditionedLocalAdapter(nn.Module):
    """Bounded per-tile policy delta conditioned on role action codes."""

    def __init__(self, feature_channels: int, hidden_channels: int = 16, max_delta: float = 0.25):
        super().__init__()
        self.max_delta = float(max_delta)
        self.context = nn.Sequential(
            nn.Conv2d(int(feature_channels), int(hidden_channels), kernel_size=1),
            nn.GELU(),
        )
        self.role_embedding = nn.Embedding(
            len(ROLE_BIAS_NAMES) + 1,
            int(hidden_channels),
            padding_idx=0,
        )
        self.hidden_channels = int(hidden_channels)
        self.output_heads = nn.ModuleDict()
        for action_space in ACTION_MEANINGS:
            head = nn.Linear(self.hidden_channels, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.output_heads[action_space] = head

    def forward(
            self,
            actor_features: torch.Tensor,
            policy_logits: Dict[str, torch.Tensor],
            role_codes: Dict[str, torch.Tensor],
            role_scale: torch.Tensor | None = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if actor_features.dim() != 4 or actor_features.shape[0] % 2:
            raise ValueError(f"Expected B*2,C,H,W features, got {tuple(actor_features.shape)}")
        frozen_features = actor_features.detach()
        batch_size = frozen_features.shape[0] // 2
        context = self.context(frozen_features).view(
            batch_size,
            2,
            self.hidden_channels,
            frozen_features.shape[-2],
            frozen_features.shape[-1],
        ).permute(0, 1, 3, 4, 2)

        output = dict(policy_logits)
        deltas = {}
        for action_space, logits in policy_logits.items():
            codes = role_codes.get(action_space)
            if codes is None:
                continue
            codes = codes.to(device=logits.device)
            role_index = codes.abs().long().clamp(max=len(ROLE_BIAS_NAMES))
            role_sign = codes.sign().to(dtype=logits.dtype).unsqueeze(-1)
            embedded = self.role_embedding(role_index) * role_sign
            local = context.unsqueeze(1).unsqueeze(-2)
            conditioned = torch.tanh(embedded + local)
            raw_delta = self.output_heads[action_space](conditioned).squeeze(-1)
            delta = self.max_delta * torch.tanh(raw_delta)
            delta = torch.where(codes != 0, delta, torch.zeros_like(delta))
            if role_scale is not None:
                delta = delta * role_scale.to(device=delta.device, dtype=delta.dtype)
            legal = torch.isfinite(logits)
            output[action_space] = torch.where(legal, logits + delta, logits)
            deltas[action_space] = delta
        return output, deltas

    def zero_projection_is_exact(self) -> bool:
        return all(
            torch.count_nonzero(head.weight).item() == 0
            and torch.count_nonzero(head.bias).item() == 0
            for head in self.output_heads.values()
        )


class RoleLocalAgentWrapper(nn.Module):
    """Frozen Actor plus fixed Role bias and a local learnable residual."""

    def __init__(self, base_agent, role_bias_layer, local_adapter):
        super().__init__()
        self.base_agent = base_agent
        self.role_bias_layer = role_bias_layer
        self.role_local_adapter = local_adapter
        self.base_out_channels = base_agent.base_out_channels
        self.train_policy_heads = False
        self.train_backbone_tail_blocks = 0
        for parameter in self.base_agent.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_agent.eval()
        if self.train_policy_heads:
            for module in (
                    self.base_agent.actor_base,
                    self.base_agent.actor,
                    self.base_agent.baseline_base,
                    self.base_agent.baseline,
            ):
                module.train(mode)
        base_children = list(self.base_agent.base_model.children())
        if self.train_backbone_tail_blocks > 0:
            for module in base_children[-self.train_backbone_tail_blocks:]:
                module.train(mode)
        return self

    def set_policy_head_training(self, enabled: bool) -> None:
        self.train_policy_heads = bool(enabled)

    def set_backbone_tail_training(self, blocks: int) -> None:
        blocks = int(blocks)
        residual_count = max(0, len(list(self.base_agent.base_model.children())) - 1)
        if blocks < 0 or blocks > residual_count:
            raise ValueError(
                f"backbone tail blocks must be in [0, {residual_count}], got {blocks}"
            )
        self.train_backbone_tail_blocks = blocks

    def forward(self, model_input, sample=True, return_features=False, **actor_kwargs):
        with torch.no_grad():
            x, input_mask, available_actions_mask, subtask_embeddings = self.base_agent.dict_input_layer(model_input)
            if subtask_embeddings is not None:
                subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)

        base_modules = list(self.base_agent.base_model.children())
        tail_blocks = self.train_backbone_tail_blocks if torch.is_grad_enabled() else 0
        frozen_count = len(base_modules) - tail_blocks
        backbone_state = (x, input_mask)
        with torch.no_grad():
            for module in base_modules[:frozen_count]:
                backbone_state = module(backbone_state)
        for module in base_modules[frozen_count:]:
            backbone_state = module(backbone_state)
        base_features, input_mask = backbone_state

        def run_heads():
            actor_features = self.base_agent.actor_base(base_features)
            policy_logits, _ = self.base_agent.actor(
                actor_features,
                available_actions_mask=available_actions_mask,
                sample=False,
                **actor_kwargs,
            )
            baseline = self.base_agent.baseline(
                self.base_agent.baseline_base(base_features),
                input_mask,
                subtask_embeddings,
            )
            return actor_features, policy_logits, baseline

        if self.train_policy_heads and torch.is_grad_enabled():
            actor_features, policy_logits, baseline = run_heads()
        else:
            with torch.no_grad():
                actor_features, policy_logits, baseline = run_heads()

        info = model_input.get("info", {})
        role_codes = info.get("role_bias_codes")
        role_scale = info.get("role_bias_scale")
        local_deltas = {}
        if role_codes is not None:
            policy_logits = self.role_bias_layer(policy_logits, role_codes, role_scale)
            policy_logits, local_deltas = self.role_local_adapter(
                actor_features,
                policy_logits,
                role_codes,
                role_scale,
            )

        actions_per_square = actor_kwargs.get("actions_per_square", MAX_OVERLAPPING_ACTIONS)
        actions = {}
        for action_space, logits in policy_logits.items():
            flat = DictActor.logits_to_actions(
                logits.view(-1, logits.shape[-1]),
                sample=sample,
                actions_per_square=actions_per_square,
            )
            actions[action_space] = flat.view(*logits.shape[:-1], -1)
        result = {
            "actions": actions,
            "policy_logits": policy_logits,
            "baseline": baseline,
            "role_local_deltas": local_deltas,
        }
        if return_features:
            result.update({"base_features": base_features, "feature_input_mask": input_mask})
        return result

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)
