from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import torch
from torch import nn

from ..lux_gym.act_spaces import MAX_OVERLAPPING_ACTIONS
from ..nns.models import DictActor
from .learned_intervention_gate import SidecarLogitDeltaGate
from .spatial_risk_sidecar import SpatialRiskAttentionSidecar


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class SidecarAgentWrapper(nn.Module):
    """Compose an unchanged frozen actor-critic with external sidecar modules."""

    def __init__(
            self,
            base_agent: nn.Module,
            spatial_risk_sidecar: SpatialRiskAttentionSidecar,
            intervention_gate: SidecarLogitDeltaGate,
            risk_gate_enabled: bool = False,
            risk_thresholds: Dict[int, float] | None = None,
            safe_expansion_threshold: float = 0.80,
            logit_bias_lambda: float = 4.0,
            freeze_base_agent: bool = True,
            role_bias_layer: nn.Module | None = None,
    ):
        super().__init__()
        if getattr(base_agent, "learned_intervention_gate", None) is not None:
            raise ValueError("SidecarAgentWrapper requires an ungated reference base agent")
        self.base_agent = base_agent
        self.spatial_risk_sidecar = spatial_risk_sidecar
        self.intervention_gate = intervention_gate
        self.risk_gate_enabled = bool(risk_gate_enabled)
        self.risk_thresholds = {int(k): float(v) for k, v in (risk_thresholds or {}).items()}
        self.safe_expansion_threshold = float(safe_expansion_threshold)
        self.logit_bias_lambda = float(logit_bias_lambda)
        self.freeze_base = bool(freeze_base_agent)
        self.role_bias_layer = role_bias_layer
        self.base_out_channels = base_agent.base_out_channels
        if self.freeze_base:
            self.freeze_base_agent()
        self._base_state_hash = state_dict_sha256(self.base_agent.state_dict())

    def freeze_base_agent(self) -> None:
        self.base_agent.eval()
        for parameter in self.base_agent.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_agent.eval()
        else:
            self.base_agent.train(mode)
        return self

    def forward(
            self,
            model_input: Dict[str, Union[dict, torch.Tensor]],
            sample: bool = True,
            return_features: bool = False,
            return_sidecar_outputs: bool = False,
            **actor_kwargs,
    ) -> Dict[str, Any]:
        base_context = torch.no_grad() if self.freeze_base else nullcontext()
        with base_context:
            x, input_mask, available_actions_mask, subtask_embeddings = self.base_agent.dict_input_layer(model_input)
            base_features, input_mask = self.base_agent.base_model((x, input_mask))
            if subtask_embeddings is not None:
                subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)
            actor_features = self.base_agent.actor_base(base_features)
            base_policy_logits, _ = self.base_agent.actor(
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

        sidecar_output = self.spatial_risk_sidecar(actor_features.detach(), input_mask)
        risk_threshold = None
        active_mask = None
        if self.risk_gate_enabled:
            risk_threshold, active_mask = self._gate_tensors(
                model_input,
                sidecar_output["logical_shapes"],
                sidecar_output["risk_logits"],
            )
        final_policy_logits, logit_deltas = self.intervention_gate(
            base_policy_logits,
            sidecar_output["risk_logits"],
            safe_expansion_logits=sidecar_output["safe_expansion_logits"],
            risk_threshold=risk_threshold,
            safe_expansion_threshold=self.safe_expansion_threshold,
            logit_bias_lambda=self.logit_bias_lambda,
            active_mask=active_mask,
        )
        role_codes = model_input.get("info", {}).get("role_bias_codes")
        if self.role_bias_layer is not None and role_codes is not None:
            final_policy_logits = self.role_bias_layer(final_policy_logits, role_codes)
        actions_per_square = actor_kwargs.get("actions_per_square", MAX_OVERLAPPING_ACTIONS)
        actions = {}
        for action_space, logits in final_policy_logits.items():
            n_actions = logits.shape[-1]
            flat_actions = DictActor.logits_to_actions(
                logits.view(-1, n_actions),
                sample=sample,
                actions_per_square=actions_per_square,
            )
            actions[action_space] = flat_actions.view(*logits.shape[:-1], -1)

        outputs: Dict[str, Any] = {
            "actions": actions,
            "policy_logits": final_policy_logits,
            "baseline": baseline,
        }
        if return_sidecar_outputs:
            outputs.update({
                "base_policy_logits": base_policy_logits,
                "risk_logits": sidecar_output["risk_logits"],
                "risk_probabilities": sidecar_output["risk_probabilities"],
                "safe_expansion_logits": sidecar_output["safe_expansion_logits"],
                "safe_expansion_probabilities": sidecar_output["safe_expansion_probabilities"],
                "logit_deltas": logit_deltas,
                "logical_shapes": sidecar_output["logical_shapes"],
            })
        if return_features:
            outputs.update({
                "base_features": base_features,
                "feature_input_mask": input_mask,
            })
        return outputs

    def _gate_tensors(self, model_input: dict, logical_shapes: torch.Tensor, like: torch.Tensor):
        batch_size = like.shape[0]
        threshold = torch.ones_like(like)
        active = torch.zeros_like(like, dtype=torch.bool)
        turn_tensor = model_input["obs"].get("turn")
        for batch_index in range(batch_size):
            map_size = int(logical_shapes[batch_index, 0, 0].item())
            if map_size not in self.risk_thresholds:
                continue
            if turn_tensor is None:
                turn = 0
            else:
                normalized_turn = float(turn_tensor[batch_index].reshape(-1)[0].item())
                turn = int(round(normalized_turn * 360.0)) if normalized_turn <= 1.0 else int(round(normalized_turn))
            threshold[batch_index].fill_(self.risk_thresholds[map_size])
            for player in range(2):
                enabled = self.map_phase_active(map_size, turn, player)
                active[batch_index, player].fill_(enabled)
        return threshold, active

    @staticmethod
    def map_phase_active(map_size: int, turn: int, player: int) -> bool:
        if map_size in (12, 16):
            return turn >= 120 and 25 <= turn % 40 <= 29
        if map_size == 24 and player == 0 and turn < 80:
            return False
        return True

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)

    def trainable_sidecar_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.spatial_risk_sidecar.parameters()
        yield from self.intervention_gate.parameters()

    def assert_base_unchanged(self) -> None:
        current_hash = state_dict_sha256(self.base_agent.state_dict())
        if current_hash != self._base_state_hash:
            raise AssertionError(
                f"Frozen base state changed: initial={self._base_state_hash}, current={current_hash}"
            )

    def composite_checkpoint(self, include_base_state: bool = False) -> dict:
        checkpoint = {
            "format": "lux_sidecar_v1",
            "base_state_hash": state_dict_sha256(self.base_agent.state_dict()),
            "spatial_risk_sidecar_state_dict": self.spatial_risk_sidecar.state_dict(),
            "intervention_gate_state_dict": self.intervention_gate.state_dict(),
        }
        if include_base_state:
            checkpoint["base_model_state_dict"] = self.base_agent.state_dict()
        return checkpoint

    def save_sidecar_checkpoint(self, path: Path, include_base_state: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.composite_checkpoint(include_base_state=include_base_state), path)

    def load_sidecar_checkpoint(self, path: Path, strict_base_hash: bool = True) -> dict:
        checkpoint = torch.load(path, map_location="cpu")
        expected_hash = checkpoint.get("base_state_hash")
        current_hash = state_dict_sha256(self.base_agent.state_dict())
        if strict_base_hash and expected_hash != current_hash:
            raise ValueError(f"Base hash mismatch: checkpoint={expected_hash}, current={current_hash}")
        self.spatial_risk_sidecar.load_state_dict(
            checkpoint["spatial_risk_sidecar_state_dict"],
            strict=True,
        )
        self.intervention_gate.load_state_dict(
            checkpoint["intervention_gate_state_dict"],
            strict=True,
        )
        return checkpoint
