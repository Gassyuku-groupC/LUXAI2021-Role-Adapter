# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
from omegaconf import OmegaConf
import os
from pathlib import Path
import pprint
from queue import Empty
import threading
import time
import timeit
import traceback
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union
import wandb
import warnings

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, td_lambda, upgo, vtrace
from .core.buffer_utils import Buffers, create_buffers, fill_buffers_inplace, stack_buffers, split_buffers, \
    buffers_apply
from ..lux_gym import create_env
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns import create_model
from ..rl_agent.auxiliary_heads import AuxiliaryRiskHead
from ..rl_agent.data_augmentation import HorizontalFlip, Rot90, Rot180, Rot270, VerticalFlip
from ..rl_agent.gate_policy import apply_runtime_gate_to_actor_output
from ..rl_agent.role_assignment import RoleAssignmentConfig
from ..rl_agent.trainable_role_bias import RoleBiasCodeBuilder, attach_role_bias_codes
from ..utils import flags_to_namespace


KL_DIV_LOSS = nn.KLDivLoss(reduction="none")
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


TRAINING_AUGMENTERS = (HorizontalFlip, VerticalFlip, Rot90, Rot180, Rot270)


def load_training_model_state(model: nn.Module, state_dict: Dict, flags: SimpleNamespace) -> None:
    """Load a legacy actor checkpoint while allowing a newly added gate head."""
    if getattr(flags, "role_local_adapter_enabled", False):
        if any(key.startswith("role_local_adapter.") for key in state_dict):
            model.load_state_dict(state_dict, strict=True)
            return
        if any(key.startswith("base_agent.") for key in state_dict):
            retained = {
                key: value for key, value in state_dict.items()
                if key.startswith("base_agent.") or key.startswith("role_bias_layer.")
            }
            incompatible = model.load_state_dict(retained, strict=False)
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith(("role_local_adapter.", "role_bias_layer."))
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible role-local checkpoint: missing={invalid_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
            logging.info("Ignored legacy Sidecar/Gate tensors and initialized zero local Role adapter.")
            return
        model.base_agent.load_state_dict(state_dict, strict=True)
        logging.info("Initialized Role-local adapter on an unchanged legacy actor.")
    elif getattr(flags, "spatial_risk_sidecar_enabled", False):
        if any(key.startswith("base_agent.") for key in state_dict):
            incompatible = model.load_state_dict(state_dict, strict=False)
            allowed_missing = (
                "role_bias_layer." if getattr(flags, "role_bias_training_enabled", False) else ()
            )
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not allowed_missing or not key.startswith(allowed_missing)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible sidecar checkpoint: missing={invalid_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
            return
        incompatible = model.base_agent.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Incompatible frozen base checkpoint: {incompatible}")
        logging.info("Initialized external spatial sidecar on an unchanged legacy actor.")
    elif getattr(flags, "learned_intervention_gate_enabled", False):
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("learned_intervention_gate.")
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                "Incompatible learned-gate checkpoint. "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
        if incompatible.missing_keys:
            logging.info("Initialized a new learned intervention gate on top of the loaded actor.")
    else:
        model.load_state_dict(state_dict)


def configure_gate_only_training(model: nn.Module, training: bool) -> None:
    """Freeze the complete base policy and expose only the intervention gate."""
    if hasattr(model, "spatial_risk_sidecar") and hasattr(model, "intervention_gate"):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.spatial_risk_sidecar, model.intervention_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            module.train(training)
        model.base_agent.eval()
        return
    gate = getattr(model, "learned_intervention_gate", None)
    if gate is None:
        raise ValueError("gate_only_training requires learned_intervention_gate_enabled=true.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in gate.parameters():
        parameter.requires_grad_(True)
    model.eval()
    gate.train(training)


def configure_role_only_training(model: nn.Module, training: bool) -> None:
    role_layer = getattr(model, "role_bias_layer", None)
    if role_layer is None:
        raise ValueError("role_only_training requires role_bias_training_enabled=true.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in role_layer.parameters():
        parameter.requires_grad_(True)
    model.eval()
    role_layer.train(training)


def configure_role_sidecar_training(model: nn.Module, training: bool) -> None:
    required = ("spatial_risk_sidecar", "intervention_gate", "role_bias_layer")
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(
            "role_sidecar_training requires sidecar, gate, and role bias modules; "
            f"missing={missing}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in required:
        module = getattr(model, name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    for name in required:
        getattr(model, name).train(training)


def configure_role_local_training(model: nn.Module, training: bool) -> None:
    local_adapter = getattr(model, "role_local_adapter", None)
    if local_adapter is None:
        raise ValueError("role_local_training requires role_local_adapter_enabled=true.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in local_adapter.parameters():
        parameter.requires_grad_(True)
    model.eval()
    local_adapter.train(training)


def configure_role_local_bias_training(model: nn.Module, training: bool) -> None:
    """Train local role deltas and compact role biases with the Actor frozen."""
    required = ("role_local_adapter", "role_bias_layer")
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(f"role_local_bias_training missing modules: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.role_local_adapter, model.role_bias_layer):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.eval()
    model.role_local_adapter.train(training)
    model.role_bias_layer.train(training)


def configure_role_joint_head_training(model: nn.Module, training: bool) -> None:
    """Train policy/value heads and role adapters while freezing the backbone."""
    required = ("base_agent", "role_bias_layer", "role_local_adapter")
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(f"role_joint_head_training missing modules: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head_modules = (
        model.base_agent.actor_base,
        model.base_agent.actor,
        model.base_agent.baseline_base,
        model.base_agent.baseline,
    )
    for module in (*head_modules, model.role_bias_layer, model.role_local_adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    tail_blocks = int(getattr(model, "role_joint_backbone_blocks", 0))
    if tail_blocks > 0:
        residual_blocks = list(model.base_agent.base_model.children())[1:]
        if tail_blocks > len(residual_blocks):
            raise ValueError(
                f"role_joint_backbone_blocks={tail_blocks} exceeds residual block count "
                f"{len(residual_blocks)}"
            )
        for block in residual_blocks[-tail_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    if hasattr(model, "set_backbone_tail_training"):
        model.set_backbone_tail_training(tail_blocks)
    if hasattr(model, "set_policy_head_training"):
        model.set_policy_head_training(training)
    model.train(training)
    if not hasattr(model, "set_backbone_tail_training"):
        model.base_agent.base_model.eval()
        if training and tail_blocks > 0:
            for block in residual_blocks[-tail_blocks:]:
                block.train(True)
    if not hasattr(model, "set_policy_head_training"):
        for module in head_modules:
            module.train(training)
    model.role_bias_layer.train(training)
    model.role_local_adapter.train(training)


def create_role_code_builder(flags) -> Optional[RoleBiasCodeBuilder]:
    if not getattr(flags, "role_bias_training_enabled", False):
        return None
    raw_config = getattr(flags, "role_assignment", {})
    if isinstance(raw_config, SimpleNamespace):
        raw_config = vars(raw_config)
    config = RoleAssignmentConfig.from_mapping(raw_config)
    if not config.enabled:
        raise ValueError("role_bias_training_enabled requires role_assignment.enabled=true.")
    return RoleBiasCodeBuilder(config)


def add_role_codes_if_enabled(env_output, env, builder):
    return attach_role_bias_codes(env_output, env, builder) if builder is not None else env_output


def load_spatial_sidecar_pretrain(model: nn.Module, path: Optional[str]) -> None:
    if not path:
        return
    if not hasattr(model, "spatial_risk_sidecar"):
        raise ValueError("spatial_risk_sidecar_checkpoint requires spatial_risk_sidecar_enabled=true")
    checkpoint = torch.load(Path(path), map_location="cpu")
    model.spatial_risk_sidecar.load_state_dict(
        checkpoint["spatial_risk_sidecar_state_dict"], strict=True
    )
    if checkpoint.get("intervention_gate_state_dict") is not None:
        model.intervention_gate.load_state_dict(
            checkpoint["intervention_gate_state_dict"], strict=True
        )


def load_student_pretrain(model: nn.Module, path: Optional[str]) -> None:
    """Restore the complete Actor+Sidecar BC student before online APPO."""
    if not path:
        return
    checkpoint = torch.load(Path(path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    logging.info("Loaded joint Actor+Sidecar student pretrain from %s", path)


def format_actor_status(actor_processes) -> str:
    statuses = []
    for idx, actor in enumerate(actor_processes or []):
        exit_code = getattr(actor, "exitcode", "n/a")
        statuses.append(f"{idx}:alive={actor.is_alive()},exit={exit_code}")
    return ", ".join(statuses) or "unavailable"


def augment_training_batch(batch: Dict, flags: SimpleNamespace) -> Dict:
    """Apply one consistent board symmetry and optional player swap per rollout batch."""
    if flags.training_spatial_augmentation:
        augmenter = TRAINING_AUGMENTERS[
            torch.randint(len(TRAINING_AUGMENTERS), (1,)).item()
        ]()
        for value in batch["obs"].values():
            if value.dim() == 6:
                value.copy_(augmenter.op(value, inverse=False, is_policy=False))
        batch["info"]["input_mask"].copy_(
            augmenter.op(batch["info"]["input_mask"], inverse=False, is_policy=False)
        )
        for space in batch["actions"]:
            for container, key in (
                (batch["policy_logits"], space),
                (batch["info"]["available_actions_mask"], space),
                (batch["info"]["actions_taken"], space),
            ):
                transformed = augmenter.op(container[key], inverse=False, is_policy=True)
                container[key].copy_(augmenter._transform_policy({space: transformed}, inverse=False)[space])
            if "role_bias_codes" in batch["info"]:
                role_codes = batch["info"]["role_bias_codes"][space]
                transformed = augmenter.op(role_codes, inverse=False, is_policy=True)
                role_codes.copy_(
                    augmenter._transform_policy({space: transformed}, inverse=False)[space]
                )

            actions = augmenter.op(batch["actions"][space], inverse=False, is_policy=True)
            policy_permutation = augmenter.transformed_action_idxs_forward[space]
            inverse_permutation = torch.empty(
                len(policy_permutation), dtype=torch.long, device=actions.device
            )
            inverse_permutation[torch.tensor(policy_permutation, device=actions.device)] = torch.arange(
                len(policy_permutation), device=actions.device
            )
            batch["actions"][space].copy_(inverse_permutation[actions])

    def maybe_swap_players(value: torch.Tensor, player_dim: int) -> torch.Tensor:
        if not flags.training_player_swap_per_sample:
            return value.flip(player_dim).clone()
        if not player_swap_mask.any():
            return value.clone()
        mask_shape = [1] * value.dim()
        mask_shape[1] = player_swap_mask.shape[0]
        mask = player_swap_mask.to(value.device).view(mask_shape)
        swapped = value.flip(player_dim).clone()
        return torch.where(mask, swapped, value).clone()

    player_swap_mask = None
    if flags.training_player_swap_per_sample:
        batch_size = next(iter(batch["actions"].values())).shape[1]
        player_swap_mask = (
            torch.rand(batch_size, device=next(iter(batch["actions"].values())).device)
            < flags.training_player_swap_probability
        )
    if (
        flags.training_player_swap_per_sample
        or torch.rand(()).item() < flags.training_player_swap_probability
    ):
        for value in batch["obs"].values():
            if value.dim() == 6 and value.shape[-3] == 2:
                value.copy_(maybe_swap_players(value, -3))
            elif value.dim() < 6 and value.shape[-1] == 2:
                value.copy_(maybe_swap_players(value, -1))
        for group_name in ("actions", "policy_logits"):
            for value in batch[group_name].values():
                value.copy_(maybe_swap_players(value, -4))
        for group_name in ("available_actions_mask", "actions_taken"):
            for value in batch["info"][group_name].values():
                value.copy_(maybe_swap_players(value, -4))
        if "role_bias_codes" in batch["info"]:
            for value in batch["info"]["role_bias_codes"].values():
                value.copy_(maybe_swap_players(value, -4))
        input_mask = batch["info"]["input_mask"]
        input_mask.copy_(maybe_swap_players(input_mask, -3))
        for key in ("reward", "baseline"):
            batch[key].copy_(maybe_swap_players(batch[key], -1))
        for key, value in batch["info"].items():
            if key.startswith("LOGGING_") and value.shape[-1:] == (2,):
                value.copy_(maybe_swap_players(value, -1))
    return batch


def combine_policy_logits_to_log_probs(
        behavior_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step

    Initial shape: time, batch, 1, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    # All-masked rows contain -inf. Keep those rows finite through softmax so
    # their masked-out backward pass cannot create NaN gradients.
    safe_policy_logits = torch.where(
        torch.isfinite(behavior_policy_logits),
        behavior_policy_logits,
        torch.full_like(behavior_policy_logits, -1e4),
    )
    probs = F.softmax(safe_policy_logits, dim=-1)
    # Ignore probabilities for actions that were not used
    probs = actions_taken_mask * probs
    # Select the probabilities for actions that were taken by stacked agents and sum these
    selected_probs = torch.gather(probs, -1, actions)
    # Convert the probs to conditional probs, since we sample without replacement
    remaining_probability_density = 1. - torch.cat([
        torch.zeros(
            (*selected_probs.shape[:-1], 1),
            device=selected_probs.device,
            dtype=selected_probs.dtype
        ),
        selected_probs[..., :-1].cumsum(dim=-1)
    ], dim=-1)
    # Avoid division by zero
    remaining_probability_density = remaining_probability_density + torch.where(
        remaining_probability_density == 0,
        torch.ones_like(remaining_probability_density),
        torch.zeros_like(remaining_probability_density)
    )
    conditional_selected_probs = selected_probs / remaining_probability_density
    # Remove 0-valued conditional_selected_probs in order to eliminate neg-inf valued log_probs
    conditional_selected_probs = conditional_selected_probs + torch.where(
        conditional_selected_probs == 0,
        torch.ones_like(conditional_selected_probs),
        torch.zeros_like(conditional_selected_probs)
    )
    log_probs = torch.log(conditional_selected_probs)
    # Sum over actions, y and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    return torch.flatten(log_probs, start_dim=-3, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def count_effective_actions(actions_taken_by_space: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Count actions for each time, batch, and player without map-size bias."""
    counts = None
    for mask in actions_taken_by_space.values():
        space_counts = mask.sum(dim=(2, 4, 5, 6))
        counts = space_counts if counts is None else counts + space_counts
    if counts is None:
        raise ValueError("At least one action space is required")
    return counts


def combine_policy_entropy(
        policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.
    NB: We are just computing the sum of individual entropies, not the joint entropy, because I don't think there is
    an efficient way to compute the joint entropy?

    Initial shape: time, batch, action_planes, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    safe_policy_logits = torch.where(
        torch.isfinite(policy_logits),
        policy_logits,
        torch.full_like(policy_logits, -1e4),
    )
    policy = F.softmax(safe_policy_logits, dim=-1)
    log_policy = F.log_softmax(safe_policy_logits, dim=-1)
    entropies = (policy * log_policy).sum(dim=-1)
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine entropies from different actions
    return entropies_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    common_legal = torch.isfinite(learner_policy_logits) & torch.isfinite(teacher_policy_logits)
    learner_safe_logits = torch.where(
        common_legal, learner_policy_logits, torch.full_like(learner_policy_logits, -1e4)
    )
    teacher_safe_logits = torch.where(
        common_legal, teacher_policy_logits.detach(), torch.full_like(teacher_policy_logits, -1e4)
    )
    learner_policy_log_probs = F.log_softmax(learner_safe_logits, dim=-1)
    learner_policy = learner_policy_log_probs.exp()
    teacher_policy_log_probs = F.log_softmax(teacher_safe_logits, dim=-1)
    # KL(pi_theta || pi_ref), matching the immutable-reference constraint.
    kl_div = (
        learner_policy * (learner_policy_log_probs - teacher_policy_log_probs)
    ).sum(dim=-1)
    kl_div = torch.nan_to_num(kl_div, nan=0., posinf=0., neginf=0.)
    assert actions_taken_mask.shape == kl_div.shape
    kl_div_masked = kl_div * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def compute_teacher_bc_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard behavior cloning from the teacher on learner-visited states."""
    teacher_actions = teacher_policy_logits.detach().argmax(dim=-1)
    n_actions = learner_policy_logits.shape[-1]
    learner_safe_logits = torch.where(
        torch.isfinite(learner_policy_logits),
        learner_policy_logits,
        torch.full_like(learner_policy_logits, -1e4),
    )
    cross_entropy = F.cross_entropy(
        learner_safe_logits.reshape(-1, n_actions),
        teacher_actions.reshape(-1),
        reduction="none",
    ).view_as(teacher_actions)
    cross_entropy = torch.nan_to_num(cross_entropy, nan=0., posinf=0., neginf=0.)
    mask = actions_taken_mask.float()
    masked_loss = cross_entropy * mask
    combined_loss = masked_loss.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)
    correct = ((learner_policy_logits.argmax(dim=-1) == teacher_actions) * actions_taken_mask).sum()
    count = actions_taken_mask.sum()
    return combined_loss, correct, count


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def compute_baseline_loss(values: torch.Tensor, value_targets: torch.Tensor, reduction: str) -> torch.Tensor:
    baseline_loss = F.smooth_l1_loss(values, value_targets.detach(), reduction="none")
    return reduce(baseline_loss, reduction=reduction)


def count_city_tiles(obs_city_tile: torch.Tensor) -> torch.Tensor:
    counts = obs_city_tile.float().sum(dim=(-1, -2))
    if counts.dim() >= 3 and counts.shape[-2] == 1 and counts.shape[-1] == 2:
        counts = counts.squeeze(-2)
    elif counts.dim() >= 3 and counts.shape[-1] == 1 and counts.shape[-2] == 2:
        counts = counts.squeeze(-1)
    if counts.shape[-1] != 2:
        raise RuntimeError(f"Expected a final player dimension of 2 for city_tile counts. Found: {counts.shape}")
    return counts


def make_city_loss_labels(city_counts: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = city_counts.shape[0]
    labels = torch.zeros_like(city_counts, dtype=torch.float32)
    valid = torch.zeros_like(city_counts, dtype=torch.bool)
    for t in range(seq_len - 1):
        end = min(seq_len, t + int(horizon) + 1)
        future_min = city_counts[t + 1:end].min(dim=0).values
        labels[t] = (future_min < city_counts[t]).float()
        valid[t] = city_counts[t] > 0
    return labels, valid


def balanced_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, pos_weight_scale: float) -> torch.Tensor:
    logits = logits[valid]
    labels = labels[valid]
    if labels.numel() == 0:
        return logits.sum() * 0.0
    positives = labels.sum().clamp(min=1.0)
    negatives = (labels.numel() - labels.sum()).clamp(min=1.0)
    pos_weight = (negatives / positives * float(pos_weight_scale)).clamp(min=1.0, max=20.0)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def compute_policy_gradient_loss(
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)
    return reduce(cross_entropy * advantages.detach(), reduction)


def compute_appo_policy_loss(
        target_action_log_probs: torch.Tensor,
        behavior_action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
        reduction: str,
        valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped surrogate over asynchronous behavior-policy rollouts."""
    if valid_mask is None:
        valid_mask = torch.isfinite(target_action_log_probs) & torch.isfinite(
            behavior_action_log_probs
        )
    else:
        valid_mask = valid_mask & torch.isfinite(target_action_log_probs) & torch.isfinite(
            behavior_action_log_probs
        )
    if not valid_mask.any():
        zero = target_action_log_probs[valid_mask].sum() * 0.0
        return zero, zero.detach(), zero.detach()
    log_ratio = (target_action_log_probs - behavior_action_log_probs).clamp(-20.0, 20.0)
    log_ratio = log_ratio[valid_mask]
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
    detached_advantages = advantages.detach()[valid_mask]
    surrogate = torch.minimum(ratio * detached_advantages, clipped_ratio * detached_advantages)
    clip_fraction = (ratio != clipped_ratio).float().mean()
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    return -reduce(surrogate, reduction), clip_fraction, approximate_kl


def stabilize_policy_advantages(
        advantages: torch.Tensor,
        normalize: bool,
        clip: Optional[float],
) -> torch.Tensor:
    advantages = torch.nan_to_num(advantages, nan=0., posinf=0., neginf=0.)
    if normalize:
        advantages = (
            (advantages - advantages.mean())
            / advantages.std(unbiased=False).clamp(min=1e-6)
        )
    if clip is not None:
        advantages = advantages.clamp(min=-clip, max=clip)
    return advantages


def role_repair_advantage_weights(batch, advantages: torch.Tensor, flags) -> torch.Tensor:
    if not (
            getattr(flags, "role_only_training", False)
            or getattr(flags, "role_local_bias_training", False)
    ):
        return torch.ones_like(advantages)
    board_size = batch["obs"]["board_size"].reshape(*advantages.shape[:2], -1)[..., 0]
    cycle = batch["obs"]["day_night_cycle"].reshape(*advantages.shape[:2], -1)[..., 0]
    weights = torch.ones_like(advantages)
    small_map_weight = float(getattr(flags, "role_small_map_policy_weight", 0.25))
    weights = torch.where((board_size == 0).unsqueeze(-1), weights * small_map_weight, weights)
    hard_map = (board_size == 1) | (board_size == 2)
    hard_window = hard_map & (cycle >= int(getattr(flags, "role_hard_window_start", 25)))
    hard_weight = float(getattr(flags, "role_hard_window_weight", 2.0))
    weights = torch.where(hard_window.unsqueeze(-1), weights * hard_weight, weights)
    return weights


def load_fixed_opponents(flags: SimpleNamespace) -> list[nn.Module]:
    config_paths = list(getattr(flags, "fixed_opponent_configs", []) or [])
    checkpoint_paths = list(getattr(flags, "fixed_opponent_checkpoints", []) or [])
    if not config_paths and not checkpoint_paths:
        config_path = getattr(flags, "fixed_opponent_config", None)
        checkpoint_path = getattr(flags, "fixed_opponent_checkpoint", None)
        if not config_path or not checkpoint_path:
            return []
        config_paths = [config_path]
        checkpoint_paths = [checkpoint_path]
    if len(config_paths) != len(checkpoint_paths) or not config_paths:
        raise ValueError("fixed opponent config/checkpoint lists must be non-empty and equal length")
    models = []
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        opponent_flags = flags_to_namespace(OmegaConf.to_container(OmegaConf.load(config_path)))
        model = create_model(opponent_flags, flags.actor_device)
        checkpoint = torch.load(checkpoint_path, map_location=flags.actor_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models


def replace_fixed_opponent_actions(
        learner_output: Dict,
        opponent_output: Dict,
        learner_players: Tuple[int, ...],
) -> Dict:
    for action_space, learner_actions in learner_output["actions"].items():
        opponent_actions = opponent_output["actions"][action_space]
        if learner_actions.shape != opponent_actions.shape or learner_actions.dim() < 4:
            raise ValueError(f"Fixed-opponent action shape mismatch for {action_space}")
        player_dim = learner_actions.dim() - 4
        for env_index, learner_player in enumerate(learner_players):
            index = [slice(None)] * learner_actions.dim()
            index[0] = env_index
            index[player_dim] = 1 - int(learner_player)
            learner_actions[tuple(index)] = opponent_actions[tuple(index)]
    return learner_output


def replace_fixed_opponent_pool_actions(
        learner_output: Dict,
        opponent_outputs: list[Dict],
        learner_players: Tuple[int, ...],
        opponent_indices: Optional[Tuple[int, ...]] = None,
) -> Dict:
    if not opponent_outputs:
        return learner_output
    if opponent_indices is None:
        opponent_indices = tuple(
            env_index % len(opponent_outputs) for env_index in range(len(learner_players))
        )
    if len(opponent_indices) != len(learner_players):
        raise ValueError("Fixed-opponent index count must match learner player count")
    for env_index, learner_player in enumerate(learner_players):
        opponent_index = int(opponent_indices[env_index])
        if not 0 <= opponent_index < len(opponent_outputs):
            raise ValueError(f"Fixed-opponent index out of range: {opponent_index}")
        opponent_output = opponent_outputs[opponent_index]
        for action_space, learner_actions in learner_output["actions"].items():
            opponent_actions = opponent_output["actions"][action_space]
            if learner_actions.shape != opponent_actions.shape or learner_actions.dim() < 4:
                raise ValueError(f"Fixed-opponent action shape mismatch for {action_space}")
            player_dim = learner_actions.dim() - 4
            index = [slice(None)] * learner_actions.dim()
            index[0] = env_index
            index[player_dim] = 1 - int(learner_player)
            learner_actions[tuple(index)] = opponent_actions[tuple(index)]
    return learner_output


def mask_fixed_opponent_actions_taken(env_output: Dict, learner_players: Tuple[int, ...]) -> None:
    actions_taken = env_output.get("info", {}).get("actions_taken", {})
    for value in actions_taken.values():
        if value.dim() < 4:
            raise ValueError(f"Unexpected actions_taken shape: {tuple(value.shape)}")
        player_dim = value.dim() - 4
        for env_index, learner_player in enumerate(learner_players):
            index = [slice(None)] * value.dim()
            index[0] = env_index
            index[player_dim] = 1 - int(learner_player)
            value[tuple(index)] = False


@torch.no_grad()
def act(
        flags: SimpleNamespace,
        teacher_flags: Optional[SimpleNamespace],
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module,
        buffers: Buffers,
):
    if flags.debug:
        catch_me = AssertionError
    else:
        catch_me = Exception
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()

        env = create_env(flags, device=flags.actor_device, teacher_flags=teacher_flags)
        fixed_opponent_models = load_fixed_opponents(flags)
        learner_players = tuple(
            (actor_index * flags.n_actor_envs + env_index) % 2
            for env_index in range(flags.n_actor_envs)
        )
        opponent_indices = [
            (actor_index * flags.n_actor_envs + env_index) % len(fixed_opponent_models)
            for env_index in range(flags.n_actor_envs)
        ] if fixed_opponent_models else []
        role_code_builder = create_role_code_builder(flags)
        if flags.seed is not None:
            env.seed(flags.seed + actor_index * flags.n_actor_envs)
        else:
            env.seed()
        env_output = add_role_codes_if_enabled(env.reset(force=True), env, role_code_builder)
        agent_output = actor_model(env_output)
        agent_output = apply_runtime_gate_to_actor_output(agent_output, env, flags)
        if fixed_opponent_models:
            agent_output = replace_fixed_opponent_pool_actions(
                agent_output,
                [model(env_output) for model in fixed_opponent_models],
                learner_players,
                tuple(opponent_indices),
            )
        agent_output.pop("role_local_deltas", None)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), 0)

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                agent_output = actor_model(env_output)
                agent_output = apply_runtime_gate_to_actor_output(agent_output, env, flags)
                if fixed_opponent_models:
                    agent_output = replace_fixed_opponent_pool_actions(
                        agent_output,
                        [model(env_output) for model in fixed_opponent_models],
                        learner_players,
                        tuple(opponent_indices),
                    )
                agent_output.pop("role_local_deltas", None)
                timings.time("model")

                env_output = env.step(agent_output["actions"])
                if env_output["done"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward"]
                    cached_done = env_output["done"]
                    cached_info_actions_taken = env_output["info"]["actions_taken"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
                    }

                    if fixed_opponent_models:
                        done_by_env = cached_done.reshape(cached_done.shape[0], -1).any(dim=1)
                        for env_index, is_done in enumerate(done_by_env.tolist()):
                            if is_done:
                                opponent_indices[env_index] = (
                                    opponent_indices[env_index] + 1
                                ) % len(fixed_opponent_models)

                    env_output = env.reset()
                    env_output["reward"] = cached_reward
                    env_output["done"] = cached_done
                    env_output["info"]["actions_taken"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)
                if fixed_opponent_models:
                    mask_fixed_opponent_actions_taken(env_output, learner_players)
                env_output = add_role_codes_if_enabled(env_output, env, role_code_builder)
                timings.time("step")

                fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), t + 1)
                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except catch_me as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags: SimpleNamespace,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
    actor_processes=None,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = []
        for _ in range(max(flags.batch_size // flags.n_actor_envs, 1)):
            try:
                indices.append(full_queue.get(timeout=flags.rollout_queue_timeout_seconds))
            except Empty as exc:
                raise RuntimeError(
                    "Timed out waiting for an actor rollout after "
                    f"{flags.rollout_queue_timeout_seconds} seconds. "
                    "This usually means an actor crashed or the Lux environment stalled. "
                    f"Actors: {format_actor_status(actor_processes)}"
                ) from exc
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices], dim=1)
    timings.time("batch")
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    return batch


def learn(
        flags: SimpleNamespace,
        actor_model: nn.Module,
        learner_model: nn.Module,
        teacher_model: Optional[nn.Module],
        aux_risk_head: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: torch.amp.GradScaler,
        lr_scheduler: torch.optim.lr_scheduler,
        total_games_played: int,
        baseline_only: bool = False,
        lock=threading.Lock(),
) -> Tuple[Dict, int]:
    """Performs a learning (optimization) step."""
    with lock:
        batch = augment_training_batch(batch, flags)
        with torch.amp.autocast("cuda", enabled=flags.use_mixed_precision):
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))
            use_aux_risk = bool(getattr(flags, "use_aux_risk", False)) and aux_risk_head is not None
            gate_only = bool(getattr(flags, "gate_only_training", False))
            learner_outputs = learner_model(
                flattened_batch,
                return_features=use_aux_risk or gate_only,
            )
            aux_base_features = learner_outputs.pop("base_features", None)
            aux_input_mask = learner_outputs.pop("feature_input_mask", None)
            gate_probabilities = learner_outputs.pop("gate_probabilities", {})
            gate_probability_mean = torch.zeros((), device=flags.learner_device)
            if gate_probabilities:
                gate_probability_mean = torch.stack([
                    probability.mean() for probability in gate_probabilities.values()
                ]).mean()
            gate_intervention_loss = (
                getattr(flags, "gate_intervention_cost", 0.0) * gate_probability_mean
            )
            aux_risk_loss = torch.zeros((), device=flags.learner_device)
            aux_loss20_recall = torch.zeros((), device=flags.learner_device)
            aux_loss20_precision = torch.zeros((), device=flags.learner_device)
            aux_loss20_positive_rate = torch.zeros((), device=flags.learner_device)
            if use_aux_risk:
                aux_logits = aux_risk_head(aux_base_features, aux_input_mask)["loss20_logit"]
                aux_logits = aux_logits.view(flags.unroll_length + 1, flags.batch_size, 2)
                city_counts = count_city_tiles(batch["obs"]["city_tile"])
                aux_labels, aux_valid = make_city_loss_labels(
                    city_counts,
                    horizon=getattr(flags, "aux_risk_horizon", 20),
                )
                aux_risk_loss = balanced_bce_with_logits(
                    aux_logits[:-1],
                    aux_labels[:-1],
                    aux_valid[:-1],
                    pos_weight_scale=getattr(flags, "aux_risk_pos_weight_scale", 1.0),
                )
                aux_probs = aux_logits[:-1].detach().sigmoid()
                aux_pred = aux_probs >= getattr(flags, "aux_risk_threshold", 0.40)
                aux_eval_valid = aux_valid[:-1]
                aux_eval_labels = aux_labels[:-1].bool()
                aux_tp = (aux_pred & aux_eval_labels & aux_eval_valid).sum().float()
                aux_fp = (aux_pred & ~aux_eval_labels & aux_eval_valid).sum().float()
                aux_fn = (~aux_pred & aux_eval_labels & aux_eval_valid).sum().float()
                aux_loss20_recall = aux_tp / (aux_tp + aux_fn).clamp(min=1.0)
                aux_loss20_precision = aux_tp / (aux_tp + aux_fp).clamp(min=1.0)
                aux_loss20_positive_rate = (aux_eval_labels & aux_eval_valid).sum().float() / aux_eval_valid.sum().clamp(min=1)
            learner_outputs = buffers_apply(learner_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                              flags.batch_size,
                                                                              *x.shape[1:]))
            if flags.use_teacher:
                with torch.no_grad():
                    teacher_outputs = teacher_model(flattened_batch)
                    teacher_outputs = buffers_apply(teacher_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                                      flags.batch_size,
                                                                                      *x.shape[1:]))
            else:
                teacher_outputs = None

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = buffers_apply(batch, lambda x: x[1:])
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])
            if flags.use_teacher:
                teacher_outputs = buffers_apply(teacher_outputs, lambda x: x[:-1])

            combined_behavior_action_log_probs = torch.zeros(
                (flags.unroll_length, flags.batch_size, 2),
                device=flags.learner_device
            )
            combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)
            effective_action_counts = count_effective_actions(
                batch["info"]["actions_taken"]
            ).to(dtype=combined_behavior_action_log_probs.dtype)
            combined_teacher_kl_loss = torch.zeros_like(combined_behavior_action_log_probs)
            teacher_kl_losses = {}
            teacher_bc_space_losses = {}
            teacher_bc_accuracies = {}
            teacher_bc_total_count = torch.zeros((), device=flags.learner_device)
            teacher_bc_total_correct = torch.zeros((), device=flags.learner_device)
            combined_learner_entropy = torch.zeros_like(combined_behavior_action_log_probs)
            entropies = {}
            for act_space in batch["actions"].keys():
                actions = batch["actions"][act_space]
                actions_taken_mask = batch["info"]["actions_taken"][act_space]

                behavior_policy_logits = batch["policy_logits"][act_space]
                behavior_action_log_probs = combine_policy_logits_to_log_probs(
                    behavior_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_behavior_action_log_probs = combined_behavior_action_log_probs + behavior_action_log_probs

                learner_policy_logits = learner_outputs["policy_logits"][act_space]
                learner_action_log_probs = combine_policy_logits_to_log_probs(
                    learner_policy_logits,
                    actions,
                    actions_taken_mask
                )
                combined_learner_action_log_probs = combined_learner_action_log_probs + learner_action_log_probs

                # Only take entropy and KL loss for tiles where at least one action was taken
                any_actions_taken = actions_taken_mask.any(dim=-1)
                if flags.use_teacher:
                    teacher_kl_loss = compute_teacher_kl_loss(
                        learner_policy_logits,
                        teacher_outputs["policy_logits"][act_space],
                        any_actions_taken
                    )
                    teacher_bc_space_loss, teacher_bc_correct, teacher_bc_count = compute_teacher_bc_loss(
                        learner_policy_logits,
                        teacher_outputs["policy_logits"][act_space],
                        any_actions_taken,
                    )
                else:
                    teacher_kl_loss = torch.zeros_like(combined_teacher_kl_loss)
                    teacher_bc_space_loss = torch.zeros_like(combined_behavior_action_log_probs)
                    teacher_bc_correct = torch.zeros((), device=flags.learner_device)
                    teacher_bc_count = torch.zeros((), device=flags.learner_device)
                combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss
                teacher_bc_total_count = teacher_bc_total_count + teacher_bc_count
                teacher_bc_total_correct = teacher_bc_total_correct + teacher_bc_correct
                teacher_bc_space_losses[act_space] = (
                    teacher_bc_space_loss.sum() / teacher_bc_count.clamp(min=1)
                )
                n_actions_taken = any_actions_taken.sum().clamp(min=1)
                teacher_kl_losses[act_space] = (reduce(
                    teacher_kl_loss,
                    reduction="sum",
                ) / n_actions_taken).detach().cpu().item()
                teacher_bc_accuracies[act_space] = (
                    teacher_bc_correct / teacher_bc_count.clamp(min=1)
                ).detach().cpu().item()

                learner_policy_entropy = combine_policy_entropy(
                    learner_policy_logits,
                    any_actions_taken
                )
                combined_learner_entropy = combined_learner_entropy + learner_policy_entropy
                entropies[act_space] = -(reduce(
                    learner_policy_entropy,
                    reduction="sum"
                ) / n_actions_taken).detach().cpu().item()

            discounts = (~batch["done"]).float() * flags.discounting
            discounts = discounts.unsqueeze(-1).expand_as(combined_behavior_action_log_probs)
            action_policy_samples = effective_action_counts > 0
            finite_policy_samples = (
                torch.isfinite(combined_behavior_action_log_probs)
                & torch.isfinite(combined_learner_action_log_probs)
            )
            valid_policy_samples = finite_policy_samples & action_policy_samples
            appo_invalid_sample_fraction = (
                (action_policy_samples & ~finite_policy_samples).sum().float()
                / action_policy_samples.sum().clamp(min=1)
            )
            # Stale legality can make a selected action impossible under one policy.
            # Use a neutral V-trace ratio and omit that sample from PPO statistics.
            combined_behavior_action_log_probs = torch.where(
                valid_policy_samples,
                combined_behavior_action_log_probs,
                torch.zeros_like(combined_behavior_action_log_probs),
            )
            combined_learner_action_log_probs = torch.where(
                valid_policy_samples,
                combined_learner_action_log_probs,
                torch.zeros_like(combined_learner_action_log_probs),
            )
            values = learner_outputs["baseline"]
            vtrace_returns = vtrace.from_action_log_probs(
                behavior_action_log_probs=combined_behavior_action_log_probs,
                target_action_log_probs=combined_learner_action_log_probs,
                discounts=discounts,
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value
            )
            td_lambda_returns = td_lambda.td_lambda(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )
            upgo_returns = upgo.upgo(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb
            )

            vtrace_policy_advantages = stabilize_policy_advantages(
                vtrace_returns.pg_advantages,
                flags.normalize_policy_advantages,
                flags.policy_advantage_clip,
            )
            policy_action_normalizer = (
                effective_action_counts.clamp(min=1.)
                if flags.normalize_policy_log_probs_by_actions
                else torch.ones_like(effective_action_counts)
            )
            normalized_learner_action_log_probs = (
                combined_learner_action_log_probs / policy_action_normalizer
            )
            normalized_behavior_action_log_probs = (
                combined_behavior_action_log_probs / policy_action_normalizer
            )
            appo_clip_fraction = torch.zeros((), device=flags.learner_device)
            appo_approx_kl = torch.zeros((), device=flags.learner_device)
            if flags.algo == "appo_vtrace":
                vs_t_plus_1 = torch.cat((vtrace_returns.vs[1:], bootstrap_value.unsqueeze(0)), dim=0)
                appo_advantages = stabilize_policy_advantages(
                    batch["reward"] + discounts * vs_t_plus_1 - values,
                    flags.normalize_policy_advantages,
                    flags.policy_advantage_clip,
                )
                appo_advantages = appo_advantages * role_repair_advantage_weights(
                    batch, appo_advantages, flags
                )
                vtrace_pg_loss, appo_clip_fraction, appo_approx_kl = compute_appo_policy_loss(
                    normalized_learner_action_log_probs,
                    normalized_behavior_action_log_probs,
                    appo_advantages,
                    flags.ppo_clip_ratio,
                    flags.reduction,
                    valid_mask=valid_policy_samples,
                )
            elif flags.algo == "impala":
                vtrace_pg_loss = compute_policy_gradient_loss(
                    normalized_learner_action_log_probs,
                    vtrace_policy_advantages,
                    reduction=flags.reduction
                )
            else:
                raise ValueError(f"Unsupported training algo: {flags.algo}")
            upgo_clipped_importance = torch.minimum(
                vtrace_returns.log_rhos.exp(),
                torch.ones_like(vtrace_returns.log_rhos)
            ).detach()
            upgo_policy_advantages = stabilize_policy_advantages(
                upgo_clipped_importance * upgo_returns.advantages,
                flags.normalize_policy_advantages,
                flags.policy_advantage_clip,
            )
            upgo_pg_loss = (
                compute_policy_gradient_loss(
                    normalized_learner_action_log_probs,
                    upgo_policy_advantages,
                    reduction=flags.reduction
                )
                if flags.algo == "impala"
                else torch.zeros_like(vtrace_pg_loss)
            )
            baseline_loss = compute_baseline_loss(
                values,
                td_lambda_returns.vs,
                reduction=flags.reduction
            )
            actor_critic_normalizer = (
                flags.unroll_length * flags.batch_size * values.shape[-1]
                if flags.normalize_actor_critic_losses
                else 1
            )
            vtrace_pg_loss = vtrace_pg_loss / actor_critic_normalizer
            upgo_pg_loss = upgo_pg_loss / actor_critic_normalizer
            baseline_loss = baseline_loss / actor_critic_normalizer
            teacher_kl_loss = flags.teacher_kl_cost * reduce(
                combined_teacher_kl_loss,
                reduction=flags.reduction
            )
            bc_games = flags.teacher_bc_game_offset + total_games_played
            bc_progress = min(bc_games / max(flags.teacher_bc_anneal_games, 1), 1.0)
            teacher_bc_cost = (
                flags.teacher_bc_cost_start
                + bc_progress * (flags.teacher_bc_cost_end - flags.teacher_bc_cost_start)
            )
            teacher_bc_weighted_loss = torch.zeros((), device=flags.learner_device)
            teacher_bc_weight_sum = 0.0
            for act_space, space_loss in teacher_bc_space_losses.items():
                space_weight = getattr(flags, f"teacher_bc_{act_space}_weight")
                if teacher_bc_accuracies[act_space] or space_loss.detach().item() > 0.:
                    teacher_bc_weighted_loss = teacher_bc_weighted_loss + space_weight * space_loss
                    teacher_bc_weight_sum += space_weight
            teacher_bc_loss = (
                teacher_bc_cost
                * teacher_bc_weighted_loss
                / max(teacher_bc_weight_sum, 1.0)
            )
            if flags.use_teacher:
                teacher_baseline_loss = flags.teacher_baseline_cost * compute_baseline_loss(
                    values,
                    teacher_outputs["baseline"],
                    reduction=flags.reduction
                ) / actor_critic_normalizer
            else:
                teacher_baseline_loss = torch.zeros_like(baseline_loss)
            entropy_loss = flags.entropy_cost * reduce(
                combined_learner_entropy,
                reduction=flags.reduction
            )
            if baseline_only:
                total_loss = baseline_loss + teacher_baseline_loss + teacher_bc_loss
                vtrace_pg_loss, upgo_pg_loss, teacher_kl_loss, entropy_loss = torch.zeros(4) + float("nan")
            else:
                total_loss = (flags.rl_policy_cost * (vtrace_pg_loss + upgo_pg_loss) +
                              baseline_loss +
                              teacher_bc_loss +
                              teacher_kl_loss +
                              teacher_baseline_loss +
                              entropy_loss)
            total_loss = (
                total_loss
                + getattr(flags, "aux_risk_cost", 0.0) * aux_risk_loss
                + gate_intervention_loss
            )

            learning_rates = lr_scheduler.get_last_lr()
            last_lr = learning_rates[0]
            role_bias_lr = learning_rates[-1]
            action_distributions_flat = {
                key[16:]: val[batch["done"]][~val[batch["done"]].isnan()].sum().item()
                for key, val in batch["info"].items()
                if key.startswith("LOGGING_") and "ACTIONS_" in key
            }
            action_distributions = {space: {} for space in ACTION_MEANINGS.keys()}
            for flat_name, n in action_distributions_flat.items():
                space, meaning = flat_name.split(".")
                action_distributions[space][meaning] = n
            action_distributions_aggregated = {}
            for space, dist in action_distributions.items():
                if space == "city_tile":
                    action_distributions_aggregated[space] = dist
                elif space in ("cart", "worker"):
                    aggregated = {
                        a: n for a, n in dist.items() if "TRANSFER" not in a and "MOVE" not in a
                    }
                    aggregated["TRANSFER"] = sum({a: n for a, n in dist.items() if "TRANSFER" in a}.values())
                    aggregated["MOVE"] = sum({a: n for a, n in dist.items() if "MOVE" in a}.values())
                    action_distributions_aggregated[space] = aggregated
                else:
                    raise RuntimeError(f"Unrecognized action_space: {space}")
                n_actions = sum(action_distributions_aggregated[space].values())
                if n_actions == 0:
                    action_distributions_aggregated[space] = {
                        key: float("nan") for key in action_distributions_aggregated[space].keys()
                    }
                else:
                    action_distributions_aggregated[space] = {
                        key: val / n_actions for key, val in action_distributions_aggregated[space].items()
                    }

            total_games_played += batch["done"].sum().item()
            stats = {
                "Env": {
                    key[8:]: val[batch["done"]][~val[batch["done"]].isnan()].mean().item()
                    for key, val in batch["info"].items()
                    if key.startswith("LOGGING_") and "ACTIONS_" not in key
                },
                "Actions": action_distributions_aggregated,
                "Loss": {
                    "algo": flags.algo,
                    "vtrace_pg_loss": vtrace_pg_loss.detach().item(),
                    "appo_clip_fraction": appo_clip_fraction.detach().item(),
                    "appo_approx_kl": appo_approx_kl.detach().item(),
                    "appo_invalid_sample_fraction": appo_invalid_sample_fraction.detach().item(),
                    "upgo_pg_loss": upgo_pg_loss.detach().item(),
                    "baseline_loss": baseline_loss.detach().item(),
                    "teacher_kl_loss": teacher_kl_loss.detach().item(),
                    "teacher_bc_loss": teacher_bc_loss.detach().item(),
                    "teacher_baseline_loss": teacher_baseline_loss.detach().item(),
                    "aux_loss20_loss": aux_risk_loss.detach().item(),
                    "gate_intervention_loss": gate_intervention_loss.detach().item(),
                    "entropy_loss": entropy_loss.detach().item(),
                    "total_loss": total_loss.detach().item(),
                },
                "Entropy": {
                    "overall": sum(e for e in entropies.values() if not math.isnan(e)),
                    **entropies
                },
                "Teacher_KL_Divergence": {
                    "overall": sum(tkld for tkld in teacher_kl_losses.values() if not math.isnan(tkld)),
                    **teacher_kl_losses
                },
                "Teacher_BC_Accuracy": {
                    "overall": (teacher_bc_total_correct / teacher_bc_total_count.clamp(min=1)).item(),
                    **teacher_bc_accuracies,
                },
                "Role_Biases": (
                    learner_model.role_bias_layer.values()
                    if getattr(learner_model, "role_bias_layer", None) is not None
                    else {}
                ),
                "Misc": {
                    "learning_rate": last_lr,
                    "role_bias_learning_rate": role_bias_lr,
                    "teacher_bc_cost": teacher_bc_cost,
                    "vtrace_advantage_abs_max": vtrace_policy_advantages.abs().max().item(),
                    "upgo_advantage_abs_max": upgo_policy_advantages.abs().max().item(),
                    "actor_critic_normalizer": actor_critic_normalizer,
                    "mean_effective_actions": effective_action_counts.mean().item(),
                    "total_games_played": total_games_played,
                    "aux_loss20_precision": aux_loss20_precision.detach().item(),
                    "aux_loss20_recall": aux_loss20_recall.detach().item(),
                    "aux_loss20_positive_rate": aux_loss20_positive_rate.detach().item(),
                    "gate_probability_mean": gate_probability_mean.detach().item(),
                },
            }

            optimizer.zero_grad()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite learner loss at games={total_games_played}: {stats}"
                )
            if flags.use_mixed_precision:
                grad_scaler.scale(total_loss).backward()
                if flags.clip_grads is not None:
                    grad_scaler.unscale_(optimizer)
                    grad_params = [p for p in learner_model.parameters() if p.requires_grad]
                    if aux_risk_head is not None:
                        grad_params += list(aux_risk_head.parameters())
                    bad_gradients = [
                        (name, int((~torch.isfinite(parameter.grad)).sum().item()))
                        for name, parameter in learner_model.named_parameters()
                        if parameter.requires_grad
                        and parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ]
                    if aux_risk_head is not None:
                        bad_gradients.extend(
                            (f"aux_risk_head.{name}", int((~torch.isfinite(parameter.grad)).sum().item()))
                            for name, parameter in aux_risk_head.named_parameters()
                            if parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        )
                    if bad_gradients:
                        raise FloatingPointError(
                            f"Non-finite learner gradient at games={total_games_played}: "
                            f"{bad_gradients[:20]}"
                        )
                    torch.nn.utils.clip_grad_norm_(grad_params, flags.clip_grads)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total_loss.backward()
                if flags.clip_grads is not None:
                    grad_params = [p for p in learner_model.parameters() if p.requires_grad]
                    if aux_risk_head is not None:
                        grad_params += list(aux_risk_head.parameters())
                    torch.nn.utils.clip_grad_norm_(grad_params, flags.clip_grads)
                optimizer.step()
            if lr_scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    lr_scheduler.step()

        # noinspection PyTypeChecker
        actor_model.load_state_dict(learner_model.state_dict())
        return stats, total_games_played


def train(flags):
    # Necessary for multithreading and multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_buffers < flags.num_actors:
        raise ValueError("num_buffers should >= num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size

    if flags.use_teacher:
        teacher_flags = OmegaConf.load(Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))
    else:
        teacher_flags = None

    example_env = create_env(flags, torch.device("cpu"), teacher_flags=teacher_flags)
    example_env_output = add_role_codes_if_enabled(
        example_env.reset(force=True), example_env, create_role_code_builder(flags)
    )
    buffers = create_buffers(
        flags,
        example_env.unwrapped[0].obs_space,
        example_env_output["info"]
    )
    del example_env

    if flags.load_dir:
        checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file, map_location=torch.device("cpu"))
    else:
        checkpoint_state = None

    actor_model = create_model(flags, flags.actor_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        load_training_model_state(actor_model, checkpoint_state["model_state_dict"], flags)
    load_spatial_sidecar_pretrain(actor_model, getattr(flags, "spatial_risk_sidecar_checkpoint", None))
    load_student_pretrain(actor_model, getattr(flags, "student_pretrain_checkpoint", None))
    if getattr(flags, "gate_only_training", False):
        configure_gate_only_training(actor_model, training=False)
    elif getattr(flags, "role_joint_head_training", False):
        configure_role_joint_head_training(actor_model, training=False)
    elif getattr(flags, "role_local_bias_training", False):
        configure_role_local_bias_training(actor_model, training=False)
    elif getattr(flags, "role_local_training", False):
        configure_role_local_training(actor_model, training=False)
    elif getattr(flags, "role_sidecar_training", False):
        configure_role_sidecar_training(actor_model, training=False)
    elif getattr(flags, "role_only_training", False):
        configure_role_only_training(actor_model, training=False)
    actor_model.eval()
    actor_model.share_memory()
    n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    logging.info(f'Training model with {n_trainable_params:,d} parameters.')

    actor_processes = []
    free_queue = mp.Queue()
    full_queue = mp.Queue()

    for i in range(flags.num_actors):
        actor_start = threading.Thread if flags.debug else mp.Process
        actor = actor_start(
            target=act,
            args=(
                flags,
                teacher_flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)

    learner_model = create_model(flags, flags.learner_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        load_training_model_state(learner_model, checkpoint_state["model_state_dict"], flags)
    load_spatial_sidecar_pretrain(learner_model, getattr(flags, "spatial_risk_sidecar_checkpoint", None))
    load_student_pretrain(learner_model, getattr(flags, "student_pretrain_checkpoint", None))
    if getattr(flags, "gate_only_training", False):
        configure_gate_only_training(learner_model, training=True)
    elif getattr(flags, "role_joint_head_training", False):
        configure_role_joint_head_training(learner_model, training=True)
    elif getattr(flags, "role_local_bias_training", False):
        configure_role_local_bias_training(learner_model, training=True)
    elif getattr(flags, "role_local_training", False):
        configure_role_local_training(learner_model, training=True)
    elif getattr(flags, "role_sidecar_training", False):
        configure_role_sidecar_training(learner_model, training=True)
    elif getattr(flags, "role_only_training", False):
        configure_role_only_training(learner_model, training=True)
    else:
        learner_model.train()
    learner_model = learner_model.share_memory()
    aux_risk_head = None
    if getattr(flags, "gate_only_training", False) and getattr(flags, "use_aux_risk", False):
        raise ValueError("gate_only_training currently requires use_aux_risk=false.")
    if getattr(flags, "use_aux_risk", False):
        aux_risk_head = AuxiliaryRiskHead(
            in_channels=learner_model.base_out_channels,
            hidden_channels=getattr(flags, "aux_risk_hidden_channels", 128),
            dropout=getattr(flags, "aux_risk_dropout", 0.10),
        ).to(flags.learner_device)
        if (
            checkpoint_state is not None
            and checkpoint_state.get("aux_risk_head_state_dict") is not None
        ):
            aux_risk_head.load_state_dict(checkpoint_state["aux_risk_head_state_dict"])
        aux_risk_head.train()
        aux_risk_head = aux_risk_head.share_memory()
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    if getattr(flags, "role_joint_head_training", False):
        learner_model.role_joint_backbone_blocks = int(
            getattr(flags, "role_joint_backbone_blocks", 0)
        )
        configure_role_joint_head_training(learner_model, training=True)
        backbone_params = [
            parameter
            for module in list(learner_model.base_agent.base_model.children())[1:][
                -int(getattr(flags, "role_joint_backbone_blocks", 0)):
            ]
            for parameter in module.parameters()
            if parameter.requires_grad
        ] if int(getattr(flags, "role_joint_backbone_blocks", 0)) > 0 else []
        policy_params = [
            parameter
            for module in (
                learner_model.base_agent.actor_base,
                learner_model.base_agent.actor,
                learner_model.base_agent.baseline_base,
                learner_model.base_agent.baseline,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        local_params = [
            parameter for parameter in learner_model.role_local_adapter.parameters()
            if parameter.requires_grad
        ]
        role_params = [
            parameter for parameter in learner_model.role_bias_layer.parameters()
            if parameter.requires_grad
        ]
        optimizer_params = backbone_params + policy_params + local_params + role_params
        optimizer_groups = [
            {"params": backbone_params, "lr": float(flags.backbone_tail_learning_rate)},
            {"params": policy_params, "lr": float(flags.policy_head_learning_rate)},
            {"params": local_params, "lr": float(flags.role_local_learning_rate)},
            {"params": role_params, "lr": float(flags.role_bias_learning_rate)},
        ]
        optimizer_kwargs = dict(flags.optimizer_kwargs)
        optimizer_kwargs.pop("lr", None)
        optimizer = flags.optimizer_class(optimizer_groups, **optimizer_kwargs)
    elif getattr(flags, "role_local_bias_training", False):
        local_params = [
            parameter for parameter in learner_model.role_local_adapter.parameters()
            if parameter.requires_grad
        ]
        role_params = [
            parameter for parameter in learner_model.role_bias_layer.parameters()
            if parameter.requires_grad
        ]
        optimizer_params = local_params + role_params
        optimizer_groups = [
            {"params": local_params, "lr": float(flags.role_local_learning_rate)},
            {"params": role_params, "lr": float(flags.role_bias_learning_rate)},
        ]
        optimizer_kwargs = dict(flags.optimizer_kwargs)
        optimizer_kwargs.pop("lr", None)
        optimizer = flags.optimizer_class(optimizer_groups, **optimizer_kwargs)
    elif getattr(flags, "role_sidecar_training", False):
        sidecar_params = [
            parameter
            for module in (learner_model.spatial_risk_sidecar, learner_model.intervention_gate)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        role_params = [
            parameter for parameter in learner_model.role_bias_layer.parameters()
            if parameter.requires_grad
        ]
        optimizer_params = sidecar_params + role_params
        optimizer_groups = [
            {"params": sidecar_params, "lr": float(flags.sidecar_learning_rate)},
            {"params": role_params, "lr": float(flags.role_bias_learning_rate)},
        ]
        optimizer_kwargs = dict(flags.optimizer_kwargs)
        optimizer_kwargs.pop("lr", None)
        optimizer = flags.optimizer_class(optimizer_groups, **optimizer_kwargs)
    else:
        optimizer_params = [p for p in learner_model.parameters() if p.requires_grad]
        if aux_risk_head is not None:
            optimizer_params += list(aux_risk_head.parameters())
        optimizer = flags.optimizer_class(
            optimizer_params,
            **flags.optimizer_kwargs
        )
    if not optimizer_params:
        raise ValueError("No trainable parameters were selected for the optimizer.")
    if checkpoint_state is not None and not flags.weights_only:
        try:
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        except ValueError as exc:
            logging.warning(f"Could not load optimizer state; starting optimizer fresh. Reason: {exc}")

    # Load teacher model for KL loss
    if flags.use_teacher:
        if (flags.teacher_kl_cost <= 0. and flags.teacher_baseline_cost <= 0.
                and flags.teacher_bc_cost_start <= 0. and flags.teacher_bc_cost_end <= 0.):
            raise ValueError("Teacher enabled but all teacher loss costs are disabled")
        teacher_model = create_model(
            flags,
            flags.learner_device,
            teacher_model_flags=teacher_flags,
            is_teacher_model=True
        )
        teacher_model.load_state_dict(
            torch.load(
                Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file,
                map_location=torch.device("cpu")
            )["model_state_dict"]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        if flags.teacher_kl_cost > 0.:
            logging.warning(f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_kl_cost to 0.")
        if flags.teacher_baseline_cost > 0.:
            logging.warning(f"flags.teacher_baseline_cost is {flags.teacher_baseline_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_baseline_cost to 0.")
        flags.teacher_kl_cost = 0.
        flags.teacher_baseline_cost = 0.
        flags.teacher_bc_cost_start = 0.
        flags.teacher_bc_cost_end = 0.

    training_step_budget = (
        flags.total_steps
        if flags.total_games is None
        else flags.total_games * flags.expected_steps_per_game
    )

    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b, training_step_budget) / training_step_budget
        scaled_pct_complete = pct_complete * (1. - min_pct)
        return 1. - scaled_pct_complete

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=flags.use_mixed_precision,
        init_scale=float(flags.amp_init_scale),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint_state is not None and not flags.weights_only:
        if "scheduler_state_dict" in checkpoint_state:
            scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

    step, total_games_played, stats = 0, 0, {}
    if checkpoint_state is not None and not flags.weights_only:
        if "step" in checkpoint_state.keys():
            step = checkpoint_state["step"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'step' saved. Starting at step 0.")
        if "total_games_played" in checkpoint_state.keys():
            total_games_played = checkpoint_state["total_games_played"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'total_games_played' saved. Starting at step 0.")

    def training_complete():
        if flags.total_games is not None:
            return total_games_played >= flags.total_games
        return step >= flags.total_steps

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, total_games_played, stats
        timings = prof.Timings()
        while not training_complete():
            timings.reset()
            full_batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
                actor_processes=actor_processes,
            )
            if flags.batch_size < flags.n_actor_envs:
                batches = split_buffers(full_batch, flags.batch_size, dim=1, contiguous=True)
            else:
                batches = [full_batch]
            for batch in batches:
                stats, total_games_played = learn(
                    flags=flags,
                    actor_model=actor_model,
                    learner_model=learner_model,
                    teacher_model=teacher_model,
                    aux_risk_head=aux_risk_head,
                    batch=batch,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=scheduler,
                    total_games_played=total_games_played,
                    baseline_only=step / (t * b) < flags.n_value_warmup_batches,
                )
                with lock:
                    step += t * b
                    if not flags.disable_wandb:
                        wandb.log(stats, step=step)
            timings.time("learn")
        if learner_idx == 0 and flags.log_detailed_stats:
            logging.info(f"Batch and learn timing statistics: {timings.summary()}")

    for m in range(flags.num_buffers):
        free_queue.put(m)

    learner_threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,)
        )
        thread.start()
        learner_threads.append(thread)

    def checkpoint(checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(getattr(flags, "checkpoint_dir", ".")) / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_path)
        if flags.log_detailed_stats:
            logging.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "aux_risk_head_state_dict": (
                    aux_risk_head.state_dict() if aux_risk_head is not None else None
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "total_games_played": total_games_played,
            },
            checkpoint_path + ".pt",
        )
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "aux_risk_head_state_dict": (
                    aux_risk_head.state_dict() if aux_risk_head is not None else None
                ),
            },
            checkpoint_path + "_weights.pt"
        )

    timer = timeit.default_timer
    monitor_start_time = timer()
    last_progress_step = step
    last_progress_time = monitor_start_time
    try:
        next_checkpoint_step = int(flags.checkpoint_freq)
        while not training_complete():
            start_step = step
            start_time = timer()
            time.sleep(flags.console_log_interval)

            # Save every checkpoint_freq learner steps.
            if flags.checkpoint_freq and step >= next_checkpoint_step:
                cp_path = str(step).zfill(int(math.log10(max(training_step_budget, 1))) + 1)
                checkpoint(cp_path)
                next_checkpoint_step += int(flags.checkpoint_freq)

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / (t * b) / (timer() - start_time)
            if step > last_progress_step:
                last_progress_step = step
                last_progress_time = timer()
            if (
                flags.actor_start_timeout_seconds
                and step == 0
                and timer() - monitor_start_time > flags.actor_start_timeout_seconds
            ):
                raise RuntimeError(
                    "No actor rollout arrived before actor_start_timeout_seconds="
                    f"{flags.actor_start_timeout_seconds}. "
                    "Check actor startup errors, node/Lux dimensions, and PATH. "
                    f"Actors: {format_actor_status(actor_processes)}"
                )
            if (
                flags.training_stall_timeout_seconds
                and step > 0
                and timer() - last_progress_time > flags.training_stall_timeout_seconds
            ):
                raise RuntimeError(
                    "No learner progress for training_stall_timeout_seconds="
                    f"{flags.training_stall_timeout_seconds}. "
                    f"Last progress step: {last_progress_step}. "
                    f"Actors: {format_actor_status(actor_processes)}"
                )
            if flags.log_detailed_stats:
                logging.info(
                    "Steps %d @ %.1f SPS / %.1f BPS. Stats:\n%s",
                    step,
                    sps,
                    bps,
                    pprint.pformat(stats),
                )
            else:
                env_stats = stats.get("Env", {})
                loss_stats = stats.get("Loss", {})
                bc_stats = stats.get("Teacher_BC_Accuracy", {})
                misc_stats = stats.get("Misc", {})
                role_stats = stats.get("Role_Biases", {})
                total_loss = float(loss_stats.get("total_loss", float("nan")))

                def format_env_stat(name: str, decimals: int) -> str:
                    value = float(env_stats.get(name, float("nan")))
                    return f"{value:.{decimals}f}" if math.isfinite(value) else "--"

                logging.info(
                    "Games %d/%s | steps %d | %.1f SPS | loss %s | reward %s | "
                    "bc %.3f W%.0f/C%.0f/K%.0f @ %.3f | aux %.3f P%.2f/R%.2f/+%.2f | "
                    "APPO invalid %.3f | city tiles %s | research %s | role B%.3f/F%.3f/T%.3f",
                    total_games_played,
                    flags.total_games if flags.total_games is not None else "--",
                    step,
                    sps,
                    f"{total_loss:.3f}" if math.isfinite(total_loss) else "INVALID",
                    format_env_stat("mean_cumulative_rewards", 3),
                    float(loss_stats.get("teacher_bc_loss", 0.0)),
                    100.0 * float(bc_stats.get("worker", 0.0)),
                    100.0 * float(bc_stats.get("city_tile", 0.0)),
                    100.0 * float(bc_stats.get("cart", 0.0)),
                    float(misc_stats.get("teacher_bc_cost", 0.0)),
                    float(loss_stats.get("aux_loss20_loss", 0.0)),
                    float(misc_stats.get("aux_loss20_precision", 0.0)),
                    float(misc_stats.get("aux_loss20_recall", 0.0)),
                    float(misc_stats.get("aux_loss20_positive_rate", 0.0)),
                    float(loss_stats.get("appo_invalid_sample_fraction", 0.0)),
                    format_env_stat("city_tiles_final", 1),
                    format_env_stat("research_points", 1),
                    float(role_stats.get("builder_build_city_bias", 0.0)),
                    float(role_stats.get("firefighter_move_bias", 0.0)),
                    float(role_stats.get("firefighter_transfer_bias", 0.0)),
                )
    except KeyboardInterrupt:
        # Try checkpointing and joining actors then quit.
        logging.info(
            "Training interrupted after %d games and %d steps.",
            total_games_played,
            step,
        )
    else:
        for thread in learner_threads:
            thread.join()
        logging.info(
            "Learning finished after %d games and %d steps.",
            total_games_played,
            step,
        )
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
        cp_path = str(step).zfill(int(math.log10(max(training_step_budget, 1))) + 1)
        checkpoint(cp_path)
