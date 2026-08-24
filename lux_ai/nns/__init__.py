import logging
import torch
from torch import nn
from typing import Optional

from .models import BasicActorCriticNetwork
from .in_blocks import ConvEmbeddingInputLayer
from .attn_blocks import ViTBlock, RPSA, GPSA
from .conv_blocks import ResidualBlock, ParallelDilationResidualBlock
from .unet import UNET
from ..lux_gym import create_flexible_obs_space, obs_spaces
from ..utility_constants import MAX_BOARD_SIZE


def create_model(
        flags,
        device: torch.device,
        teacher_model_flags: Optional = None,
        is_teacher_model: bool = False
) -> nn.Module:
    obs_space = create_flexible_obs_space(flags, teacher_model_flags)
    if isinstance(obs_space, obs_spaces.MultiObs):
        if is_teacher_model:
            obs_space_prefix = "teacher_"
        else:
            obs_space_prefix = "student_"
        assert obs_space_prefix in obs_space.named_obs_spaces, f"{obs_space_prefix} not in {obs_space.named_obs_spaces}"
    else:
        obs_space_prefix = ""

    return _create_model(
        teacher_model_flags if is_teacher_model else flags,
        device,
        obs_space,
        obs_space_prefix
    )


def _create_model(
        flags,
        device: torch.device,
        obs_space: obs_spaces.BaseObsSpace,
        obs_space_prefix: str
):
    act_space = flags.act_space()
    conv_embedding_input_layer = ConvEmbeddingInputLayer(
        obs_space=obs_space.get_obs_spec(),
        embedding_dim=flags.embedding_dim,
        out_dim=flags.hidden_dim,
        n_merge_layers=flags.n_merge_layers,
        sum_player_embeddings=flags.sum_player_embeddings,
        use_index_select=flags.use_index_select,
        obs_space_prefix=obs_space_prefix,
        feature_embedding_dims=getattr(flags, "feature_embedding_dims", None),
    )
    if flags.model_arch == "conv_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "pd_conv_model":
        logging.warning("Dilation is slow for some Pytorch/CUDNN versions.")
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ParallelDilationResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "unet_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            UNET(
                n_blocks_per_reduction=flags.n_blocks_per_reduction,
                in_out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                # Residual block kwargs
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            )
        )
    elif flags.model_arch == "RPSA_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=RPSA(
                    in_channels=flags.hidden_dim,
                    heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "GPSA_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=GPSA(
                    dim=flags.hidden_dim,
                    n_heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model = BasicActorCriticNetwork(
        base_model=base_model,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=1,
        rescale_value_input=flags.rescale_value_input,
        learned_intervention_gate_enabled=getattr(flags, "learned_intervention_gate_enabled", False),
        learned_intervention_gate_hidden_channels=getattr(
            flags, "learned_intervention_gate_hidden_channels", 32
        ),
        learned_intervention_gate_initial_probability=getattr(
            flags, "learned_intervention_gate_initial_probability", 1e-3
        ),
        learned_intervention_gate_hard_threshold=getattr(
            flags, "learned_intervention_gate_hard_threshold", 0.80
        ),
        learned_intervention_gate_temperature=getattr(
            flags, "learned_intervention_gate_temperature", 1.0
        ),
    )
    model = model.to(device=device)
    if getattr(flags, "role_local_adapter_enabled", False):
        from ..rl_agent.role_assignment import RoleCityBiasParams
        from ..rl_agent.role_conditioned_local_adapter import (
            RoleConditionedLocalAdapter,
            RoleLocalAgentWrapper,
        )
        from ..rl_agent.trainable_role_bias import TrainableRoleBiasLayer
        model = RoleLocalAgentWrapper(
            model,
            TrainableRoleBiasLayer(
                RoleCityBiasParams.from_mapping(getattr(flags, "role_bias_initial", None))
            ).to(device),
            RoleConditionedLocalAdapter(
                feature_channels=flags.hidden_dim,
                hidden_channels=getattr(flags, "role_local_hidden_channels", 16),
                max_delta=getattr(flags, "role_local_max_delta", 0.25),
            ).to(device),
        ).to(device)
    elif getattr(flags, "spatial_risk_sidecar_enabled", False):
        from ..rl_agent.learned_intervention_gate import SidecarLogitDeltaGate
        from ..rl_agent.sidecar_agent_wrapper import SidecarAgentWrapper
        from ..rl_agent.spatial_risk_sidecar import SpatialRiskAttentionSidecar
        role_bias_layer = None
        if getattr(flags, "role_bias_training_enabled", False):
            from ..rl_agent.role_assignment import RoleCityBiasParams
            from ..rl_agent.trainable_role_bias import TrainableRoleBiasLayer
            role_bias_layer = TrainableRoleBiasLayer(
                RoleCityBiasParams.from_mapping(getattr(flags, "role_bias_initial", None))
            ).to(device)
        model = SidecarAgentWrapper(
            model,
            SpatialRiskAttentionSidecar(
                in_channels=flags.hidden_dim,
                attention_dim=getattr(flags, "spatial_risk_attention_dim", 64),
                num_heads=getattr(flags, "spatial_risk_num_heads", 4),
                pool_size=getattr(flags, "spatial_risk_pool_size", 8),
            ).to(device),
            SidecarLogitDeltaGate(
                hidden_channels=getattr(flags, "sidecar_gate_hidden_channels", 16),
            ).to(device),
            risk_gate_enabled=getattr(flags, "risk_gate_enabled", False),
            risk_thresholds=getattr(flags, "risk_thresholds", None),
            safe_expansion_threshold=getattr(flags, "safe_expansion_threshold", 0.80),
            logit_bias_lambda=getattr(flags, "logit_bias_lambda", 4.0),
            freeze_base_agent=getattr(flags, "sidecar_freeze_base_agent", True),
            role_bias_layer=role_bias_layer,
        ).to(device)
    return model
