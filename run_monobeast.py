from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import types
import warnings

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*The version_base parameter is not specified.*")
warnings.filterwarnings("ignore", message=".*Future Hydra versions will no longer change working directory.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays is extremely slow.*")

# Gym 0.26 imports gym_notices only to print its end-of-life banner. The Lux
# environment still depends on Gym, so suppress the banner without changing Gym.
gym_notices = types.ModuleType("gym_notices")
gym_notices_notices = types.ModuleType("gym_notices.notices")
gym_notices_notices.notices = {}
gym_notices.notices = gym_notices_notices
sys.modules.setdefault("gym_notices", gym_notices)
sys.modules.setdefault("gym_notices.notices", gym_notices_notices)

# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import kaggle_environments

import hydra
import logging
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
with redirect_stderr(io.StringIO()):
    from lux_ai.torchbeast.monobeast import train


os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("total_games", None)
    flags.setdefault("expected_steps_per_game", 400)
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})
    flags.setdefault("env_configuration", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("amp_init_scale", 16.0)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 10.)
    flags.setdefault("checkpoint_dir", ".")
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)
    flags.setdefault("teacher_bc_cost_start", 0.0)
    flags.setdefault("teacher_bc_cost_end", 0.0)
    flags.setdefault("teacher_bc_anneal_games", 1)
    flags.setdefault("teacher_bc_game_offset", 0)
    flags.setdefault("teacher_bc_worker_weight", 1.0)
    flags.setdefault("teacher_bc_city_tile_weight", 1.0)
    flags.setdefault("teacher_bc_cart_weight", 1.0)
    flags.setdefault("rl_policy_cost", 1.0)
    flags.setdefault("algo", "impala")
    flags.setdefault("ppo_clip_ratio", 0.2)
    flags.setdefault("reference_policy_kl_cost", flags.get("teacher_kl_cost", 0.0))
    # Backward-compatible alias. New configs should use reference_policy_kl_cost.
    flags["teacher_kl_cost"] = flags["reference_policy_kl_cost"]
    flags.setdefault("role_bias_training_enabled", False)
    flags.setdefault("role_only_training", False)
    flags.setdefault("role_sidecar_training", False)
    flags.setdefault("role_local_bias_training", False)
    flags.setdefault("role_local_adapter_enabled", False)
    flags.setdefault("role_local_training", False)
    flags.setdefault("role_joint_head_training", False)
    flags.setdefault("role_local_hidden_channels", 16)
    flags.setdefault("role_local_max_delta", 0.25)
    flags.setdefault("sidecar_learning_rate", flags.get("optimizer_kwargs", {}).get("lr", 1e-6))
    flags.setdefault("role_bias_learning_rate", 5e-6)
    flags.setdefault("role_local_learning_rate", 5e-6)
    flags.setdefault("policy_head_learning_rate", 1e-6)
    flags.setdefault("backbone_tail_learning_rate", 2e-7)
    flags.setdefault("role_joint_backbone_blocks", 0)
    flags.setdefault("role_small_map_policy_weight", 0.25)
    flags.setdefault("role_hard_window_start", 25)
    flags.setdefault("role_hard_window_weight", 2.0)
    flags.setdefault("normalize_policy_advantages", False)
    flags.setdefault("policy_advantage_clip", None)
    flags.setdefault("normalize_actor_critic_losses", False)
    flags.setdefault("normalize_policy_log_probs_by_actions", False)
    flags.setdefault("training_spatial_augmentation", False)
    flags.setdefault("training_player_swap_probability", 0.0)
    flags.setdefault("training_player_swap_per_sample", False)
    flags.setdefault("use_aux_risk", False)
    flags.setdefault("aux_risk_cost", 0.0)
    flags.setdefault("aux_risk_horizon", 20)
    flags.setdefault("aux_risk_hidden_channels", 128)
    flags.setdefault("aux_risk_dropout", 0.10)
    flags.setdefault("aux_risk_pos_weight_scale", 1.0)
    flags.setdefault("aux_risk_threshold", 0.40)
    flags.setdefault("spatial_risk_sidecar_checkpoint", None)
    flags.setdefault("student_pretrain_checkpoint", None)
    flags.setdefault("sidecar_freeze_base_agent", True)

    # Model params
    flags.setdefault("use_index_select", True)
    if flags.get("use_index_select"):
        logging.info("index_select disables padding_index and is equivalent to using a learnable pad embedding.")

    # Reloading previous run params
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_value_warmup_batches", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)
    flags.setdefault("log_config", False)
    flags.setdefault("log_detailed_stats", False)
    flags.setdefault("console_log_interval", 30.0)
    flags.setdefault("actor_start_timeout_seconds", 180.0)
    flags.setdefault("rollout_queue_timeout_seconds", 300.0)
    flags.setdefault("training_stall_timeout_seconds", 600.0)
    flags.setdefault("resume_from_local_config", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="resume_config", version_base=None)
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()
    if flags.get("resume_from_local_config", False) and Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    if flags.log_config:
        logging.info(OmegaConf.to_yaml(flags, resolve=True))
    else:
        logging.info(
            "Training %s | games=%s steps=%s | map=%s | actors=%s envs=%s | teacher=%s",
            flags.name,
            flags.total_games,
            flags.total_steps,
            flags.env_configuration,
            flags.num_actors,
            flags.n_actor_envs,
            flags.use_teacher,
        )
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            config=vars(flags),
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
