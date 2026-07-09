# ===================================================================
# This is used for configuring PPO for a Jetcobot Manipulator using RLlib


# ===================================================================
import os
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.manipulation import Manipulation
from train.custom_model import ManipulationModel

# ----------Specifying Env Info to Rllib config-----------------------
def make_env(cfg):
    return Manipulation(
        json_file               = cfg.get("json_file",    "environment_params.json"),
        frame_skip              = cfg.get("frame_skip",                           5),
        render_mode             = cfg.get("render_mode",                "rgb_array"),
        n_obstacles             = cfg.get("n_obstacles",                          3),
        reward_scale_options    = cfg.get("reward_scale_options",              None),
        randomization_options   = cfg.get("randomization_options",             None),
        is_eval                 =   cfg.get("is_eval",                        False),
    )

register_env("Manipulation-v0", make_env)

# ----------Custom Model Registration-----------------------
ModelCatalog.register_custom_model("manipulation_model", ManipulationModel)

# ----------Environment Config-----------------------
ENV_CONFIG={
     "json_file"            : "environment_params.json",
    "frame_skip"           : 5,          # Δt_sim = 0.001 * 5 = 5ms per MuJoCo step
    "render_mode"          : "rgb_array",
    "n_obstacles"          : 1,          # start with 1, match paper's single dynamic obstacle
    "is_eval"              : False,
    "reward_scale_options" : {
        "rew_target_scale"   : 200.0,
        "rew_collision_scale": -100.0,
        "rew_dist_scale"     : 10.0,
        "rew_effort_scale"   : -0.1,
        "rew_time"           : -0.5,
    },
    "randomization_options": {
        "randomization_freq" : 1,        # randomize every episode
    },
}

# ----------PPO Config Builder-----------------------
def build_ppo_config() -> PPOConfig:
    """
    Builds the RLlib PPO config

    """

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env             =   "Manipulation-v0",
            env_config      =   ENV_CONFIG,
            clip_actions    =   True,
        )
        .rollouts(
            num_rollout_workers     = 4,
            num_envs_per_worker     = 1,
            rollout_fragment_length = 0,
            batch_mode              = "truncate_episodes",
        )
        .resources(
            num_gpus            = 1,
            num_cpus_per_worker = 2,
        )
        .training(
            lr              = 1e-3,
            gamma           = 0.99,
            clip_param      = 0.2,
            entropy_coeff   = 0.01,
            vf_loss_coeff   = 0.5,
            kl_coeff        = 0.2,
            num_sgd_iter    = 10,
            train_batch_size = 1600,
            sgd_minibatch_size = 128,
        )
    )
    

