# ===================================================================
# This is used for configuring PPO for a Jetcobot Manipulator using RLlib


# ===================================================================
import yaml
import os
import numpy as np

from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from envs.manipulation import Manipulation

CONFIG_PATH = Path(__file__).resolve().parents[2]/"config.yaml"

def load_yaml_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["experiment_name"] = cfg["training"]["experiment_name"].replace(
        "${algorithm}", cfg["algorithm"]
    )
    return cfg


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

def _apply_common(config, cfg:dict):
    res_cfg = cfg["resources"]
    return (
        config
        .framework("torch")
        .environment(env="Manipulation-v0", env_config=cfg["env"], clip_actions=True)
        .env_runners(
            num_env_runners         = res_cfg["num_env_runners"],     
            num_envs_per_env_runner = res_cfg["num_envs_per_env_runner"],
            num_cpus_per_env_runner = res_cfg["num_cpus_per_worker"],
        )
        .learners(num_gpus_per_learner = res_cfg["num_gpus"])
    )

# ----------PPO Config Builder-----------------------
def build_algo_config(algorithm: str | None = None, overrides: dict | None = None) -> PPOConfig:
    """
    Builds a PPOConfig or SACConfig from config.yaml.

    'algorithms' defaults to config.yaml's top-level algorithm field
    'overrides' lets an Optuna trail replace hyperparameters for whichever algorithm is selected

    """
    cfg = load_yaml_config()
    algorithm = algorithm or cfg["algorithm"]
    algo_cfg = dict(cfg[algorithm])

    if overrides:
        algo_cfg.update(overrides)
    
    if algorithm == "ppo":
        config = _apply_common(PPOConfig(), cfg).training(
            lr                            = algo_cfg["lr"],
            gamma                         = algo_cfg["gamma"],
            clip_param                    = algo_cfg["clip_param"],
            entropy_coeff                 = algo_cfg["entropy_coeff"],
            vf_loss_coeff                 = algo_cfg["vf_loss_coeff"],
            kl_coeff                      = algo_cfg["kl_coeff"],
            num_epochs                    = algo_cfg["num_epochs"],
            train_batch_size_per_learner  = algo_cfg["train_batch_size_per_learner"],
            minibatch_size                = algo_cfg["minibatch_size"],
        )
    
    elif algorithm == "sac":
            config = _apply_common(SACConfig(), cfg).training(
            lr                            = algo_cfg["lr"],
            gamma                         = algo_cfg["gamma"],
            tau                           = algo_cfg["tau"],
            train_batch_size_per_learner  = algo_cfg["train_batch_size_per_learner"],
            target_entropy                = algo_cfg["target_entropy"],
            n_step                        = algo_cfg["n_step"],
            replay_buffer_config          = algo_cfg["replay_buffer_config"],
            num_steps_sampled_before_learning_starts = algo_cfg["num_steps_sampled_before_learning_starts"],
        )
 
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}' — expected 'ppo' or 'sac'")
 
    return config
