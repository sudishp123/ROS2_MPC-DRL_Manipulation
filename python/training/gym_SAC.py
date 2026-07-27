# ===================================================================
# python/train/gym_PPO.py
# Single training run (PPO or SAC, whichever config.yaml's `algorithm`
# is set to) — no Optuna sweep. Use this for a normal training run or
# to smoke-test the env/model before launching tune_optuna.py.
#
#   python training/gym_PPO.py
#   tensorboard --logdir <config.yaml training.storage_path>
# ===================================================================
import argparse
from pathlib import Path

import yaml
import ray

from training.algorithms import build_algo_config, load_yaml_config
from ray import tune

def main():
    cfg = load_yaml_config()
    algo_config = build_algo_config(algorithm = "sac")
    ray.init()

    tuner = tune.Tuner(
        "SAC",
        param_space=algo_config,
        run_config=tune.RunConfig(
            name="jetcobot_sac_run1",
            storage_path = cfg["training"]["storage_path"],
            stop=cfg["training"]["max_iterations"],
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=25,
                checkpoint_at_end=True,
            ),
        ),
    )

    result_grid = tuner.fit()  

if __name__ == "__main__":
    main()