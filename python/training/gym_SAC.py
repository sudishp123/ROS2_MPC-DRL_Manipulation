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

def main():
    cfg = load_yaml_config()
    ray.init()

    algo = build_algo_config(algorithm = "sac").build()

    for i in range(cfg["training"]["max_iterations"]):
        result = algo.train()
        print(
            f"iter {i:4d} | "
            f"return_mean = {result["env_runners"]["episode_return_mean"]:.2f}"
        )
        if i % cfg["training"]["checkpoint_freq"] == 0:
            ckpt = algo.save()
            print(f" checkpoint saved: {ckpt.checkpoint.path}")

if __name__ == "__main__":
    main()