# ===================================================================
# python/train/gym_PPO.py
# Single training run (PPO or SAC, whichever config.yaml's `algorithm`
# is set to) — no Optuna sweep. Use this for a normal training run or
# to smoke-test the env/model before launching tune_optuna.py.
#
#   python train/gym_PPO.py
#   tensorboard --logdir <config.yaml training.storage_path>
# ===================================================================
import argparse
from pathlib import Path

import yaml
import ray

from training.algo_config import build_algo_config, load_yaml_config

def load_best_overrides(algorithm: str) -> dict:
    path = Path(__file__).resolve().parents[1]/"best_configs"/f"best_{algorithm}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved best config at {path}. Run optuna_tune.py --algo {algorithm} first."
        )
    with open(path) as f:
        return yaml.safe_load(f)["hyperparameters"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true",
                         help="Train using the best hyperparameters saved by optuna_tune.py.")
    args = parser.parse_args()
 
    cfg = load_yaml_config()
    ray.init()
 
    overrides = load_best_overrides(cfg["algorithm"]) if args.best else None
    algo = build_algo_config(algorithm = "ppo",overrides=overrides).build()
 
    for i in range(cfg["training"]["max_iterations"]):
        result = algo.train()
        print(
            f"iter {i:4d} | "
            f"return_mean={result['env_runners']['episode_return_mean']:.2f}"
        )
        if i % cfg["training"]["checkpoint_freq"] == 0:
            ckpt = algo.save()
            print(f"  checkpoint saved: {ckpt.checkpoint.path}")

    algo.stop()


if __name__ == "__main__":
    main()