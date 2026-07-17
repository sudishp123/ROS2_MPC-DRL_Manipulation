# ===================================================================
# python/train/tune_optuna.py
#   python train/tune_optuna.py            # uses config.yaml's `algorithm`
#   python train/tune_optuna.py --algo sac # overrides it for this run
# ===================================================================
import argparse
from pathlib import Path

import yaml
import ray
from ray import tune
from ray.tune import RunConfig, TuneConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from training.algo_config import build_algo_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_search_space(spec: dict) -> dict:
    space = {}
    for name, s in spec.items():
        if "choices" in s:
            space[name] = tune.choice(s["choices"])
        elif s.get("log"):
            space[name] = tune.loguniform(s["low"], s["high"])
        else:
            space[name] = tune.uniform(s["low"], s["high"])
    return space


def make_trainable(algorithm: str):
    def trainable(trial_config: dict):
        cfg = load_cfg()
        algo_config = build_algo_config(algorithm=algorithm, overrides=trial_config)
        algo = algo_config.build()

        for i in range(cfg["training"]["max_iterations"]):
            result = algo.train()
            tune.report({
                "episode_return_mean": result["env_runners"]["episode_return_mean"],
                "training_iteration": i,
            })
            if i % cfg["training"]["checkpoint_freq"] == 0:
                algo.save()

        algo.stop()
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default=None,
                         help="Overrides config.yaml's `algorithm` for this sweep.")
    args = parser.parse_args()

    cfg = load_cfg()
    algorithm = args.algo or cfg["algorithm"]
    opt_cfg = cfg["optuna"]
    algo_search_space = opt_cfg["search_space"][algorithm]

    # Each trial's own algo_config.build() spins up num_env_runners actors,
    # each wanting num_cpus_per_env_runner CPUs — Tune has no way to know
    # that unless told explicitly, so it under-reserves (1 CPU/trial default)
    # and RLlib's placement group request inside the trial then fails.
    res_cfg = cfg["resources"]
    cpus_per_trial = (
        res_cfg["num_env_runners"] * res_cfg["num_cpus_per_env_runner"]
        + 1  # local/driver process
    )
    gpus_per_trial = res_cfg["num_gpus"]

    ray.init()

    trainable_with_resources = tune.with_resources(
        make_trainable(algorithm),
        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    )

    searcher = OptunaSearch(metric=opt_cfg["metric"], mode=opt_cfg["mode"])
    scheduler = ASHAScheduler(metric=opt_cfg["metric"], mode=opt_cfg["mode"])

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=build_search_space(algo_search_space),
        tune_config=TuneConfig(
            search_alg=searcher,
            scheduler=scheduler,
            num_samples=opt_cfg["n_trials"],
            max_concurrent_trials=opt_cfg["max_concurrent_trials"],
        ),
        run_config=RunConfig(
            name=f"{cfg['training']['experiment_name']}_{algorithm}".replace("${algorithm}", algorithm),
            storage_path=cfg["training"]["storage_path"],
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(opt_cfg["metric"], opt_cfg["mode"])
    print(f"[{algorithm}] Best config:", best.config)
    print(f"[{algorithm}] Best metrics:", best.metrics)

    # Persist the winning hyperparameters so they can be reloaded later
    # without re-running the sweep: build_algo_config(overrides=<this file's content>)
    out_dir = Path(__file__).resolve().parents[1] / "best_configs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"best_{algorithm}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump({
            "algorithm": algorithm,
            "metric": opt_cfg["metric"],
            "metric_value": best.metrics[opt_cfg["metric"]],
            "hyperparameters": best.config,
        }, f, sort_keys=False)
    print(f"[{algorithm}] Saved best hyperparameters to {out_path}")


if __name__ == "__main__":
    main()