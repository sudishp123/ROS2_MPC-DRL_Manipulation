# ===================================================================
# Runs an Optuna study over hyperparameters via Ray Tune.
# tensorboard --logdir <training.storage_path>/<experiment_name>

# ===================================================================
import argparse
from pathlib import Path
import yaml
import ray

from ray import tune
from ray.tune import RunConfig, TuneConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from training.algo_config import build_ppo_config, load_yaml_config

CONFIG_PATH = Path(__file__).resolve().parents[1]/"config.yaml"

cfg = load_yaml_config()

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
    def trainable(trial_config:dict):
        algo_config = build_ppo_config(algorith=algorithm, overrides=trial_config)
        algo = algo_config.build()

        for i in range(cfg["training"]["max_iterations"]):
            result = algo.train()
            tune.report({
                "episode_return_mean": result["env_runners"]["episode_return_mean"],
                "training_iteration": i
            })
            if i % cfg["training"]["checkpoint_freq"] == 0:
                algo.save()
        
        algo.stop()
    return trainable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default=None,
                        help="Overrides config.yaml's 'algorith' for this sweep")
    args = parser.parse_args()

    algorithm = args.algo or cfg["algorithm"]
    opt_cfg = cfg["optuna"]
    algo_search_space = opt_cfg["search_space"][algorithm]

    ray.init()

    searcher    = OptunaSearch(metric=opt_cfg["metric"], mode=opt_cfg["mode"])
    scheduler   = ASHAScheduler(metri=opt_cfg["metric"], mode=opt_cfg["mode"])

    tuner = tune.Tuner(
        make_trainable(algorithm),
        param_space = build_search_space(algo_search_space),
        tune_config=TuneConfig(
            search_alg=searcher,
            scheduler=scheduler,
            num_samples=opt_cfg["n_trials"],
            max_concurrent_trials=opt_cfg["max_concurrent_trials"]
        ),
        run_config=RunConfig(
            name=f"{cfg['training']['experiment_name']}_{algorithm}".replace("${algorithm}",algorithm),
            storage_path=cfg["training"]["storage_path"]
        )
    )

    results = tuner.fit()
    best = results.get_best_result(opt_cfg["metric"], opt_cfg["mode"])
    print(f"[{algorithm}] Best Config: ", best.config)
    print(f"[{algorithm}] Best Metrics: ", best.metrics)

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

        