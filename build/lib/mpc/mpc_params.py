from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2]/"config/yaml"

@dataclass
class MPCParams:
    N: int
    dt: float          
    ds: float
    q_min: np.ndarray
    q_max: np.ndarray
    qdot_min: np.ndarray
    qdot_max: np.ndarray
    S_N: np.ndarray            
    eta: np.ndarray
    eta_f: np.ndarray
    init_theta_s: np.ndarray
    init_theta_r: np.ndarray
    init_theta_g: np.ndarray
    obstacle_radius: float
    link_radius: float
    ipopt_max_iter: int
    ipopt_tol: float

def load_mpc_params() -> MPCParams:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["mpc"]
    nq = 6
    return MPCParams(
        N               = cfg["N"],
        dt              = cfg["dt"],
        ds              = cfg["ds"],
        q_min           = np.array(cfg["q_min"]),
        q_max           = np.array(cfg["q_max"]),
        qdot_min        = np.full(nq, cfg["qdot_min"]),
        qdot_max        = np.full(nq, cfg["qdot_max"]),
        S_N             = np.diag(cfg["S_N_diag"]),
        eta             = np.array([cfg["eta"]]),
        eta_f           = np.array([cfg["eta_f"]]),
        init_theta_s    = np.array(cfg["init_theta_s"]),
        init_theta_r    = np.array(cfg["init_theta_r"]),
        init_theta_g    = np.array(cfg["init_theta_g"]),
        obstacle_radius = cfg["obstacle_radius"],
        link_radius     = cfg["link_radius"],
        ipopt_max_iter  = cfg["ipopt_max_iter"],
        ipopt_tol       = cfg["ipopt_tol"],

    )
