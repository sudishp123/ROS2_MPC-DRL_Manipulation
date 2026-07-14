```bashrobot_drl_nmpc/
├── 📂 assets/
├── jetcobot
    ├── meshes/
    └── urdf/
├── SO-ARM101
├── 📂 python/  
    ├── 📂 envs/
    │   ├── __init__.py
    │   ├── env_creation.py                     # Generating random envs for training
    │   ├── manipulation.py                     # Gymnasium Environment
    │   └── environment_params.json             # Parameters for the env_creation.py 
    ├── 📂 mpc/
        ├── 📂 jacobian/
        |    ├── __init__.py
        |    ├── jacobian.py
        |    └── jetcobot_jacobian_casadi.py    # Formulating a CASADi function for the Jacobian Matrix of Jetcobot
    │   ├── __init__.py
    │   ├── nmpc_controller.py    # HILO-MPC wrapper
    │   └── mpc_params.py         # Q, R, horizon defaults
    ├── 📂 training/
    │   ├── train.py              # RLlib entry point
    │   └── rllib_config.py       # Algorithm + env config
    └── config.yaml               # Top-level hyperparameters for optuna study and training DRL
```
