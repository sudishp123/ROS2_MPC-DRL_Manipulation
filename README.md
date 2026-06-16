```bashrobot_drl_nmpc/
├── assets/
    ├── urdf/
│     └── jetcobot.urdf
├── envs/
│   ├── __init__.py
│   ├── environment_rand.py   # Generating random envs for training
│   └── reward.py             # Reward shaping
├── mpc/
│   ├── __init__.py
│   ├── nmpc_controller.py    # HILO-MPC wrapper
│   └── mpc_params.py         # Q, R, horizon defaults
├── training/
│   ├── train.py              # RLlib entry point
│   └── rllib_config.py       # Algorithm + env config
├── utils/
│   ├── urdf_parser.py        # Extract joint info from URDF
│   └── logging.py
├── eval/
│   └── evaluate.py           # Run trained policy
└── config.yaml               # Top-level hyperparameters
```
