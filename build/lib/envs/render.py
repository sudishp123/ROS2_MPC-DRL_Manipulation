import json
from envs.env_creation import MakeEnv

# load params
with open("/home/sudhishp/ROS2_MPC+DRL_Manipulation/python/envs/environment_params.json") as f:
    params = json.load(f)

env = MakeEnv(params)

env.make_env(
    robot_pos=[0.0, 0.0],
    target_pos = [0.3, 0.0, 0.3],
    obs_pos = [[0.4, 0.2, 0.1],
               [-0.3, 0.3, 0.1]]
)

env.render()