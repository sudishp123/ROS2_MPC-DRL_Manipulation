from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

import torch
import gymnasium as gym
import envs.manipulation
import numpy as np
import os, json
from datetime import datetime
from tqdm import tqdm
import optuna

# Ignore user warnings (for creating a new folder to save policies):
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def init_model(hyperparameters: dict,
               reward_scale: dict = {str, float},
               randomization_options: dict = {str, int},
               obstacle_options: dict = {str, int},
               normalize: bool = False):
    
    # environment vectorization settings
    n_proc = 24

    # max episode steps:
    max_episode_steps = 3_000

    # Vectorized Environment
    env = make_vec_env("Manipulation-v0",
                       n_envs = n_proc,
                       env_kwargs = {"max_episode_steps": max_episode_steps,
                                     "reward_scale_options": reward_scale,
                                     "randomization_options": randomization_options,
                                     "obstacle_options": obstacle_options,
                                     "render_mode": "rgb_array",
                                     },
                                     vec_env_cls=SubprocVecEnv,
                                     vec_env_kwargs = dict(start_method = "forkserver"),
                                     seed = 42,
                       )

    if normalize:
        env = VecNormalize(env, norm_obs = True, norm_reward = True)

    model = SAC(policy = hyperparameters["policy"],
                env = env,
                buffer_size             = hyperparameters["buffer_size"],
                batch_size              = hyperparameters["batch_size"],
                tau                     = hyperparameters["tau"],
                gamma                   = hyperparameters["gamma"],
                ent_coef                = hyperparameters["ent_coef"],
                train_freq              = hyperparameters["train_freq"],
                learning_starts         = hyperparameters["learning_starts"],
                target_update_interval  = hyperparameters["target_update_interval"],
                gradient_steps          = hyperparameters["gradient_steps"],
                target_entropy          = hyperparameters["target_entropy"],
                action_noise            = NormalActionNoise(mean = np.zeros(env.action_space.shape), sigma = hyperparameters["action_noise_std"] * np.ones(env.action_space.shape)),
                verbose                 = hyperparameters["verbose"],
                device                  = "cuda" if hyperparameters["gpu"] else "cpu",
                policy_kwargs           = hyperparameters["policy_kwargs"],
                tensorboard_log         = hyperparameters["tensorboard_log"]
            )

    # apply custom learning rate
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr = hyperparameters["actor_lr"])
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr = hyperparameters["critic_lr"])




env = gym.make("Manipulation-v0")
s_size = env.observation_space.shape
a_size = env.action_space.shape

# ___________ Observation Space (Size -> 19 Elements) ___________
#     [0:3] -> Pos error
#     [3: 6] -> Quat Error
#     [6:12] -> Joint Positions q
#     [12: 18] -> Joint Velocities qdot
#     [18] -> nearest obstacle distance

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation


# ___________ Action Space (Size -> 15 Elements) ___________
#     [0:6] -> Theta_s (diagonal weights on cartesian and orientation error)
#     [6: 9] -> Theta_r (diagonal weights on joint velocities)
#     [9] -> Collision Avoidance Margin per link (N_links)

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

env_id = "Manipulation-v0"
env = make_vec_env(env_id, n_envs=4)
