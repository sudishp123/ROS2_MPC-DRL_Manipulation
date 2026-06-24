# Gymnasium Environment for JetCobot DRL-NMPC Manipulation

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import os, json

from env_creation import MakeEnv

# Directories
_THIS_FILE = os.path.abspath(__file__)
_ENVS_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_ENVS_DIR))

class Manipulation(MujocoEnv):
    """
    class constructor to initialize the environment (Mujoco model and data), the observation space, and renderer

    Architecture:
    DRL Policy -> NMPC cost weights (action)
    NMPC -> joint velocity commands (qdot)
    MuJoCo -> forward simulation

    Observation:
    [0:3] -> Cartesian EE position error (target - ee)
    [3:6] -> Cartesian EE orientation aerror (axis-angle)
    [6:12] -> Joint Positions (q)
    [12:18] -> Joint Velocities (qdot)
    [18] -> Distance to nearest obstalce
    """
    metadata = {"render_modes": ["human","rgb_array"], "render_fps":500}

    def __init__(self,
                 json_file: str="environment_params.json",
                 frame_skip: int = 5,
                 render_mode: str = "rgb_array",
                 width: int = 480,
                 height: int = 480,
                 reward_scale_options: dict[str, float] | None = None,
                 randomization_options: dict[str,float] | None = None,
                 obstacle_options: dict[str, int] = {"n_obstacles":0},
                 visual_options: dict[int, bool] | None = None,
                 is_eval: bool = False,
                 n_obstacles: int = 3,
                 ):
        """
        Arguments:
        json_file:                a string that contains the name of the environment parameters json file, which contains
                                  compiler info, visual settings, and element settings (ground, wall, light, robot, target)
        
        render_mode:            a string that specifies the MuJoCo renderer mode, such as ``human``, ``rgb_array``, or ``None``
        width:                  width of the rendering window
        height:                 height of the rendering window
        reward_scale_options:   a dictionary containing the value for each of the reward scale 
        """
        # load the simulation parameters:
        json_path = os.path.join(_ENVS_DIR, json_file)
        with open (json_path) as f:
            params = json.load(f)

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.is_eval = is_eval
        self.n_obstacles = n_obstacles

        # reward scales
        rs =                       reward_scale_options or {}
        self.rew_target_scale =    rs.get("rew_target_scale", 200.0)
        self.rew_collision_scale = rs.get("rew_collision_scale", -100.0)
        self.rew_dist_scale =      rs.get("rew_dist_scale", 10.0)
        self.rew_ori_scale =       rs.get("rew_ori_scale", 5.0)
        self.rew_effort_scale =    rs.get("rew_effort_scale", -0.1)
        self.rew_time =            rs.get("rew_time", -0.5)

        # thresholds
        self.pos_threshold =    0.02
        self.ori_threshold =    0.1
        self.collision_thresh = params["obstacle_settings"]["allowance"]
        self.d_safe =           0.15

        # episode counters
        self.episode_counter = 0
        rand =                    randomization_options or {}
        self.randomization_freq = rand.get("randomization_freq", 1)
        self.reset_randomize =    False

        # workspace bounds
        # keep targets reachable: JetCobot max reach ~0.40 m
        self.target_bound_low = np.array([0.1, -0.25, 0.1])
        self.target_bound_high = np.array([0.35, 0.25, 0.35])

        # build MuJoco scene
        initial_target = np.array([0.25, 0.0, 0.25])
        initial_obs_positions = self._sample_obstacle_positions(
            target_pos=initial_target, n=self.n_obstacles
        )

        env = MakeEnv(params)
        env.make_env(
            robot_pos = [0.0, 0.0],
            target_pos = initial_target.tolist(),
            obs_pos = initial_obs_positions,
        )

        self.model = env.model
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mj.MjData(self.model)

        # cache body/sensor IDs
        self.ee_body_id = self.model.body("6_Link").id
        self.target_body_id = self.model.body("target").id
        self.joint_ids = [self.model.joint]

        # sensor layout set by add_sensors():
        self._q_slice = slice(0,6)
        self._qdot_slice = slice(6, 12)
        self._ee_pos_slice = slice(12, 15)
        self._ee_quat_slice = slice(15, 19)

        # spaces
        self._set_action_space()
        self._set_observation_space()

        # pre-allocate buffers
        self._obs_buffer = np.zeros(self._obs_space_size, dtype=np.float32)

        # init qpos/qvel snapshots
        mj.mj_forward(self.model, self.data)
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # tracking state:
        self.d_pos_last = np.inf
        self.action_last = np.zeros(self.action_space.shape)
        self.nearest_obstacle = np.inf

        # render
        self.mujoco_renderer = None
        if self.render_mode == "human":
            self.mujoco_renderer = MujocoRenderer(
                self.model, self.data,
                width = self.width, height=self.height
            )
            


        

