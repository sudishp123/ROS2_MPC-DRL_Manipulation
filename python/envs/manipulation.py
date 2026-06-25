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
    [3:9] -> Joint Positions (q)
    [9:15] -> Joint Velocities (qdot)
    [15] -> Distance to nearest obstalce
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
        self.rew_effort_scale =    rs.get("rew_effort_scale", -0.1)
        self.rew_time            = rs.get("rew_time", -0.5)

        # thresholds
        self.pos_threshold    = 0.02
        self.collision_thresh = params["obstacle_settings"]["allowance"]
        self.d_safe           = 0.15

        # episode counters
        self.episode_counter    = 0
        rand                    = randomization_options or {}
        self.randomization_freq = rand.get("randomization_freq", 1)
        self.reset_randomize    = False

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
        
    # action space:
    def _set_action_space(self):
        """
        DRL action = NMPC weight vector
        NMPC maps w -> joint velocity commands at each control step
        """
        self.action_low = np.zeros(4, dtype=np.float32)
        self.action_high = np.ones(4, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float32
        )

    # initialize observation space:
    def _set_observation_space(self):
        """
        Observation Layout:
            [0:3] -> Pos error
            [4:9] -> Joint Positions q
            [10: 15] -> Joint Velocities qdot
            [16] -> nearest obstacle distance
        """

        obs_states = (
            ["ex", "ey", "ez"]
        +   [f"q{i}" for i in range(6)]
        +   [f"q_dot{i}" for i in range(6)]
        +   ["d_obs"]
        )

        self.obs_space_size = len(obs_states)
        for i, name in enumerate(obs_states):
            setattr(self, f"obs_{name}_idx", i)
        
        low = np.full(self._obs_space_size, -np.pi, dtype = np.float32)
        high = np.full(self._obs_space_size, np.pi, dtype=np.float32)

        # pos error
        low[0:3] = -0.8 ; high[0:3] = 0.8

        # joint limits from params
        j_low = np.array([-3.05, -1.57, -1.57, -3.05, -1.57])
        j_high = np.array([3.05, 1.57, 1.57, 3.05, 1.57])
        low[3:9] = j_low; high[3:9] = j_high

        # joint velocities
        qdot_limit = 4.0
        low[9:15] = -qdot_limit; high[9:15] = qdot_limit

        # obstacle distance
        low[15] = 0.0; high[16] = 2.0

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    # obtain observations:
    def _get_obs(self) -> np.ndarray:
        sd = self.data.sensordata

        # joint state from sensors:
        q = sd[self._q_slice].astype(np.float32)
        qdot = sd[self._qdot_slice].astype(np.float32)

        ## EE pose from sensors
        ee_pos = sd[self._ee_pos_slice]
        
        # target position
        tgt_pos = self.data.xpos[self.target_body_id]

        # Cartesian Position Error
        pos_error = (tgt_pos - ee_pos).astype(np.float32)

        # nearest obstacle distance
        self.nearest_obstacle = self._nearest_obstacle_dist()

        self._obs_buffer[0:3] = pos_error
        self._obs_buffer[3:9] = q
        self._obs_buffer[9:15] = qdot
        self._obs_buffer[15] = self.nearest_obstacle

        return self._obs_buffer
    
    # step function
    

    # helper functions:
    def _compute_link_obstalce_distances(self) -> np.ndarray:
        """
        Computes minimum distance between each robot link and each obstalce,
        returning minimum over obstacles per link

        Link geometry: capsule approximated as line segment
        Obstacle geometry: sphere with center T and radius r
        """

        LINK_SEGMENTS = [
            ("1_Link", "2_Link", 0.070),
            ("2_Link", "3_Link", 0.060),
            ("3_Link", "4_Link", 0.056),
            ("4_Link", "5_Link", 0.050),
            ("5_Link", "6_Link", 0.040),
            ("6_Link", "jiazhua_Link", 0.040),
        ]

        g_hat = np.full(len(LINK_SEGMENTS), np.inf)

        for link_idx, (body_start, body_end, diameter) in enumerate(LINK_SEGMENTS):
            try:
                T1 = self.data.xpos[self.model.body(body_start).id].copy()
                T2 = self.data.xpos[self.model.body(body_end).id].copy()
                D = diameter
                r = self.obstacle_radius

                min_dist_over_obstacles = np.inf

                for obs_idx in range(1, self.n_obstacles + 1):
                    T = self.data.xpos[
                        self.model.body(f"obstalce_{obs_idx}").id
                    ].copy()

                    dist = self._segment_sphere_distance(T1, T2, T, D, r)
                    if dist < min_dist_over_obstacles:
                        min_dist_over_obstacles = dist
            except:
                g_hat[link_idx] = np.inf # body not found - skip
        return g_hat
    
    # Calculating distance between link j and the obstacle denoted
    def _segment_sphere_distance(
            T1: np.ndarray,
            T2: np.ndarray,
            T: np.ndarray,
            D: float,
            r: float
    ) -> float:
        """
        Minimum distance between:
            - Cylindrical Link segment T1 -> T2 with diameter D
            - spherical obstacle centered at T with radius r

        Returns scalar clearance distance (positive = safe, negative = collision)
        """

        segment = T2-T1
        seg_len_sq = np.dot(segment, segment)

        # Projection factor t
        t = np.dot(T-T1, segment)/seg_len_sq

        if 0.0 <= t <= 1.0:
            cross = np.cross(T-T1, T-T2)
            dist_to_axis = np.linalg.norm(cross) / np.sqrt(seg_len_sq)
            return float((dist_to_axis) - (D/2.0) - r)
        
        else:
            # projection falls outside segment - use closer endpoint
            d1 = np.linalg.norm(T-T1)
            d2 = np.linalg.norm(T-T2)
            return float(min(d1, d2) - D/2.0 - r)









