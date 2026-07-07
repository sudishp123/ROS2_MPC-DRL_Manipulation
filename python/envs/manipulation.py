# Gymnasium Environment for JetCobot DRL-NMPC Manipulation

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


import os, json, sys

from env_creation import MakeEnv
from python.mpc.nmpc_controller import NMPCController

# Directories
_THIS_FILE = os.path.abspath(__file__)
_ENVS_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_ENVS_DIR))

class Manipulation(gym.Env):
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
    [15] -> Distance to nearest obstacle
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

        #TODO - initialize and add counters for these
        self._qdot_norm_prev = 0.0
        self.step_count = 0
        #-------------------------------------------------

        
        self.obstacle_radius = params["obstacle_settings"]["size_high"]

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

        # overhead camera
        self.cam_width = 84
        self.cam_height = 84
        self.cam_id = self.model.camera("overhead_camera").id
        self._renderer = mj.Renderer(self.model, height=self.cam_height, width=self.cam_height)

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

        self.nmpc = NMPCController(N=5, dt=0.001*frame_skip, ds=self.d_safe)

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mj.mj_forward(self.model, self.data)

    def _get_image(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self.cam_id)
        return self._renderer.render().copy()
        
    # action space:
    def _set_action_space(self):
        """
        DRL action = NMPC weight vector
        NMPC maps w -> joint velocity commands at each control step
        """
        self.action_low = np.zeros(12, dtype=np.float32)
        self.action_high = np.ones(12, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float32
        )

    # initialize observation space:
    def _set_observation_space(self):
        """
        Observation Layout:
            [0:3] -> Pos error
            [3:9] -> Joint Positions q
            [9: 15] -> Joint Velocities qdot
            [16] -> nearest obstacle distance
        """

        obs_states = (
            ["ex", "ey", "ez"]
        +   [f"q{i}" for i in range(6)]
        +   [f"q_dot{i}" for i in range(6)]
        +   ["d_obs"]
        )

        self._obs_space_size = len(obs_states)
        for i, name in enumerate(obs_states):
            setattr(self, f"obs_{name}_idx", i)
        
        low = np.full(self._obs_space_size, -np.pi, dtype = np.float32)
        high = np.full(self._obs_space_size, np.pi, dtype=np.float32)

        # pos error
        low[0:3] = -0.8 ; high[0:3] = 0.8

        # joint limits from params
        j_low = np.array([-3.05, -1.57, -1.57, -1.57, -3.05, -1.57])
        j_high = np.array([3.05, 1.57, 1.57, 1.57, 3.05, 1.57])
        low[3:9] = j_low; high[3:9] = j_high

        # joint velocities
        qdot_limit = 4.0
        low[9:15] = -qdot_limit; high[9:15] = qdot_limit

        # obstacle distance
        low[15] = 0.0; high[15] = 2.0

        self.observation_space = gym.spaces.Dict({
        "state": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        "image": gym.spaces.Box(low=0, high=255, shape=(self.cam_height, self.cam_width, 3),  dtype=np.uint8)
        })


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
        self.nearest_obstacle = min(self._compute_link_obstacle_distances())

        state = np.concatenate([pos_error, q, qdot, [self.nearest_obstacle]]).astype(np.float32)

        return {
            "state": state,
            "image": self._get_image()
        }
    
    # nearest obstacle distance

    
    # step function
    def step(self, action: np.ndarray):
        """
        Unpack NMPC weights from DRL action

        Solve NMPC -> qdot_cmd that is used in this function for reward calculation       
        """
        action = np.clip(action, self.action_low, self.action_high)

        theta_s = action[0:3]
        theta_r = action[3:9]
        theta_g = action[9:12]
        self.nmpc.set_drl_params(theta_s, theta_r, theta_g)

        q               = self.data.sensordata[self._q_slice]
        p_des           = self.data.xpos[self.target_body_id]
        ee_pos          = self.data.sensordata[self._ee_pos_slice]

        T_obs           = min(
            [self.data.xpos[self.model.body(f"obstacle_{i}").id] for i in range(1, self.n_obstacles + 1)],
            key=lambda pos:np.linalg.norm(pos-ee_pos)
            ).copy()
        qdot_cmd, info  = self.nmpc.solve(q, p_des, T_obs)

        self.data.ctrl[:] = np.clip(qdot_cmd, -4.0, 4.0)
        mj.mj_step(self.model, self.data, nstep=self.frame_skip)

        nobs= self._get_obs()
        pos_err = nobs["state"][0:3]
        d_pos = float(np.linalg.norm(pos_err))

        goal_cond = (d_pos < self.pos_threshold)
        collision_cond = self.nearest_obstacle < self.collision_thresh
        term = goal_cond or collision_cond

        if goal_cond:
            rew = self.rew_target_scale
        elif collision_cond:
            rew = self.rew_collision_scale
        else:
            rew_dist = (self.d_pos_last - d_pos) * self.rew_dist_scale
            rew_effort = -float(np.sum(qdot_cmd**2)) * self.rew_effort_scale
            rew = rew_dist + rew_effort + self.rew_time

        info = {}
        if term:
            info["is_success"] = bool(goal_cond)
            info["collision"]  = bool(collision_cond)

        self.d_pos_last = d_pos
        self.action_last = action

        if self.render_mode == "human":
            self.render()
        
        return nobs, rew, term, False, info

    # reset:
    def reset(self, seed=None, options=None):
        self.episode_counter +=1
        self._qdot_norm_prev = 0.0
        self.step_count = 0

        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)

        should_randomize = self.is_eval or (
            self.episode_counter % self.randomization_freq == 0
        )
        ob = self._reset_model(randomize=should_randomize)

        if self.render_mode == "human":
            self.render()
        return ob, {}
    
    def _reset_model(self, randomize: bool = False) -> np.ndarray:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if randomize:
            # randomize joint home position slightly
            qpos[:6] += self.np_random.uniform(-0.1, 0.1, size=6)
            qpos[:6] = np.clip(qpos[:6],
                               [-3.05, -1.57, -1.57, -1.57, -3.05, -1.57],
                                [ 3.05,  1.57,  1.57,  1.57,  3.05,  1.57])
            
            # randomize target position
            new_target = self.np_random.uniform(
                low = self.target_bound_low, high= self.target_bound_high
            )
            target_id = self.model.body("target").mocapid[0]
            self.data.mocap_pos[target_id] = new_target

            # randomize obstacle positions
            new_obs_pos = self._sample_obstacle_positions(new_target, self.n_obstacles)
            for i, pos in enumerate(new_obs_pos):
                obs_id = self.model.body(f"obstacle_{i+1}").mocapid[0]
                self.data.mocap_pos[obs_id] = pos

        self.set_state(qpos, qvel)
        mj.mj_forward(self.model, self.data)

        ob = self._get_obs()
        self.d_pos_last = float(np.linalg.norm(ob["state"][0:3]))
        self.action_last = np.zeros(self.action_space.shape)
        return ob
        
    def render(self):
        if self.mujoco_renderer is not None:
            return self.mujoco_renderer.render(self.render_mode)
        
    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        self._renderer.close()   

    # helper functions:
    def _compute_reward(self, pos_err_norm, poss_err_norm_prev, qdot, g_hat):
        """
        r1: relative change in EE pose error (setpoint control)
        r2: goal bonus when error < threshold
        r3: collision avoidance per link
        r4: time penalty
        r5: relative change in joint velocity norm
        """

        eps = 1e-6 # Avoids sudden jumps vs absolute error
        r1 =  (pos_err_norm - poss_err_norm_prev)/ (pos_err_norm + eps)

        # r2 - goal bonus
        if pos_err_norm < 0.03:
            r2 = -500.0
        else:
            r2 = 0.0
        
        # r3 - collision avoidance per link
        # g_hat: array of min distances per link
        r3 = 0.0
        for g in g_hat:
            if g >= self.d_safe:
                r3 += -10.0 * (g**2)           # reward for being safe
            elif g > 0:
                r3 += 10.0 * (g**2)            # penalty for proximity
            else:
                r3 += 1000.0                   # collision penalty

        # r4 - time penalty
        r4 = self.step_count                  # penalize longer episdoes

        # r5 - relative joint velocity change
        qdot_norm = np.linalg.norm(qdot)
        r5 = (qdot_norm - self._qdot_norm_prev)/(self._qdot_norm_prev + eps)
        self._qdot_norm_prev = qdot_norm

        # Reward linear combination - based on paper
        C1, C2, C3, C4, C5 = 1000.0, 1.0, 1.0, 1.0, 100.0
        R = C1*r1 + C2*r2 + C3*r3 + C4*r4 + C5*r5
        return R

    def _compute_link_obstacle_distances(self) -> np.ndarray:
        """
        Computes minimum distance between each robot link and each obstacle,
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
                        self.model.body(f"obstacle_{obs_idx}").id
                    ].copy()

                    dist = self._segment_sphere_distance(T1, T2, T, D, r)
                    if dist < min_dist_over_obstacles:
                        min_dist_over_obstacles = dist
                    g_hat[link_idx] = min_dist_over_obstacles
            except:
                g_hat[link_idx] = np.inf # body not found - skip
        return g_hat
    
    # Calculating distance between link j and the obstacle denoted
    def _segment_sphere_distance(self,
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
        










