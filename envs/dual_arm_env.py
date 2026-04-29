import numpy as np
import pandas as pd
import mujoco.viewer
import mujoco as mj
import gymnasium as gym
from gymnasium import spaces
import os

class DualArmEnv(gym.Env):
    metadata = {"render_modes":["human","rgb_array"],"render_fps": 50}
    
    LEFT_JOINTS = ["left_shoulder_pan","left_shoulder_lift", "left_elbow_flex",
                   "left_wrist_flex", "left_wrist_roll", "left_gripper"]
    RIGHT_JOINTS = ["right_shoulder_pan","right_shoulder_lift", "right_elbow_flex",
                   "right_wrist_flex", "right_wrist_roll", "right_gripper"]

    def __init__(self, render_mode = None, max_episode_steps=500):
        super().__init__()

        # ------- Load MuJoCo Model ------------------
        xml_path = os.path.join(os.path.dirname(__file__),"../scene/SO101/dual_scene.xml")
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self.max_epsiode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        # ------- Cache Frequently-used IDs ------------------
        # End-effector sites defined in each arm's XML
        self.left_ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "left_gripperframe")
        self.right_ee_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "right_gripperframe")

        # Actuator Indices - position actuators expect radian targets
        self.left_act_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                               for n in self.LEFT_JOINTS]
        self.right_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                               for n in self.RIGHT_JOINTS]
        self.all_act_ids   = self.left_act_ids + self.right_act_ids

        # Joint indices (for qpos/qvel reads)
        self.left_jnt_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                               for n in self.LEFT_JOINTS]
        self.right_jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                               for n in self.RIGHT_JOINTS]

        # ----- Control limits from XML (position actuators use ctrlrange) ---------
        self.ctrl_low  = self.model.actuator_ctrlrange[self.all_act_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[self.all_act_ids, 1]

        # ----- Spaces -------------------
        # Actions: taret joint angles in radians for all 12 joints
        self.action_space = spaces.Box(
            low = self.ctrl_low.astype(np.float32),
            high = self.ctrl_high.astype(np.float32),
            dtype=np.float32
        )

        # Observations:
        # 12 joint positions + 12 joint velocities
        # + left EE pos (3) + right EE pos (3)
        # + left EE quat (4) + right EE quat (4)
        # + object pos (3) + goal pos (3)

        obs_dim = 12+12+3+3+4+4+3+3
        self_observation_space = spaces.Box(
            low = -np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ---- Object/goal (add to scene.xml worldbody)
        self.obj_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "object")
        self.goal_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "goal_site")

        # ----- Home position (all joints at 0) ------------
        self.home_qpos = np.zeros(12)

        # ------ Renderer ------------
        self._viewer = None
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mj.Renderer(self.model, height=480, width=640)



