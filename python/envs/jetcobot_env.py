# imports:
import mujoco as mj
import mujoco.viewer
import numpy as np
import os

# define main class for creating the environment:
class MakeEnv:
    """
    This class is for creating the environment using the python API for MuJoCo
    """

    # constructor:
    def __init__(self,
                 params : dict):
        """
        this is the constructor for the class, which does the instantiation of the environment.

        : param params: a dict that contains the relevant parameters for creating the environment
        : type params: dict
        """

        # add params to self:
        self.params = params

        ### OBJECT PARAMETERS ###
        # env settings:
        self.env_name = params["env_settings"]["name"]

        # compiler settings:
        self.compiler_angle = params["compiler_settings"]["compiler_angle"]

        # option settings:
        self.timestep = params["option_settings"]["timestep"]
        self.integrator = params["option_settings"]["integrator"]
        self.gravity = params["option_settings"]["gravity"]

        # default settings
        self.joint_damping = params["default_settings"]["joint_damping"]

        # visual settings
        



    