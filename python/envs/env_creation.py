# imports:
import mujoco as mj
import mujoco.viewer
import numpy as np
import os

# maps a "collision.type" string in the JSON to the corresponding MuJoCo geom enum:
_COLLISION_GEOM_TYPE = {
    "sphere": mj.mjtGeom.mjGEOM_SPHERE,
    "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
    "box": mj.mjtGeom.mjGEOM_BOX,
    "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
}

def _collision_geom_size(spec:dict) -> list:
    """
    Translate a JSON ``collision`` block into the ``size`` triple MuJoCo expects
    for the given geom type. MuJoCo's size semantics differ per primitive:
    sphere: [radius, 0, 0]
    capsule: [radius, half_length, 0]
    cylinder: [radius, half_length, 0]
    box: [half_x, half_y, half_z] 
    """

    # Collision Geometries - simpler than the urdf file for physics based contact dynamics
    gtype = spec["type"]
    if gtype == "sphere":
        return [spec["radius"], 0.0, 0.0]
    elif gtype in ("capsule", "cylinder"):
        return [spec["radius"], spec["half_length"], 0.0]
    elif gtype == "box":
        return list(spec["half_extents"])
    else:
        raise ValueError(f"Unsupported Collision geom type: {gtype}")


# define main class for creating the environment:
class MakeEnv:
    """
    This class is for creating the environment with robot manipulato9r using the python API for MuJoCo.

    This file is responsible -> static strucutre of joint angles and obstacle position
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

         # paths:
        self.base_path = self.base_path = os.getcwd()                               # Goes to Root of the folder
        self.mesh_dir  = os.path.join(self.base_path, 'assets','jetcobot','meshes') # Can change jetcobot to another folder name

        ### OBJECT PARAMETERS ###
        # camera settings:
        self.camera_name = params["camera_settings"]["name"]
        self.camera_pos  = params["camera_settings"]["pos"]

        # compiler settings:
        self.compiler_angle = params["compiler_settings"]["compiler_angle"]

        # default settings:
        self.joint_damping = params["default_settings"]["joint_damping"]
        self.joint_armature = params["default_settings"]["joint_armature"]

        # env settings:
        self.env_name = params["env_settings"]["name"]

        # ground plane settings:
        self.ground_name = params["ground_settings"]["name"]
        self.ground_type = mj.mjtGeom.mjGEOM_PLANE
        self.ground_contype = params["ground_settings"]["contype"]
        self.ground_conaffinity = params["ground_settings"]["conaffinity"]
        self.ground_actual_length = params["ground_settings"]["internal_length"]
        self.ground_z_spacing = params["ground_settings"]["z_spacing"]
        self.ground_size = [self.ground_actual_length, self.ground_actual_length, self.ground_z_spacing]
        self.ground_pos = params["ground_settings"]["pos"]
        self.ground_rgba = params["ground_settings"]["rgba"]

        # light settings:
        self.light_name     = params["light_settings"]["name"]
        self.light_pos      = params["light_settings"]["pos"]
        self.light_diffuse  = params["light_settings"]["diffuse"]
        self.light_specular = params["light_settings"]["specular"]
        self.light_ambient  = params["light_settings"]["ambient"]
        self.light_shadows  = params["light_settings"]["castshadows"]

        # obstacle settings:
        self.obstacle_counter = 0
        self.allowance = params["obstacle_settings"]["allowance"]
        self.obstacle_thickness = params["obstacle_settings"]["thickness"]
        self.obstacle_thickness = params["obstacle_settings"]["thickness"]
        self.obstacle_height = params["obstacle_settings"]["height"] # Diameter of the spherical obstacle high = 0.1
        self.obstacle_size_low = params["obstacle_settings"]["size_low"] # Diameter of the spherical obstacle low = 0.25
        self.obstacle_size_high = params["obstacle_settings"]["size_high"]

        # option settings:
        self.timestep = params["option_settings"]["timestep"]
        self.integrator = params["option_settings"]["integrator"]
        self.gravity = params["option_settings"]["gravity"]

        # robot footprint settings:
        self.robot_footprint_name = params["robot_footprint_settings"]["name"]
        self.robot_footprint_radius = params["robot_footprint_settings"]["radius"]
        self.robot_footprint_height = params["robot_footprint_settings"]["height"]
        self.robot_footprint_contype = params["robot_footprint_settings"]["contype"]
        self.robot_footprint_conaffinity = params["robot_footprint_settings"]["conaffinity"]
        self.robot_footprint_rgba = params["robot_footprint_settings"]["rgba"]

        # robot settings:
        self.robot_base_mesh = params["robot_settings"]["base_mesh"]
        self.robot_base_rgba = params["robot_settings"]["base_rgba"]
        self.link_rgba = params["robot_settings"]["link_rgba"]
        self.tool_rgba = params["robot_settings"]["tool_rgba"]
        self.qdot_limit = params["robot_settings"]["qdot_limit"]
        self._joint_data = params["robot_settings"]["revolute_joints"]
        self._fixed_links = params["robot_settings"]["fixed_links"]

        # robot - derived joint names
        self.joint_names = [jd["joint_name"] for jd in self._joint_data]
        self.q_min = np.array([jd["lower"] for jd in self._joint_data])
        self.q_max = np.array([jd["upper"] for jd in self._joint_data])

        # skybox settings:
        self.skybox_name    = params["skybox_settings"]["name"]
        self.skybox_type    = mj.mjtTexture.mjTEXTURE_SKYBOX
        self.skybox_builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
        self.skybox_rgb1    = params["skybox_settings"]["rgb1"]
        self.skybox_rgb2    = params["skybox_settings"]["rgb2"]
        self.skybox_width   = params["skybox_settings"]["width"]
        self.skybox_height  = params["skybox_settings"]["height"]

        # target settings:
        self.target_radius = params["target_settings"]["radius"]
        self.target_height = params["target_settings"]["height"]
        
        # visual settings:
        self.znear       = params["visual_settings"]["znear"]
        self.zfar        = params["visual_settings"]["zfar"]
        self.shadowsize  = params["visual_settings"]["shadowsize"]
        self.framelength = params["visual_settings"]["framelength"]
        self.framewidth  = params["visual_settings"]["framewidth"]
        self.jointlength = params["visual_settings"]["framelength"]
        self.jointwidth  = params["visual_settings"]["jointwidth"]

    # function for initiliazing the MjSpec:
    def make_spec(self):
        """
        this function initializes the ``MjSpec`` and applies the passed basic settings/requirements for the 
        environment (plane, skybox, light, camera, walls, etc.)
        """

        # initialize spec:
        self.spec = mj.MjSpec()
        self.spec.modelfiledir = self.mesh_dir

        # set the compiler settings:
        self.spec.compiler.degree = self.compiler_angle # 1 -> degress, 0 -> radians

        # set the option settings:
        self.spec.option.timestep   = self.timestep
        self.spec.option.integrator = self.integrator
        self.spec.option.gravity    = self.gravity

        # set the visualization settings:
        self.spec.visual.quality.shadowsize = self.shadowsize
        self.spec.visual.map.znear          = self.znear
        self.spec.visual.map.zfar           = self.zfar
        self.spec.visual.scale.framelength  = self.framelength
        self.spec.visual.scale.framewidth   = self.framewidth

        # default settings:
        self.spec.default.joint.damping = np.array([self.joint_damping, 0.0, 0.0])
        self.spec.default.joint.armature = self.joint_armature

        # add the skybox:
        self.spec.add_texture(name = self.skybox_name,
                              type = self.skybox_type,
                              builtin = self.skybox_builtin,
                              width = self.skybox_width,
                              height = self.skybox_height,
                              rgb1 = self.skybox_rgb1,
                              rgb2 = self.skybox_rgb2)
        
        # add the light:
        self.spec.worldbody.add_light(name = self.light_name,
                                      pos = self.light_pos,
                                      diffuse = self.light_diffuse,
                                      specular = self.light_specular,
                                      ambient = self.light_ambient,
                                      castshadow = self.light_shadows)
        
        # add camera:
        self.spec.worldbody.add_camera(name = self.camera_name,
                                       pos = self.camera_pos)
        
        # add ground plane:
        self.spec.worldbody.add_geom(name = self.ground_name,
                                     type = self.ground_type,
                                     contype = self.ground_contype,
                                     conaffinity = self.ground_conaffinity,
                                     pos = self.ground_pos,
                                     size = self.ground_size,
                                     rgba = self.ground_rgba)
        

    def add_robot(self, robot_pos: list):
        """
        Attach the robot to the world at ``robot_pos`` and build the kinematic chain
        described by ``self._joint_data`` (one nested body per revolute joint),
        then weld ``self._fixed_links`` onto whichever body matches their ``parent`` name.

        :param robot_pos: ``[x, y, z] position of the robot base in the world frame.
        :type robot_pos: list
        """

        mesh_names = [self.robot_base_mesh] + [jd["mesh"] for jd in self._joint_data] \
                    + [fl["mesh"] for fl in self._fixed_links]
        self._mesh_name_lookup = {}

        for mesh_file in mesh_names:
            mesh_name = os.path.splitext(mesh_file)[0]
            if mesh_name in self._mesh_name_lookup:
                continue
            self.spec.add_mesh(name=mesh_name, file=mesh_file)
            self._mesh_name_lookup[mesh_file] = mesh_name

        # base body, fixed to the world at robot_pos
        self.robot = self.spec.worldbody.add_body(name=self.env_name + "_base", pos=robot_pos)
        self.robot.add_geom(name="base_visual",
                            type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=self._mesh_name_lookup[self.robot_base_mesh],
                            contype=0,
                            conaffinity=0,
                            rgba=self.robot_base_rgba)
        
        # visual-only footprint disc marking the robot' reach/no-go zone on the ground:
        self.robot.add_geom(name=self.robot_footprint_name,
                            type = mj.mjtGeom.mjGEOM_CYLINDER,
                            size=[self.robot_footprint_radius, self.robot_footprint_height,0.0],
                            contype=self.robot_footprint_contype,
                            conaffinity=self.robot_footprint_conaffinity,
                            rgba=self.robot_footprint_rgba)
        
        # walk the joint chain, nesting each new body inside the previous one:
        parent_body = self.robot
        self._body_lookup = {}
        for jd in self._joint_data:
            body = parent_body.add_body(name=jd["link_name"],
                                        pos=jd["origin_xyz"],
                                        euler=jd["origin_rpy"])
            
            body.add_joint(name=jd["joint_name"],
                           type=mj.mjtJoint.mjJNT_HINGE,
                           axis=jd["axis"],
                           range=[jd["lower"], jd["upper"]],
                           damping=self.joint_damping,
                           armature=self.joint_armature)
            
            # visual geom: the actual mesh, no collision:
            body.add_geom(name=f"{jd['link_name']}_visual",
                          type=mj.mjtGeom.mjGEOM_MESH,
                          meshname=self._mesh_name_lookup[jd["mesh"]],
                          contype=0,
                          conaffinity=0,
                          rgba=self.link_rgba)
            
            col = jd["collision"]
            body.add_geom(name=f"{jd["link_name"]}_collision",
                          type=_COLLISION_GEOM_TYPE[col["type"]],
                          size=_collision_geom_size(col),
                          pos=col["pos"],
                          euler=col["rpy"],
                          contype=1,
                          conaffinity=1,
                          rgba=[1,0,0,0.3])
            
            self._body_lookup[jd["link_name"]] = body
            parent_body = body

        for fl in self._fixed_links:
            parent_body = self._body_lookup[fl["parent"]]
            body = parent_body.add_body(name=fl["link_name"],
                                        pos=fl["origin_xyz"],
                                        euler=fl["origin_rpy"])
            
            # visual geom: the actual mesh, no collision:
            body.add_geom(name=f"{fl['link_name']}_visual",
                        type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=self._mesh_name_lookup[fl["mesh"]],
                        contype=0,
                        conaffinity=0,
                        rgba=self.tool_rgba)
            
            col = fl["collision"]
            body.add_geom(name=f"{fl["link_name"]}_collision",
                        type=_COLLISION_GEOM_TYPE[col["type"]],
                        size=_collision_geom_size(col),
                        pos=col["pos"],
                        euler=col["rpy"],
                        contype=1,
                        conaffinity=1,
                        rgba=[1,0,0,0.3])
            
            self._body_lookup[fl["link_name"]] = body

        self.end_effector_body = self._body_lookup[self._fixed_links[-1]["link_name"]]

    def add_actuators(self): 
        """
        This function add the actuators at each joint
        """
        for name in self.joint_names:
            act = self.spec.add_actuator()
            act.name = f'act_{name}'
            act.trntype = mj.mjtTrn.mjTRN_JOINT
            act.target = name
            act.gaintype = mj.mjtGain.mjGAIN_FIXED
            act.gainprm = [10.0, 0.0, 0.0] + [0.0]*7
            act.biastype = mj.mjtBias.mjBIAS_AFFINE
            act.biasprm = [0.0, 0.0, -10.0] + [0.0]*7
            act.ctrlrange = [-self.qdot_limit, self.qdot_limit]
            act.ctrllimited = True
            
        

    def add_sensors(self):
        """
        Add joint-state and end-effector sensors.

        After compile(), readings are available via:
        data.sensordata - flat array, ordered by sensor definition

        Sensor layout (18 values total):

            [0:6] jointpos - q for joints 1-6 (rads)
            [6:12] jointvel - qdot for joints 1-6 (rads/s)
            [12:15] framepos - EE Cartesian position (m)
            [15:18] framequat - EE orientation quarternion (w,x,y,z)
        """
        # joint position sensors (one per revolute joint):
        for jd in self._joint_data:
            s = self.spec.add_sensor()
            s.name = f"pos_{jd['joint_name']}"
            s.type = mj.mjtSensor.mjSENS_JOINTPOS
            s.objtype = mj.mjtObj.mjOBJ_JOINT
            s.objname = jd['joint_name']

        # joint velocity sensors (one per revolute joint):
        for jd in self._joint_data:
            s = self.spec.add_sensor()
            s.name = f"vel_{jd['joint_name']}"
            s.type = mj.mjtSensor.mjSENS_JOINTVEL
            s.objtype = mj.mjtObj.mjOBJ_JOINT
            s.objname = jd['joint_name']

        # end-effector Cartesian Position
        s = self.spec.add_sensor()
        s.name = "ee_pos"
        s.type = mj.mjtSensor.mjSENS_FRAMEPOS
        s.objtype = mj.mjtObj.mjOBJ_BODY
        s.objname = self._joint_data[-1]['link_name']

        # end-effector orientation (quarternion)
        s = self.spec.add_sensor()
        s.name = "ee_quat"
        s.type = mj.mjtSensor.mjSENS_FRAMEQUAT
        s.objtype = mj.mjtObj.mjOBJ_BODY
        s.objname = self._joint_data[-1]['link_name']  

    def add_obstacle(self, obs_pos:list):
        """
        This function generates a simple primitive obstacle.

        :param obs_pos: a list containing the ``[X,Y,Z]`` position of the obstacle.
        :type obs_pos: list
        """

        self.obstacle_counter += 1

        # add the obstacle to the worldbody:
        self.obstacle = self.spec.worldbody.add_body(name=f"obstacle_{self.obstacle_counter}",
                                                     mocap = True,
                                                     pos = obs_pos
                                                     )
        
        # randomize the size of the obstacle between low and high
        obstacle_size = np.array([np.random.uniform(low= self.obstacle_size_low, high = self.obstacle_size_high), self.obstacle_height, 0.0])
        footprint_size = obstacle_size.copy()
        footprint_size[0] += self.allowance
        footprint_size[1] = self.robot_footprint_height


        self.obstacle.add_geom(name=f"obstacle_{self.obstacle_counter}_geom",
                               type = mj.mjtGeom.mjGEOM_SPHERE,
                               size = obstacle_size,
                               contype = 1,
                               conaffinity = 1,
                               rgba = [0,0,1,1]
                               )

    def add_target(self, target_pos:list):
        """
        Add a visual target marker that the arm should try to reach. 
        Implemented as a mocap body so it has no physics/joints of its own but can still be repositioned per-episode
        via ``data.mocap_pos`` without recompiling.

        :param target_pos: ``[x,y,z]`` position of the target in the world frame.
        :type target_pos: list
        """

        self.target = self.spec.worldbody.add_body(name="target",
                                                   mocap=True,
                                                   pos= target_pos)
        
        self.target.add_geom(name="target_geom",
                             type=mj.mjtGeom.mjGEOM_SPHERE,
                             size= self.target_radius,
                             contype=0,
                             conaffinity=0,
                             rgba=[0.0,1.0,0.0,0.6]
                             )

    def compile(self):
        """
        This function compiles the model using the builtin method for ``mj.MjSpec()``, ``.compile()``, the spec must be compiled such 
        that it can be used in a broader MuJoCo simulation context.
        
        """
        self.model = self.spec.compile()

    def make_env(self,
                 robot_pos: list,
                 target_pos:  list,
                 obs_pos:   list | None = None):
        """
        Function uses methods defined above and basically just chain them together to make and compile the environment.
        Responsible -> making the ``spec`` and applying the defauly settings (options, visual, lightining, camera, skybox, plane, walls)
        adding in the manipulator (robot), actuators, sensors, targets and obstacles and then compiling the ``spec`` into a usable ``model``
        """
        # initialize the spec:
        self.make_spec()

        # add the robot:
        self.add_robot(robot_pos = [robot_pos[0], robot_pos[1], self.robot_footprint_height])

        # add actuators:
        self.add_actuators()

        # add sensors:
        self.add_sensors()

        # add target:
        self.add_target(target_pos = [target_pos[0], target_pos[1], self.target_height])

        # add obstalces:
        n_obstacles = len(obs_pos)

        # for every obstacle
        for i in range (n_obstacles):
            # add a primitive obstacle:
            self.add_obstacle(obs_pos = [obs_pos[i][0], obs_pos[i][1], self.obstacle_height])
        
        # compile into model:
        self.compile()

    def render(self):
        """
        This function renders and steps through the environment every timestep. Takes the compiled model and extracts 
        the data struct, which containts the simulation states. It then launces a viewer using the model and the data.

        The settings that are altered are:
            viewer.cam.type:              this specifies the type of camera that is used
            viewer.cam.ifxedcamid:        this specifies the ID of the user-defined camera
            viewer.opt.frame:             this specifies which frame(s) to have active
            viewer.opt.flags:             this specifies the flag(s) to enable

        """
        self.data = mj.MjData(self.model)
        self.data.ctrl[:] = 0.0


        # launch a passive window using the model and the data contained within:
        with mujoco.viewer.launch_passive(self.model, self.data) as self.viewer:
            # switch the camera
            self.viewer.camtype = mj.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.model.camera(self.camera_name).id

            # enable viewer options:
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

            # while viewer is active, step the model every timestep:
            while self.viewer.is_running():
                mj.mj_step(self.model, self.data)
                self.viewer.sync()


        