import json, math
from pathlib import Path

from urdf_parser_py.urdf import URDF, Joint

DEFAULT_VEL_LIMIT = 4.0

COLLISION_OVERRIDES = {
    "1_Joint":{
        "type": "cylinder", "radius": 0.038, "half_length": 0.035,
        "pos": [0, 0, 0.035], "rpy": [0,0,0]
    },
    "2_Joint": {
        "type": "capsule", "radius": 0.030, "half_length": 0.055,
        "pos": [0, -0.023, 0.052], "rpy": [0, 0, 0],
    },
    "3_Joint": {
        "type": "capsule", "radius": 0.028, "half_length": 0.048,
        "pos": [0, 0.023, 0.044], "rpy": [0, 0, 0],
    },
    "4_Joint": {
        "type": "capsule", "radius": 0.025, "half_length": 0.015,
        "pos": [0, -0.03, 0.008], "rpy": [0, 0, 0],
    },
    "5_Joint": {
        "type": "capsule", "radius": 0.025, "half_length": 0.015,
        "pos": [0.001, 0, 0.04], "rpy": [0, 1.5, 0],
    },
    "6_Joint":{
            "type": "sphere","radius": 0.020,
             "pos": [0.007, 0, 0.0],"rpy": [0,0,0]
    },
    "jiazhua_Joint":{
                "type": "box", "half_extents": [0.035, 0.025, 0.020],
                 "pos": [0.0, 0, 0.0],"rpy": [0,0,0]
        },
    "camera_Joint":{
                    "type": "box", "half_extents": [0.015, 0.015, 0.015],
                     "pos": [0.0, 0, 0.0],"rpy": [0,0,0]
            },

}

def _mesh_filename(link):
    if link and link.visuals:
        geom = link.visuals[0].geometry
        if geom.__class__.__name__ == "Mesh":
            return Path(geom.filename).name
    return None

def _joints_from_base(robot, base_link):
    ordered_joint_names = []
    visited_links       = set()

    def visit(link_name):
        if link_name in visited_links:
            return 
        visited_links.add(link_name)
        for child_joint_name, child_link_name in robot.child_map.get(link_name, []):
            ordered_joint_names.append(child_joint_name)
            visit(child_link_name)

    visit(base_link)

    print(ordered_joint_names)
    return ordered_joint_names

def _intertial(link):
    if link is None or link.inertial is None:
        return None
    inert = link.inertial
    origin_xyz = list(inert.origin.xyz) if inert.origin and inert.origin.xyz else [0.0,0.0,0.0]
    origin_rpy = list(inert.origin.rpy) if inert.origin and inert.origin.rpy else [0.0,0.0,0.0]
    i = inert.inertia
    return {
        "mass": inert.mass,
        "origin_xyz": origin_xyz,
        "origin_rpy": origin_rpy,
        "ixx": i.ixx, "iyy": i.iyy, "izz": i.izz,
        "ixy": i.ixy, "ixz": i.ixz, "iyz": i.iyz,
    }


def parse_robot_settings(
        urdf_path: str,
        base_link: str,
        default_vel_limit: float = DEFAULT_VEL_LIMIT,
):
    with open(urdf_path, "rb") as f:
        xml_bytes = f.read()
    robot = URDF.from_xml_string(xml_bytes)

    chain_joint_names = _joints_from_base(robot, base_link)

    revolute_joints = []
    fixed_links = []

    for joint_name in chain_joint_names:
        joint = robot.joint_map[joint_name]
        child_link = robot.link_map.get(joint.child)

        origin_xyz  = list(joint.origin.xyz) if joint.origin and joint.origin.xyz else [0.0, 0.0, 0.0]
        origin_rpy  = list(joint.origin.rpy) if joint.origin and joint.origin.rpy else [0.0, 0.0, 0.0]
        mesh_file   = _mesh_filename(child_link)
        collision   = COLLISION_OVERRIDES.get(joint.name)

        if joint.type in ("revolute", "continuous"):
            axis = list(joint.axis) if joint.axis else [0.0, 0.0, 1.0]
            if joint.type == "revolute" and joint.limit:
                lower, upper = joint.limit.lower, joint.limit.upper
            else:
                lower, upper = -math.pi, math.pi

            revolute_joints.append({
                "joint_name"    : joint.name,
                "link_name"     : joint.child,
                "mesh"          : mesh_file,
                "origin_xyz"    : origin_xyz,
                "origin_rpy"    : origin_rpy,
                "axis"          : axis,
                "lower"         : lower,
                "upper"         : upper,
                "collision"     : collision,
                "inertial"      : _intertial(child_link)     
            })

        elif joint.type == "fixed":
            fixed_links.append({
                "link_name"     : joint.child,
                "mesh"          : mesh_file,
                "origin_xyz"    : origin_xyz,
                "origin_rpy"    : origin_rpy,
                "parent"        : joint.parent,
                "collision"     : collision,
                "inertial"      : _intertial(child_link)
            })

        else:
            raise ValueError(f"Unhandled Joint Type '{joint.type}' for joint '{joint.name}'")

    return{
        "mesh_dir": "assets/meshes",
        "base_mesh": "base_link.STL",
        "base_rgba": [1.0,1.0,1.0,1.0],
        "link_rgba": [0.8, 0.8, 0.8, 1.0],
        "tool_rgba": [0.6, 0.6, 0.6, 1.0],
        "qdot_limit": default_vel_limit,
        "revolute_joints": revolute_joints,
        "fixed_links": fixed_links
    }

def update_environment_params(env_json_path: str, robot_settings: dict):
    path = Path(env_json_path)
    with open(path, "r") as f:
        env_params = json.load(f)

    env_params["robot_settings"].update(robot_settings)

    with open(path, "w") as f:
        json.dump(env_params, f, indent=4)

if __name__=="__main__":
    urdf_path = "/home/sudhishp/ROS2_MPC+DRL_Manipulation/assets/jetcobot/urdf/jetcobot.urdf"
    json_path = "/home/sudhishp/ROS2_MPC+DRL_Manipulation/python/envs/environment_params.json"
    settings  = parse_robot_settings(urdf_path, "base_link")

    update_environment_params(json_path, settings)
