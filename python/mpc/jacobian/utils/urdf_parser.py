"""
urdf_parser.py

Walks a URDF's kinematic chain from base_link to ee_link and produces a
RobotDescription — the generic replacement for hand-transcribed ETS
tables.

Consecutive `fixed` joints are folded into the offset of the next
movable joint (matching how ETS table already bundles mounting
offsets before each active joint). Any fixed joints found *after* the
last movable joint become the end-effector / tool offset.

Requires: pip install git+https://github.com/ros/urdf_parser_py.git@ros2
"""

from typing import Optional, Tuple

import numpy as np
from urdf_parser_py.urdf import URDF, Joint

from mpc.jacobian.utils.robot_description import RobotDescription

DEFAULT_JOINT_LIMIT = np.pi  # fallback for continuous joints (no URDF limit tag)
DEFAULT_VEL_LIMIT = 4.0      # rad/s or m/s fallback if URDF has no <limit velocity=".."/>


def _rpy_to_matrix(rpy) -> np.ndarray:
    """Extrinsic R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching URDF's <origin rpy=".."> convention."""
    if rpy is None:
        rpy = (0.0, 0.0, 0.0)
    r, p, y = rpy

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_to_Rp(origin) -> Tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.eye(3), np.zeros(3)
    xyz = np.array(origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0], dtype=float)
    R = _rpy_to_matrix(origin.rpy)
    return R, xyz


def _compose(R1, p1, R2, p2) -> Tuple[np.ndarray, np.ndarray]:
    """Compose T1 @ T2 in (R, p) form."""
    return R1 @ R2, R1 @ p2 + p1


def parse_robot_description(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    default_vel_limit: float = DEFAULT_VEL_LIMIT,
) -> RobotDescription:
    # Read as raw bytes and use from_xml_string rather than from_xml_file:
    with open(urdf_path, "rb") as f:
        xml_bytes = f.read()
    robot = URDF.from_xml_string(xml_bytes)

    # get_chain returns an alternating [joint, link, joint, link, ...] list
    # from base_link to ee_link, in order — exactly the traversal we need.
    chain = robot.get_chain(base_link, ee_link, joints=True, links=False, fixed=True)

    joint_types, joint_axes = [], []
    R_fixed_list, p_fixed_list = [], []
    q_min, q_max, qdot_min, qdot_max = [], [], [], []
    joint_names = []

    # pending = accumulated transform since the last movable joint (or base_link)
    R_pending, p_pending = np.eye(3), np.zeros(3)

    for jname in chain:
        joint: Joint = robot.joint_map[jname]
        R_o, p_o = _origin_to_Rp(joint.origin)
        R_pending, p_pending = _compose(R_pending, p_pending, R_o, p_o)

        if joint.type == "fixed":
            continue  # keep accumulating, don't reset

        if joint.type not in ("revolute", "continuous", "prismatic"):
            raise ValueError(
                f"Joint '{jname}' has unsupported type '{joint.type}'. "
                "Only revolute, continuous, prismatic, and fixed are handled."
            )

        axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0], dtype=float)
        axis = axis / np.linalg.norm(axis)

        joint_types.append(joint.type)
        joint_axes.append(axis)
        R_fixed_list.append(R_pending)
        p_fixed_list.append(p_pending)
        joint_names.append(jname)

        if joint.type == "continuous" or joint.limit is None:
            lo, hi = -DEFAULT_JOINT_LIMIT, DEFAULT_JOINT_LIMIT
        else:
            lo = joint.limit.lower if joint.limit.lower is not None else -DEFAULT_JOINT_LIMIT
            hi = joint.limit.upper if joint.limit.upper is not None else DEFAULT_JOINT_LIMIT
        q_min.append(lo)
        q_max.append(hi)

        vel = default_vel_limit
        if joint.limit is not None and joint.limit.velocity is not None:
            vel = joint.limit.velocity
        qdot_min.append(-vel)
        qdot_max.append(vel)

        # this movable joint's own motion resets what's "pending" for the next one
        R_pending, p_pending = np.eye(3), np.zeros(3)
        
    # anything left over after the last movable joint is the end-effector offset
    R_ee, p_ee = R_pending, p_pending

    return RobotDescription(
        name=robot.name,
        n_joints=len(joint_types),
        joint_types=joint_types,
        joint_axes=joint_axes,
        R_fixed=R_fixed_list,
        p_fixed=p_fixed_list,
        R_ee=R_ee,
        p_ee=p_ee,
        q_min=np.array(q_min),
        q_max=np.array(q_max),
        qdot_min=np.array(qdot_min),
        qdot_max=np.array(qdot_max),
        joint_names=joint_names,
        base_link=base_link,
        ee_link=ee_link,
    )

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python urdf_parser.py <robot.urdf> <base_link> <ee_link>")
        sys.exit(1)

    desc = parse_robot_description(sys.argv[1], sys.argv[2], sys.argv[3])
    print(desc)
    for i, jn in enumerate(desc.joint_names):
        print(
            f"  [{i}] {jn:20s} type={desc.joint_types[i]:10s} "
            f"axis={desc.joint_axes[i]} q=[{desc.q_min[i]:.3f}, {desc.q_max[i]:.3f}] "
            f"qdot=±{desc.qdot_max[i]:.2f}"
        )
    print(f"  EE offset: p={desc.p_ee}")