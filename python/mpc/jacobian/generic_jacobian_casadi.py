import casadi as ca

from mpc.jacobian.utils.robot_description import RobotDescription

def skew(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

def axis_angle_rot(axis, angle):
    K = skew(axis)                  #v_rot = v + sin(angle)k x v + (1- cos(angle))k x (k x v); K = k x v
    return ca.DM.eye(3) + ca.sin(angle) * K + (1-ca.cos(angle)) * (K@K) #v_rot = Rv ===> R = I + sin(angle)*K + (1-cos(angle))*(K^2)

def make_T(R, p):
    """4x4 homogeneous transform from 3x3 R and 3x1 p."""
    p = ca.reshape(p, 3, 1)
    return ca.vertcat(
        ca.horzcat(R, p),
        ca.horzcat(ca.DM.zeros(1, 3), ca.DM([[1]]))
    )

def joint_transform(jtype: str, axis, q_i):
    if jtype in ("revolute", "continuous"):
        R = axis_angle_rot(axis, q_i)
        return make_T(R, ca.DM([0, 0, 0]))
    elif jtype == "prismatic":
        p = axis * q_i
        return make_T(ca.DM.eye(3), p)
    else:
        raise ValueError(f"Unsupported joint type: {jtype}")

def build_kinematics(desc: RobotDescription):
    n = desc.n_joints
    q = ca.SX.sym("q", n)

    T_links = []
    for i in range(n):
        R_f     = ca.DM(desc.R_fixed[i])
        p_f     = ca.DM(desc.p_fixed[i])
        axis_i  = ca.DM(desc.joint_axes[i])
        T_i     = make_T(R_f, p_f) @ joint_transform(desc.joint_types[i], axis_i, q[i])
        T_links.append(T_i)

    T_cum = [ca.DM.eye(4)]
    for i in range(n):
        T_cum.append(T_cum[-1] @ T_links[i])

    T_ee_fixed  = make_T(ca.DM(desc.R_ee), ca.DM(desc.p_ee))
    T_0e        = T_cum[n] @ T_ee_fixed
    p_e         = T_0e[:3,3]
    R_e         = T_0e[:3, :3] 

    cols = []
    for i in range(n):
        T_at_joint  = T_cum[i] @ make_T(ca.DM(desc.R_fixed[i]), ca.DM(desc.p_fixed[i]))
        axis_world  = T_at_joint[:3, :3] @ ca.DM(desc.joint_axes[i])   # joint axis direction
        p_i         = T_at_joint[:3, 3]   # joint origin position

        if desc.joint_types[i] in ("revolute", "continuous"):
            Jv = ca.cross(axis_world, p_e-p_i)
            Jw = axis_world
        else:
            Jv = axis_world
            Jw = ca.DM.zeros(3,1)

        cols.append(ca.vertcat(Jv, Jw))

    J_sym   = ca.horzcat(*cols)
    J_fn    = ca.Function('J', [q],[J_sym])
    fk_fn   = ca.Function('fk',[q],[p_e])
    R_fn    = ca.Function('R', [q],[R_e])

    return J_fn, fk_fn, R_fn, J_sym, p_e, R_e, q

if __name__ == "__main__":
    import numpy as np
    import sys

    if len(sys.argv) != 4:
        print("Usage: python generic_jacobian_casadi.py <robot.urdf> <base_link> <ee_link>")
        sys.exit(1)

    from mpc.jacobian.utils.urdf_parser import parse_robot_description

    desc = parse_robot_description(sys.argv[1], sys.argv[2], sys.argv[3])
    J_fn, fk_fn, R_fn, J_sym, p_e, R_e, q = build_kinematics(desc)

    rng = np.random.default_rng(0)
    q_test = rng.uniform(desc.q_min, desc.q_max)

    J_analytic = np.array(J_fn(q_test))[:3, :]

    eps = 1e-6
    J_numeric = np.zeros((3, desc.n_joints))
    for i in range(desc.n_joints):
        dq = np.zeros(desc.n_joints)
        dq[i] = eps
        p_plus  = np.array(fk_fn(q_test+dq)).flatten()
        p_minus = np.array(fk_fn(q_test-dq)).flatten()
        J_numeric[:, i] = (p_plus - p_minus)/(2*eps)

    err = np.max(np.abs(J_analytic - J_numeric))
    print(f"Robot: {desc.name}, n_joints={desc.n_joints}")
    print(f"Max |analytic - finit-difference| Jacobian error: {err:.3e}")