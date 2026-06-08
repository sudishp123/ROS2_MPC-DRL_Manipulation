import casadi as ca
import numpy as np

#Building Rotation Matrices symbolically
def Rx(a):
    """Rotation about X axis"""
    return ca.vertcat(
        ca.horzcat(1,     0,          0    ),
        ca.horzcat(0, ca.cos(a), -ca.sin(a)),
        ca.horzcat(0, ca.sin(a),  ca.cos(a))
    )

def Ry(a):
    """Rotation about Y axis"""
    return ca.vertcat(
        ca.horzcat( ca.cos(a), 0, ca.sin(a) ),
        ca.horzcat( 0,         1,      0    ),
        ca.horzcat(-ca.sin(a), 0, ca.cos(a) )
    )

def Rz(a):
    """Rotation about Z axis."""
    return ca.vertcat(
        ca.horzcat( ca.cos(a), -ca.sin(a), 0 ),
        ca.horzcat( ca.sin(a),  ca.cos(a), 0 ),
        ca.horzcat( 0,          0,         1 )
    )

def rpy_to_rot(roll, pitch, yaw):
    """RPY convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

def make_T(R, p):
    """4x4 homogeneous transform from 3x3 R and 3x1 p."""
    p = ca.reshape(p, 3, 1)
    return ca.vertcat(
        ca.horzcat(R, p),
        ca.horzcat(ca.DM.zeros(1, 3), ca.DM([[1]]))
    )

deg = np.pi/180.0

q = ca.SX.sym('q', 6)

#Fixed offsets from ETS table
R_fixed = [
    ca.DM.eye(3),
    rpy_to_rot(-0.06536*deg, -57.3*deg, 0.0357*deg),
    rpy_to_rot(0.06536*deg, -57.3*deg, -0.0357*deg),
    rpy_to_rot(0.06536*deg, -57.3*deg, -0.0357*deg),
    rpy_to_rot(-0.0007867*deg, 0.0004298*deg, 57.3*deg),
    rpy_to_rot(57.3*deg, 0.04823*deg, 0.02533*deg)    
]

p_fixed=[
    ca.DM([0, 0, 0.07452]),
    ca.DM([-2.563e-05, -0.035, 0.06]),
    ca.DM([0, 0.00679, 0.11]),
    ca.DM([0, -0.001345, 0.096]),
    ca.DM([-2.513e-05, -0.03354, 0.03483]),
    ca.DM([0.03197, -2.831e-05, 0.03992])
]

# Full link transforms: fixed_offset @ Rz(qi)
T_links = [
    make_T(R_fixed[i], p_fixed[i]) @ make_T(Rz(q[i]), ca.DM([0,0,0]))
    for i in range (6)
]

# Cumulative Transforms : T_cum[i] = base -> after joint 1
T_cum = [ca.DM.eye(4)]
for i in range(6):
    T_cum.append(T_cum[-1] @ T_links[i])

T_0e = T_cum[6]   # end-effector
p_e  = T_0e[:3, 3]

cols = []
for i in range(6):
    T_at_joint = T_cum[i] @ make_T(R_fixed[i], p_fixed[i])
    z_i = T_at_joint[:3, 2]   # joint axis direction
    p_i = T_at_joint[:3, 3]   # joint origin position

    Jv = ca.cross(z_i, p_e - p_i)
    Jw = z_i
    cols.append(ca.vertcat(Jv, Jw))

J_sym = ca.horzcat(*cols)
J_fn  = ca.Function('J', [q], [J_sym])

print("J shape:", J_sym.shape)  # (6, 6)

# Test at home position
q_test = np.zeros(6)
J_num  = J_fn(q_test)
# print(np.array(J_num))