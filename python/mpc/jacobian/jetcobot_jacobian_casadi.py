#jetcobot_jacobian-casadi.py
import casadi as ca
import numpy as np

deg = np.pi/180.0 #Need to convert the ETS table values to radians

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

def build_kinematics():
    """
    Returns:
        J_fn  : ca.Function [q(6,)] -> [J (6,6)]
        fk_fn : ca.Function [q(6,)] -> [p_e(3,)]
        J_sym : CasADi SX expression
        p_e   : CasADi SX expression for end-effector position
        q     : CasADi SX symbolic variable (6,)
    """
    q = ca.SX.sym('q', 6) #Creates a vector of 6 symbolic variables — one per joint angle

    #Fixed offsets between consecutive joints from ETS (Elementary Transform Structure) table
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


    # Full link transforms: Constant geometry @ Joint Rotation around the local Z axis
    T_links = [
        make_T(R_fixed[i], p_fixed[i]) @ make_T(Rz(q[i]), ca.DM([0,0,0]))
        for i in range (6)
    ]

    # Cumulative Transforms : T_cum buils the cumulative transforms by multiplying left-to-right
    T_cum = [ca.DM.eye(4)]
    for i in range(6):
        T_cum.append(T_cum[-1] @ T_links[i])

    T_0e = T_cum[6]  # end-effector
    p_e  = T_0e[:3, 3] #Rows 0-3 for Column #3 (or the first three rows of final column)
    
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
    fk_fn = ca.Function('fk',[q],[p_e])

    return J_fn, fk_fn, J_sym, p_e, q

# Evaluation of build_kinematics to test whether Jacobian is correct

# print("J shape:", build_kinematics()[2].shape)  # (6, 6)

# # Test at home position
# q_test = np.zeros(6)
# J_num  = build_kinematics()[0](q_test)
# p_enum = build_kinematics()[1](q_test)
# print(np.array(p_enum))