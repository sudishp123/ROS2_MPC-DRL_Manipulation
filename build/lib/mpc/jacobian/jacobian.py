import roboticstoolbox as rtb
import numpy as np

robot = rtb.ERobot.URDF("/home/sudhishp/ROS2_MPC+DRL_Manipulation/assets/jetcobot/urdf/jetcobot.urdf")
robot_6dof = robot.ets(end=robot.link_dict["6_Link"])


# Printing Robot Link and Their Characteristics
print(robot)

# Define a joint configuration (e.g., all zeros = home position)
q = np.zeros(robot.n)

# Compute the full geometric Jacobian
J = robot.jacob0(q)
print(J)

# p_e = robot.fkine(q).t
# print(p_e)
