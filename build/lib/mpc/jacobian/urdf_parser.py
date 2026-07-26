"""
Walks a URDF's kinematic chain from base link to ee link and produces a
RobotDescription 

"""
from typing import Optional, Tuple

import numpy as np
from urdf_parser_py.urdf import URDF, Joint

DEFAULT_JOINT_LIMIT = np.pi
DEFAULT_VEL_LIMIT   = 4.0

def _rpy_to_matrix(rpy) -> np.ndarray:
    if rpy is None:
        rpy = (0.0, 0.0, 0.0)
    r, p, y = rpy

    Rx = np.array([[1.0, 0.0, 0.0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r),  np.cos(r)]])

    Ry = np.array([[np.cos(p), 0, np.sin(p)],[ 0.0,1.0,0.0],[-np.sin(p), 0, np.cos(p)]])

     return np.vertnpt(
            np.horznpt( np.cos(p), 0, np.sin(p) ),
            np.horznpt( 0,         1,      0    ),
            np.horznpt(-np.sin(p), 0, np.cos(p) )
        )


