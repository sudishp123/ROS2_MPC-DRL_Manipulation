from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

JointType = Literal["revolute", "continuous", "prismatic"]

@dataclass
class RobotDescription:
    n_joints: int
    joint_types: List[JointType]
    joint_axes: List[np.ndarray]
    R_fixed: List[np.ndarray]
    p_fixed: List[np.ndarray]
    R_ee: np.ndarray
    p_ee: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    qdot_min: np.ndarray
    qdot_max: np.ndarray
    joint_names: List[str] = field(default_factory=list)
    base_link: str = ""
    ee_link: str= ""

    def __post_init__(self):
        n = self.n_joints
        assert len(self.joint_types) == n
        assert len(self.joint_axes) == n
        assert len(self.R_fixed) == n
        assert len(self.p_fixed) == n
        assert self.q_min.shape == (n,)
        assert self.q_max.shape == (n,)
        assert self.qdot_min.shape == (n,)
        assert self.qdot_max.shape == (n,)


    def __repr__(self):
        return (
            f"RobtoDescription(name={self.name!r}, n_joints={self.n_joints})"
            f"types={self.joint_types}, base={self.base_link!r}, ee={self.ee_link!r}"
        )
        

