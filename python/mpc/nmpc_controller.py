import numpy as np
import casadi as ca
from hilo_mpc import NMPC, SimpleControlLoop

class NMPCController:
    def __init__(self, model, s_diag, r_diag, horizon = 10):
        self.nq = model.nq #Defines the mrobot's joint angles
        self.nv = model.nv
        self.nu = model.nu
        self.horizon = horizon

        self.s_diag_base = np.array(s_diag, dtype = float)
        self.r_diag_base = np.array(r_diag, dtype = float)
        self.s_diag = self.s_diag_base.copy()
        self.r_diag = self.r_diag_base.copy()

        self._build_mpc(model)

    def _build_mpc(self, model):
        """Construct HILO-MPC Optimal Control Problem with symbolic robot dynamics"""
        n_states = self.nq + self.nv # Number of states
        n_inputs = self.nu # Number of inputs

        # Symbolic state and input
        x = ca.SX.sym("x", n_states)
        u = ca.SX.sym("u", n_inputs)

    def update_cost_matrices(self, delta_s: np.ndarray, delta_r: np.ndarray, scale:float=0.1):
        """DRL agent perturbs Q and R diagonals."""
        self.s_diag = np.clip(
            self.s_diag_base * (1.0 + scale * delta_s), 1e-4, 1e4
        )
        self.r_diag = np.clip(
            self.r_diag_base * (1.0 + scale * delta_r), 1e-4, 1e4
        )

        #Push updated weights to HILO-MP
        self.nmpc.quad_stage_cost.update_weights(
            states=np.concatenate([self.s_diag, np.zeros(self.nv)]),
            inputs = self.r_diag
        )




