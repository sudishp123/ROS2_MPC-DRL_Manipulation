import numpy as np
import casadi as ca
from hilo_mpc import NMPC, SimpleControlLoop
from mpc.jacobian.generic_jacobian_casadi import build_kinematics

class NMPCController:
    """NMPC Based on Baselizadeh et al. (2024)
    N: prediction horizon of 5
    dt: timestep
    ds: safety distance threshold (m)"""
    def __init__(self, N = 5, dt = 0.01, ds = 0.2):
        self.N =  N
        self.dt = dt
        self.ds = ds
        self.nq = 6 # number of joints
        self.np = 3 # Cartersian Position (x,y,z)

    #----------------Load Kinematics----------------
        self.J_fn, self.fk_fn, self.J_sym, self.p_e_sym, self.q_kin = build_kinematics()

    #TODO - Generalize the joint limits instead of hardcoding 
        self.q_min = np.array([-3.05, -1.57, -1.57, -1.57, -3.05, -1.57])
        self.q_max = np.array([3.05, 1.57, 1.57, 1.57, 3.05, 1.57])

        self.qdot_min = np.full(6, -4.0) #Locked Motor speed is 52 RPM = 5.445 rads/s so haing a 75% operating limit
        self.qdot_max = np.full(6, 4.0)

        #Terminal Cost Weight S_N - fixed, based on paper
        self.S_N = np.diag([10.0, 10.0, 10.0])

        #Defauly NMPC parameter (DRL will takeover during runtime)
        #theta_S: diagonal weights on Cartesian error (3,) - paper uses 6 for pose error
        #theta_R: diagonal weights on joint velocities (6,)
        #theta_g: collision avoidance margin per link (N_links)

        self.theta_s = np.ones(self.np) * 10.0
        self.theta_r = np.ones(self.nq) * 1.0
        self.theta_g = np.ones(1) * 0.0

        # Large penalty weights on slack variables (eta in paper)
        # eta and eta_f - paper uses [5,5,5,3,3,3]
        self.eta = np.array([5.0])
        self.eta_f = np.array([5.0])
        
        self._build_ocp()

    #---------------- Collision Distance Calculator Helper Function (NEEDS TO BE CHANGED)----------------
    def _collision_distance(self, q_k, T_obs):
        """
        Minimum distance between end-effector (approximated as a point) and a spherical obstacle

        Args:
            q_k    : ca.SX(6,) — joint angles at step k
            T_obs  : ca.SX(3,) — obstacle centre position
        """
        p_ee = ca.substitute(self.p_e_sym, self.q_kin, q_k)
        dist = ca.norm_2(p_ee - T_obs)
        r_obs = 0.15   #obstacle_raidus (m)
        r_link = 0.05  #link_radius approx (m)
        g_k = dist - r_obs - r_link # > 0 means no collision
        return g_k

    #----------------Optimal Control Problem Building Helper Function----------------
    def _build_ocp(self):
        """
        Transcribes the paper's cost function into a parametric nonlinear program

        Parameters passed at solve-time
        p = [q_init(6),        — current joint angles
                 p_des(3),         — desired EE position
                 T_obs(3),         — obstacle centre
                 theta_s(3),       — DRL-tuned S^theta diagonal
                 theta_r(6),       — DRL-tuned R^theta diagonal
                 theta_g(1)]       — DRL-tuned collision margin
        """
        N   = self.N
        nq  = self.nq
        np_ = self.np

        #----------------Symbolic Parameters----------------
        # Everything the DRL or environment provides at runtime
        p_param = ca.SX.sym('p', nq + np_ + 3 + np_ + nq + 1)
        #                        q0   p_des  T_obs  θ_s   θ_r  θ_g

        q_init   = p_param[                : nq              ]
        p_des    = p_param[nq              : nq+np_          ]
        T_obs    = p_param[nq+np_          : nq+np_+3        ]
        theta_s  = p_param[nq+np_+3        : nq+np_+3+np_    ]
        theta_r  = p_param[nq+np_+3+np_    : nq+np_+3+np_+nq ]
        theta_g  = p_param[nq+np_+3+np_+nq :                 ]

        #----------------Decision Variables----------------
        # w = [q_0, qdot_0, zeta_0,  q_1, qdot_1, zeta_1, ..., q_N, zeta_N]
        #
        # q_k     : joint angles at step k        (6,)
        # qdot_k  : joint velocity control at k   (6,)  ← the control input
        # zeta_k  : slack variable at step k      (1,)  for collision constraint

        w   = []
        w0  = []
        lbw = []
        ubw = []
        g   = []
        lbg = []
        ubg = []

        #----------------Build Decision Variables lists----------------
        Q    = [] #joint angle variables at each step
        Qdot = [] #joint velocity variables at each step (control action)
        Zeta = [] #slack variables at each step

        for k in range(N+1):
            #Joint angle state q_k
            q_k = ca.SX.sym(f'q_{k}',nq)
            Q.append(q_k)
            w   += [q_k]
            lbw += list(self.q_min)
            ubw += list(self.q_max)
            w0  += [0.0] * nq

            #Slack Variable zeta_k 
            zeta_k = ca.SX.sym(f'zeta_{k}', 1)
            Zeta.append(zeta_k)
            w   += [zeta_k]
            lbw += [0.0]
            ubw += [1e6]
            w0  += [0.0]
            
            #Control qdot_k — not including terminal conditions
            if k < N:
                qdot_k = ca.SX.sym(f'qdot_{k}', nq)
                Qdot.append(qdot_k)
                w   += [qdot_k]
                lbw += list(self.qdot_min)
                ubw += list(self.qdot_max)
                w0  += [0.0] * nq
                
        #Dynamic Constraints (Eq. 8c, 8d)
        # p_{k+1} -> implict through q_{k+1} - propogate q directly
        # q_{k+1} = q_k + dt * qdot_k
        # p_{k+1} = p_k + dt*J(q_k)*qdot_k is encoded in the cost function via FK
        for k in range(N):
            q_next_pred = Q[k] + self.dt * Qdot[k]
            g += [Q[k+1] - q_next_pred]
            lbg += [0.0] * nq
            ubg += [0.0] * nq

        #----------------Collision Avoidance Constraints----------------
        # g^j_k(q_k, T_k)  + theta^j_g <= zeta^j_k
        # Rearranged: g^j_k + theta^j_g - zeta^j_k <= 0

        for k in range(N):
            g_dist = self._collision_distance(Q[k], T_obs)
            # ds - g_dist is the constraint from Equation (7) of paper g^j_k = ds-g_dist
            g_col = (self.ds - g_dist) + theta_g - Zeta[k]
            g   += [g_col]
            lbg += [-1e6]
            ubg += [0.0]

        # Terminal Collision Constraint (Eq. 8f)
        g_dist_N = self._collision_distance(Q[N], T_obs)
        g_col_N = (self.ds - g_dist_N) + theta_g - Zeta[N]
        g   += [g_col_N]
        lbg += [-1e6]
        ubg += [0.0]

        #----------------Cost Function----------------
        cost = ca.SX(0)

        # Terminal Cost: x_N^T S_N x_N + eta_f^T zeta_N
        p_e_N = ca.substitute(self.p_e_sym, self.q_kin, Q[N])
        x_N = p_des - p_e_N
        S_N = ca.DM(self.S_N)
        cost += ca.mtimes([x_N.T, S_N, x_N])
        cost += self.eta_f[0] * Zeta[N]

        #Stage Cost: sum_{k=0}^{N-1} (x_k^T S^theta_k x_k + qdot^T R^theta_k qdot^T + eta^T zeta_k)
        for k in range(N):
            # Cartesian Error x_k = p_des - p_k
            p_e_k = ca.substitute(self.p_e_sym, self.q_kin,Q[k])
            x_k = p_des - p_e_k

            #S^theta_l = diag(theta_s) - DRL tunes the diagonal
            S_k = ca.diag(theta_s)
            R_k = ca.diag(theta_r)

            cost += ca.mtimes([x_k.T,     S_k,  x_k])  #Setpoint Tracking
            cost += ca.mtimes([Qdot[k].T, R_k, Qdot[k]]) #Control Minimisation
            cost += self.eta[0] * Zeta[k]                #Slack Penalty

        
        #----------------Assembling the Non-linear Program (NLP)----------------
        nlp = {
            'x': ca.vertcat(*w),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': p_param
        }

        opts = {
            'ipopt.max_iter'     : 500,
            'ipopt.tol'          : 1e-4,
            'ipopt.print_level'  : 0,
            'print_time'         : False,
        }

        self.solver = ca.nlpsol('nmpc', 'ipopt', nlp, opts)

        #Store strucutre info needed in solve()
        self._w0  = np.array(w0,   dtype=float)
        self._lbw = np.array(lbw, dtype=float)
        self._ubw = np.array(ubw, dtype=float)
        self._lbg = np.array(lbg, dtype=float)
        self._ubg = np.array(ubg, dtype=float)

        #Stride: each step k has [q_k(6), zeta_k(1), qdot_k(6)] = 13 vars
        #except: the last step which has no qdot: [q_N(6). zeta_N(1)] = 7
        self.stride = nq + 1 + nq #13

        print(f"NMPC compiled - decision vars: {ca.vertcat(*w).shape[0]}")

    #----------------Setting DRL parameters----------------
    def set_drl_params(self, theta_s, theta_r, theta_g):
        """Called by the DRL policy to update NMPC weights"""
        self.theta_s = np.clip(theta_s, 0.01, 10000.0)
        self.theta_r = np.clip(theta_r, 0.01, 100.0)
        self.theta_g = np.clip(theta_g, 0.0, 10.0)

     #----------------NMPC solve helper function----------------
    def solve(self, q_current, p_des, T_obs):
        """Run one NMPC step."""
        #Pack parameter vector
        p_val = np.concatenate([
            q_current,
            p_des,
            T_obs,
            self.theta_s,
            self.theta_r,
            self.theta_g,
        ])

        sol = self.solver(
            x0  = self._w0,
            lbx = self._lbw,
            ubx = self._ubw,
            lbg = self._lbg,
            ubg = self._ubg,
            p   = p_val 
        )

        w_opt = np.array(sol['x']).flatten()

        #Extract qdot_0
        qdot_opt = w_opt[self.nq + 1: self.nq + 1 + self.nq]

        self._w0 = w_opt

        info = {
            'cost'   : float(sol['f']),
            'status' : self.solver.stats()['return_status'],
            'p_e'    : np.array(self.fk_fn(q_current)).flatten(),
        }

        return qdot_opt, info
