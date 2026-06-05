from dataclasses import dataclass
import numpy as np
import mujoco as MJ
import mujoco_warp as MW
import warp as wp
from scipy.spatial.transform import Rotation as Rot

NWORLD = 4

@dataclass
class ArmSpec:
    name: str
    body_id: int
    dof_ids: np.ndarray
    act_ids: np.ndarray
    mocap_id: int
    offset: float
    K: np.ndarray      # shape (6,) or (6,6) diagonal
    KI: np.ndarray     # shape (6,) or (6,6) diagonal
    K0: np.ndarray     # optional nullspace gains, shape (ndof,)


class ParallelOSC:
    def __init__(self, model, data, specs, nworld=NWORLD):
        self.model = model
        self.data = data
        self.nw = nworld
        self.nv = model.nv
        self.dt = float(model.opt.timestep.numpy().mean())
        self.specs = specs

        self.twist_int = {
            s.name: np.zeros((self.nw, 6), dtype=np.float32) for s in specs
        }
        self.prev_J = {s.name: None for s in specs}

    def _batched_kinematics(self):
        MW.kinematics(self.model, self.data)
        MW.com_vel(self.model, self.data)
        MW.crb(self.model, self.data)
        MW.factor_m(self.model, self.data)
        MW.rne(self.model, self.data, flg_acc=False)

    def _jacobian(self, spec: ArmSpec):
        jacp = wp.zeros((self.nw, 3, self.nv), dtype=wp.float32)
        jacr = wp.zeros((self.nw, 3, self.nv), dtype=wp.float32)

        points = wp.array(
            np.tile(np.array([0.0, 0.0, spec.offset], dtype=np.float32),
                    (self.nw, 1)),
            dtype=wp.vec3f,
        )
        bodies = wp.array(
            np.full(self.nw, spec.body_id, dtype=np.int32),
            dtype=wp.int32,
        )

        MW.jac(self.model, self.data, jacp, jacr, points, bodies)

        jacp = jacp.numpy()  # (nworld, 3, nv)      
        jacr = jacr.numpy()  # (nworld, 3, nv)
        J = np.concatenate(
            [jacp[:, :, spec.dof_ids], jacr[:, :, spec.dof_ids]],
            axis=1,
        )  # (nworld, 6, ndof)
        return J

    def _solve_m(self, rhs_full):
        # rhs_full: (nworld, nv)
        rhs_w = wp.array(rhs_full.astype(np.float32), dtype=wp.float32)
        sol_w = wp.empty_like(rhs_w)
        MW.solve_m(self.model, self.data, sol_w, rhs_w)
        return sol_w.numpy()  # (nworld, nv)

    def _mj_warp_state(self, spec: ArmSpec):
        qpos = self.data.qpos.numpy()
        qvel = self.data.qvel.numpy()
        xpos = self.data.xpos.numpy()
        xquat = self.data.xquat.numpy()
        xmat = self.data.xmat.numpy()
        mocap_pos = self.data.mocap_pos.numpy()
        mocap_quat = self.data.mocap_quat.numpy()
        qfrc_bias = self.data.qfrc_bias.numpy()

        ee_pos = xpos[:, spec.body_id, :]
        ee_quat = xquat[:, spec.body_id, :]
        ee_xmat = xmat[:, spec.body_id, :, :]
        ee_pos = ee_pos + spec.offset * ee_xmat[:, :, 2]

        target_pos = mocap_pos[:, spec.mocap_id, :]
        target_quat = mocap_quat[:, spec.mocap_id, :]

        # MuJoCo quats are w,x,y,z; SciPy wants x,y,z,w
        R_t = Rot.from_quat(np.column_stack([
            target_quat[:, 1], target_quat[:, 2], target_quat[:, 3], target_quat[:, 0]
        ]))
        R_e = Rot.from_quat(np.column_stack([
            ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3], ee_quat[:, 0]
        ]))
        rotvec = (R_t * R_e.inv()).as_rotvec().astype(np.float32)

        twist = np.concatenate([target_pos - ee_pos, rotvec], axis=1).astype(np.float32)

        q_arm = qpos[:, spec.dof_ids]
        qd_arm = qvel[:, spec.dof_ids]
        h_arm = qfrc_bias[:, spec.dof_ids]

        return twist, q_arm, qd_arm, h_arm

    def step_arm(self, spec: ArmSpec):
        self._batched_kinematics()

        J = self._jacobian(spec)                 # (nworld, 6, ndof)
        twist, q_arm, qd_arm, h_arm = self._mj_warp_state(spec)

        # M^{-1} h
        rhs_h = np.zeros((self.nw, self.nv), dtype=np.float32)
        rhs_h[:, spec.dof_ids] = h_arm
        Minv_h_full = self._solve_m(rhs_h)
        Minv_h = Minv_h_full[:, spec.dof_ids]

        # M^{-1} J^T, one RHS per task dimension
        Minv_Jt = np.zeros((self.nw, len(spec.dof_ids), 6), dtype=np.float32)
        for k in range(6):
            rhs = np.zeros((self.nw, self.nv), dtype=np.float32)
            rhs[:, spec.dof_ids] = J[:, k, :]
            sol = self._solve_m(rhs)
            Minv_Jt[:, :, k] = sol[:, spec.dof_ids]

        # Operational-space inverse inertia
        Mx_inv = np.einsum("wij, wjk -> wik", J, Minv_Jt)  # (nworld, 6, 6)

        # Small 6x6 inversion per world
        Mx = np.empty_like(Mx_inv)
        for i in range(self.nw):
            Mx[i] = np.linalg.pinv(Mx_inv[i], rcond=1e-4)

        # Damping
        K = np.diag(spec.K) if spec.K.ndim == 1 else spec.K
        KI = np.diag(spec.KI) if spec.KI.ndim == 1 else spec.KI
        D = np.empty((self.nw, 6, 6), dtype=np.float32)
        sqrtK = np.diag(np.sqrt(np.diag(K)).astype(np.float32))
        for i in range(self.nw):
            eigvals, eigvecs = np.linalg.eigh(Mx[i])
            sqrtMx = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
            D[i] = sqrtMx @ sqrtK + sqrtK @ sqrtMx

        # Jdot * qdot via finite difference
        if self.prev_J[spec.name] is None:
            Jdot_qd = np.zeros((self.nw, 6), dtype=np.float32)
        else:
            Jdot = (J - self.prev_J[spec.name]) / self.dt
            Jdot_qd = np.einsum("wij,wj->wi", Jdot, qd_arm)

        self.prev_J[spec.name] = J.copy()

        xd = np.einsum("wij,wj->wi", J, qd_arm)
        JMinv_h = np.einsum("wij,wj->wi", J, Minv_h)
        mu = np.einsum("wij,wj->wi", Mx, JMinv_h + Jdot_qd)

        F = (
            -np.einsum("wij,wj->wi", D, xd)
            + twist @ K.T
            + self.twist_int[spec.name] @ KI.T
            + mu
        )

        tau = np.einsum("wij,wi->wj", J, F)

        # Write torques back to the batched control buffer
        ctrl = self.data.ctrl.numpy()  # (nworld, na)
        ctrl[:, spec.act_ids] = tau
        self.data.ctrl = wp.array(ctrl.astype(np.float32), dtype=wp.float32)
        

    def step(self):
        for spec in self.specs:
            self.step_arm(spec)