from dataclasses import dataclass
import numpy as np
import mujoco_warp as MW
import warp as wp
import torch
from torch import Tensor

NWORLD = 4


@dataclass
class ArmSpec:
    name: str
    body_id: int
    dof_ids: np.ndarray
    act_ids: np.ndarray
    mocap_id: int
    offset: float
    K: np.ndarray
    KI: np.ndarray
    K0: np.ndarray


# ── GPU quaternion helpers (no scipy, no CPU, no allocations in hot loop) ────

def _quat_inv(q: Tensor) -> Tensor:
    """Unit quaternion inverse (= conjugate). (..., 4) wxyz."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product. Both (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_to_rotvec(q: Tensor) -> Tensor:
    """
    Unit quaternion → rotation vector. (..., 4) wxyz → (..., 3).

    Numerically stable for small angles: uses L'Hôpital limit
    (2θ / sin(θ/2) → 2 as θ → 0) instead of dividing by a near-zero sin.
    """
    q       = torch.where(q[..., :1] < 0, -q, q)   # canonical: w ≥ 0
    w       = q[..., 0].clamp(-1.0, 1.0)
    xyz     = q[..., 1:]
    ha      = torch.acos(w)                          # half-angle ∈ [0, π/2]
    sin_ha  = torch.sin(ha).unsqueeze(-1)
    scale   = torch.where(
        sin_ha.abs() > 1e-7,
        2.0 * ha.unsqueeze(-1) / sin_ha,
        torch.full_like(sin_ha, 2.0),               # limit value
    )
    return scale * xyz


# ─────────────────────────────────────────────────────────────────────────────

class ParallelOSC:
    def __init__(self, model, data, specs, nworld=NWORLD):
        self.model = model
        self.data  = data
        self.nw    = nworld
        self.dt    = float(model.opt.timestep.numpy().mean())
        self.specs = specs

        nv = int(data.qvel.numpy().shape[-1])
        self.nv = nv

        self.twist_int = {
            s.name: np.zeros((nworld, 6), dtype=np.float32) for s in specs
        }

        # ── 1. Per-arm warp + numpy allocations ───────────────────────────────
        self._jacp_wp   = {}
        self._jacr_wp   = {}
        self._jacp_th   = {}
        self._jacr_th   = {}
        self._jacp_pin  = {}
        self._jacr_pin  = {}
        self._points_wp = {}
        self._bodies_wp = {}
        self._bufs      = {}

        for s in specs:
            nd = len(s.dof_ids)

            jacp_wp = wp.zeros((nworld, 3, nv), dtype=wp.float32)
            jacr_wp = wp.zeros((nworld, 3, nv), dtype=wp.float32)
            self._jacp_wp[s.name]  = jacp_wp
            self._jacr_wp[s.name]  = jacr_wp
            self._jacp_th[s.name]  = wp.to_torch(jacp_wp)
            self._jacr_th[s.name]  = wp.to_torch(jacr_wp)
            self._jacp_pin[s.name] = torch.empty(nworld, 3, nv, dtype=torch.float32).pin_memory()
            self._jacr_pin[s.name] = torch.empty(nworld, 3, nv, dtype=torch.float32).pin_memory()

            self._points_wp[s.name] = wp.array(
                np.tile([0.0, 0.0, s.offset], (nworld, 1)).astype(np.float32),
                dtype=wp.vec3f,
            )
            self._bodies_wp[s.name] = wp.array(
                np.full(nworld, s.body_id, dtype=np.int32), dtype=wp.int32
            )

            K     = (np.diag(s.K)  if s.K.ndim  == 1 else s.K ).astype(np.float32)
            KI    = (np.diag(s.KI) if s.KI.ndim == 1 else s.KI).astype(np.float32)
            sqrtK = np.diag(np.sqrt(np.diag(K)).astype(np.float32))

            self._bufs[s.name] = dict(
                nd       = nd,
                K        = K,
                KI       = KI,
                sqrtK    = sqrtK,
                J        = np.empty((nworld, 6, nd), dtype=np.float32),
                J_prev   = np.zeros((nworld, 6, nd), dtype=np.float32),
                has_prev = False,
                sol      = np.empty((nworld, nv),    dtype=np.float32),
                Minv_h   = np.empty((nworld, nd),    dtype=np.float32),
                rhs      = np.zeros((nworld, nv),    dtype=np.float32),
                Minv_Jt  = np.zeros((nworld, nd, 6), dtype=np.float32),
                Mx_inv   = np.empty((nworld, 6, 6),  dtype=np.float32),
                Mx       = np.empty((nworld, 6, 6),  dtype=np.float32),
                D        = np.empty((nworld, 6, 6),  dtype=np.float32),
                Jdot     = np.empty((nworld, 6, nd), dtype=np.float32),
                Jdot_qd  = np.zeros((nworld, 6),     dtype=np.float32),
                xd       = np.empty((nworld, 6),     dtype=np.float32),
                JMinv_h  = np.empty((nworld, 6),     dtype=np.float32),
                mu       = np.empty((nworld, 6),     dtype=np.float32),
                F        = np.empty((nworld, 6),     dtype=np.float32),
                tau      = np.empty((nworld, nd),    dtype=np.float32),
            )

        # ── 2. Shared solve buffers ────────────────────────────────────────────
        self._rhs_wp  = wp.zeros((nworld, nv), dtype=wp.float32)
        self._sol_wp  = wp.zeros((nworld, nv), dtype=wp.float32)
        self._rhs_th  = wp.to_torch(self._rhs_wp)
        self._sol_th  = wp.to_torch(self._sol_wp)
        self._sol_pin = torch.empty(nworld, nv, dtype=torch.float32).pin_memory()

        # ── 3. Zero-copy GPU views of simulator state ─────────────────────────
        # xpos / xquat / xmat / mocap are now used ONLY on GPU for twist; they
        # never need a full copy to CPU anymore.  qpos / qvel / qfrc_bias still
        # come to CPU for the Jacobian & bias arithmetic.
        self._data_th: dict[str, Tensor] = {
            "xpos":       wp.to_torch(data.xpos),
            "xquat":      wp.to_torch(data.xquat),
            "mocap_pos":  wp.to_torch(data.mocap_pos),
            "mocap_quat": wp.to_torch(data.mocap_quat),
            "qpos":       wp.to_torch(data.qpos),
            "qvel":       wp.to_torch(data.qvel),
            "qfrc_bias":  wp.to_torch(data.qfrc_bias),
        }

        # xmat needs a shape check: wp.to_torch of a mat33 field may come out
        # as (nworld, nbody, 9); reshape to (nworld, nbody, 3, 3) if needed.
        _xmat_np_shape = data.xmat.numpy().shape   # one-time, in __init__ only
        _xmat_th       = wp.to_torch(data.xmat)
        if _xmat_th.shape != torch.Size(_xmat_np_shape):
            _xmat_th = _xmat_th.view(_xmat_np_shape)
        self._data_th["xmat"] = _xmat_th           # (nworld, nbody, 3, 3)

        # Pinned CPU buffers ONLY for the three fields needed on CPU.
        # Everything else (the six pose fields above) stays on GPU permanently.
        _cpu_keys = ("qpos", "qvel", "qfrc_bias")
        self._st_pin: dict[str, Tensor] = {
            k: torch.empty_like(self._data_th[k], device="cpu").pin_memory()
            for k in _cpu_keys
        }
        self._st: dict[str, np.ndarray] = {}

        # ── 4. Per-arm pinned twist landing zones (nworld, 6) ─────────────────
        # Only 96 bytes per arm arrives on CPU, vs the full xpos/xmat/mocap
        # arrays that used to be pulled for every arm every step.
        self._twist_pin = {
            s.name: torch.empty(nworld, 6, dtype=torch.float32).pin_memory()
            for s in specs
        }

        # ── 5. In-place ctrl writes ────────────────────────────────────────────
        self._ctrl_th    = wp.to_torch(data.ctrl)
        ctrl_device      = self._ctrl_th.device
        self._act_ids_th = {
            s.name: torch.tensor(s.act_ids.tolist(), device=ctrl_device)
            for s in specs
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _batched_kinematics(self):
        MW.kinematics(self.model, self.data)
        MW.com_vel(self.model, self.data)
        MW.crb(self.model, self.data)
        MW.factor_m(self.model, self.data)
        MW.rne(self.model, self.data, flg_acc=False)

    def _read_state(self):
        """
        Transfer qpos / qvel / qfrc_bias to CPU via pinned memory.

        Compared to before: xpos (nworld×nbody×3), xquat (nworld×nbody×4),
        xmat (nworld×nbody×9), mocap_pos, mocap_quat are NO LONGER transferred
        here — they are consumed entirely on GPU inside _compute_twist_gpu.
        """
        for k, pin in self._st_pin.items():
            pin.copy_(self._data_th[k], non_blocking=True)
        torch.cuda.synchronize()
        self._st = {k: v.numpy() for k, v in self._st_pin.items()}

    def _compute_twist_gpu(
        self,
        spec: ArmSpec,
        waypoint_th: Tensor | None,
    ) -> np.ndarray:
        """
        Compute 6-DOF pose error entirely on GPU; return (nworld, 6) numpy
        via a (nworld*6*4 = 96 byte) pinned-memory transfer.

        waypoint_th — (nworld, 7) CUDA tensor: [pos(3) | quat_wxyz(4)].
                      Pass None to use the arm's mocap body target.

        The _data_th views are zero-copy aliases of the live warp arrays, so
        they reflect the values written by _batched_kinematics() with no copy.
        """
        xpos_th  = self._data_th["xpos"]    # (nworld, nbody, 3)
        xquat_th = self._data_th["xquat"]   # (nworld, nbody, 4) wxyz
        xmat_th  = self._data_th["xmat"]    # (nworld, nbody, 3, 3)

        # EE position: body origin shifted by 'offset' along its z-axis.
        ee_pos  = (xpos_th[:, spec.body_id, :]
                   + spec.offset * xmat_th[:, spec.body_id, :, 2])  # (nworld, 3)
        ee_quat = xquat_th[:, spec.body_id, :]                       # (nworld, 4) wxyz

        if waypoint_th is None:
            target_pos  = self._data_th["mocap_pos"][:, spec.mocap_id, :]
            target_quat = self._data_th["mocap_quat"][:, spec.mocap_id, :]
        else:
            target_pos  = waypoint_th[:, :3]    # (nworld, 3)
            target_quat = waypoint_th[:, 3:]    # (nworld, 4) wxyz

        pos_err = target_pos - ee_pos
        rot_err = _quat_to_rotvec(_quat_mul(target_quat, _quat_inv(ee_quat)))

        twist_th = torch.cat([pos_err, rot_err], dim=-1).float()  # (nworld, 6)

        pin = self._twist_pin[spec.name]
        pin.copy_(twist_th, non_blocking=True)
        torch.cuda.synchronize()
        return pin.numpy()   # zero-copy view of pinned memory

    def _jacobian_inplace(self, spec: ArmSpec):
        """Fill b['J'] in-place via pre-allocated warp arrays."""
        self._jacp_th[spec.name].zero_()
        self._jacr_th[spec.name].zero_()
        MW.jac(
            self.model, self.data,
            self._jacp_wp[spec.name], self._jacr_wp[spec.name],
            self._points_wp[spec.name], self._bodies_wp[spec.name],
        )
        jacp_pin = self._jacp_pin[spec.name]
        jacr_pin = self._jacr_pin[spec.name]
        jacp_pin.copy_(self._jacp_th[spec.name], non_blocking=True)
        jacr_pin.copy_(self._jacr_th[spec.name], non_blocking=True)
        torch.cuda.synchronize()
        b   = self._bufs[spec.name]
        ids = spec.dof_ids
        b["J"][:, :3, :] = jacp_pin.numpy()[:, :, ids]
        b["J"][:, 3:, :] = jacr_pin.numpy()[:, :, ids]

    def _solve_m(self, rhs_np: np.ndarray, out_np: np.ndarray):
        """In-place solve with zero warp-array allocations."""
        self._rhs_th.copy_(torch.from_numpy(rhs_np))
        MW.solve_m(self.model, self.data, self._sol_wp, self._rhs_wp)
        self._sol_pin.copy_(self._sol_th, non_blocking=True)
        torch.cuda.synchronize()
        np.copyto(out_np, self._sol_pin.numpy())

    def step_arm(self, spec: ArmSpec, waypoint_th: Tensor | None = None):
        b   = self._bufs[spec.name]
        st  = self._st
        ids = spec.dof_ids
        J   = b["J"]

        # twist: stays on GPU until the final (nworld, 6) lands on CPU
        twist = self._compute_twist_gpu(spec, waypoint_th)

        self._jacobian_inplace(spec)

        q_arm  = st["qpos"][:, ids]
        qd_arm = st["qvel"][:, ids]
        h_arm  = st["qfrc_bias"][:, ids]

        # ── M^{-1} h ──────────────────────────────────────────────────────────
        b["rhs"].fill(0.0)
        b["rhs"][:, ids] = h_arm
        self._solve_m(b["rhs"], b["sol"])
        np.copyto(b["Minv_h"], b["sol"][:, ids])

        # ── M^{-1} J^T ────────────────────────────────────────────────────────
        for k in range(6):
            b["rhs"].fill(0.0)
            b["rhs"][:, ids] = J[:, k, :]
            self._solve_m(b["rhs"], b["sol"])
            b["Minv_Jt"][:, :, k] = b["sol"][:, ids]

        # ── Op-space inertia ───────────────────────────────────────────────────
        np.einsum("wij,wjk->wik", J, b["Minv_Jt"], out=b["Mx_inv"])
        for i in range(self.nw):
            b["Mx"][i] = np.linalg.pinv(b["Mx_inv"][i], rcond=1e-4)

        # ── Damping ────────────────────────────────────────────────────────────
        sqrtK = b["sqrtK"]
        for i in range(self.nw):
            eigvals, eigvecs = np.linalg.eigh(b["Mx"][i])
            sqrtMx    = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
            b["D"][i] = sqrtMx @ sqrtK + sqrtK @ sqrtMx

        # ── Jdot * qd ─────────────────────────────────────────────────────────
        if not b["has_prev"]:
            b["Jdot_qd"].fill(0.0)
            b["has_prev"] = True
        else:
            np.subtract(J, b["J_prev"], out=b["Jdot"])
            b["Jdot"] /= self.dt
            np.einsum("wij,wj->wi", b["Jdot"], qd_arm, out=b["Jdot_qd"])
        np.copyto(b["J_prev"], J)

        # ── OSC wrench ─────────────────────────────────────────────────────────
        np.einsum("wij,wj->wi", J,       qd_arm,      out=b["xd"])
        np.einsum("wij,wj->wi", J,       b["Minv_h"], out=b["JMinv_h"])
        np.add(b["JMinv_h"], b["Jdot_qd"],             out=b["mu"])
        np.einsum("wij,wj->wi", b["Mx"], b["mu"],      out=b["mu"])

        np.einsum("wij,wj->wi", b["D"],  b["xd"],      out=b["F"])
        b["F"] *= -1.0
        b["F"] += twist                      @ b["K"].T
        b["F"] += self.twist_int[spec.name]  @ b["KI"].T
        b["F"] += b["mu"]

        # ── tau → ctrl in-place (no ctrl read, no new wp.array) ───────────────
        np.einsum("wij,wi->wj", J, b["F"], out=b["tau"])
        self._ctrl_th[:, self._act_ids_th[spec.name]] = torch.from_numpy(b["tau"]).to(
            device=self._ctrl_th.device, non_blocking=True
        )

    def step(self, waypoint: Tensor | None = None):
        """
        waypoint — (num_arms, nworld, 7) CUDA tensor: [pos(3) | quat_wxyz(4)].
                   Pass None to use each arm's mocap body target.

        Calling convention (replaces the old .cpu().numpy() pattern):

            # Before:
            osc.step(traj[i].cpu().numpy())

            # After:
            osc.step(traj[i])   # traj[i] is (num_arms, nworld, 7) on CUDA
        """
        self._batched_kinematics()
        # Ensure kinematics writes are visible to torch before reading _data_th.
        wp.synchronize()
        self._read_state()   # pulls only qpos / qvel / qfrc_bias to CPU
        for i, spec in enumerate(self.specs):
            wp_i = waypoint[i] if waypoint is not None else None
            self.step_arm(spec, wp_i)