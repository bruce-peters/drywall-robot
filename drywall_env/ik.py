"""Damped-least-squares inverse kinematics on a site, using mj_jacSite."""

from __future__ import annotations

import mujoco
import numpy as np

from .transforms import quat_error


class DLSIK:
    """Iterative DLS IK for the 7 arm joints, run on a private scratch MjData.

    dq = J^T (J J^T + lambda^2 I)^-1 e  +  null-space pull towards a home posture.
    e = [position error (m); orientation error as rotation vector (rad)], both in world
    (= robot base) frame.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: list[str],
        q_home: np.ndarray,
        damping: float = 0.05,
        iters: int = 10,
        max_step: float = 0.2,
        null_gain: float = 0.1,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        jids = [model.joint(j).id for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.q_home = np.asarray(q_home, float)
        self.damping = damping
        self.iters = iters
        self.max_step = max_step
        self.null_gain = null_gain
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and quaternion for arm configuration q."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return self.data.site_xpos[self.site_id].copy(), quat

    def solve(self, q_init: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (q, residual position error in m) for target site pose (pos, quat)."""
        q = np.clip(np.asarray(q_init, float).copy(), self.lo, self.hi)
        lam2 = self.damping**2
        for _ in range(self.iters):
            cur_pos, cur_quat = self.fk(q)
            err = np.concatenate([pos - cur_pos, quat_error(quat, cur_quat)])
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
            JJt = J @ J.T + lam2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Null-space pull towards home keeps the redundant DOF from drifting.
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
            dq += (np.eye(len(q)) - J_pinv @ J) @ (self.null_gain * (self.q_home - q))
            step = np.max(np.abs(dq))
            if step > self.max_step:
                dq *= self.max_step / step
            q = np.clip(q + dq, self.lo, self.hi)
        cur_pos, _ = self.fk(q)
        return q, float(np.linalg.norm(pos - cur_pos))
