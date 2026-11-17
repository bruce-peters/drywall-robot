"""Small rotation helpers. Quaternions are MuJoCo convention: [w, x, y, z]."""

from __future__ import annotations

import mujoco
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, float).reshape(9))
    return canonical(q)


def canonical(q: np.ndarray) -> np.ndarray:
    """Unit quaternion with w >= 0 (q and -q are the same rotation)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0, 0, 0])
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / angle, angle)
    return q


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, canonical(q), 1.0)
    return v


def quat_error(q_target: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that rotates q onto q_target."""
    return quat_to_rotvec(quat_mul(q_target, quat_conj(q)))


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two orientations, in [0, pi]."""
    d = abs(float(np.dot(canonical(a), canonical(b))))
    return 2.0 * float(np.arccos(min(1.0, d)))


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed-axis roll, pitch, yaw about x, y, z)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def mat_to_rpy(m: np.ndarray) -> np.ndarray:
    """Inverse of rpy_to_mat."""
    m = np.asarray(m, float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
    roll = np.arctan2(m[2, 1], m[2, 2])
    yaw = np.arctan2(m[1, 0], m[0, 0])
    return np.array([roll, pitch, yaw])


def pose_inv(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qc = quat_conj(quat)
    return -quat_to_mat(qc) @ pos, qc


def pose_mul(p1, q1, p2, q2) -> tuple[np.ndarray, np.ndarray]:
    """(p1, q1) * (p2, q2): compose frame 2 expressed in frame 1."""
    return np.asarray(p1) + quat_to_mat(q1) @ np.asarray(p2), canonical(quat_mul(q1, q2))
