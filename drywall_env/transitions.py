"""Shared learning interface for Parts B and C: the 37-D state, UMI -> robot action conversion,
reward relabelling and the transition-file format.

State (37-D, float32, everything in the robot base frame)
---------------------------------------------------------
    ee_pos 3 | ee_rot6 6 | attached 1 | panel_pos 3 | panel_rot6 6 | target_pos 3 | target_rot6 6
    | panel_in_target_pos 3 | panel_in_target_rot6 6

rot6 = the first two columns of the rotation matrix, [R[:,0], R[:,1]] (Zhou et al., "On the
Continuity of Rotation Representations in Neural Networks", CVPR 2019; UMI's `rotation_6d`).
Quaternions are not used because the tool-down EE orientation has w ~ 0, where a w >= 0 convention
flips sign from step to step (LABNOTEBOOK 2026-11-17).

The SAME function, `build_state`, is used for UMI data (`state_from_umi_rows`) and for a live env
(`state_from_obs`, deployable observations only), so a policy trained on either runs in the env.

Transition file (.npz, D4RL-style; one row per transition s_t, a_t -> r_t, s_t+1)
--------------------------------------------------------------------------------
    observations        (N, 37) float32
    actions             (N, 7)  float32   env action: [dx dy dz droll dpitch dyaw attach] in [-1, 1]
    rewards             (N,)    float32   sum of the stored reward terms
    next_observations   (N, 37) float32
    terminals           (N,)    bool      true episode end (success or drop): no bootstrapping
    timeouts            (N,)    bool      episode cut off without a terminal (time limit / failure)
    reward/<term>       (N,)    float32   each reward term separately (relabel = sum a subset)
    episode_ends        (E,)    int64     cumulative transition counts
    episode_success     (E,)    bool
    episode_seed        (E,)    int64     env seed (only to check that test seeds are not in training)
    episode_info/<key>  (E,)    any       optional per-episode labels (e.g. behaviour policy, noise std)
    meta_json           ()      str       state / action layout, dt, source, which reward terms are real
"""

from __future__ import annotations

import json

import numpy as np

from .env import EnvConfig

STATE_LAYOUT = [
    ("ee_pos", 3), ("ee_rot6", 6), ("attached", 1),
    ("panel_pos", 3), ("panel_rot6", 6),
    ("target_pos", 3), ("target_rot6", 6),
    ("panel_in_target_pos", 3), ("panel_in_target_rot6", 6),
]
STATE_DIM = sum(n for _, n in STATE_LAYOUT)  # 37
ACTION_DIM = 7
REWARD_TERMS = ("pos_error", "ori_error", "collision", "contact_force", "action", "jerk", "drop", "success")
# What UMI-style data supports: no force / contact sensing, so no collision, contact_force or drop.
UMI_REWARD_TERMS = ("pos_error", "ori_error", "action", "jerk", "success")

# Seed allocation (LABNOTEBOOK 2026-11-17). Test seeds are never used for data or training.
TEST_SEEDS = range(10000, 10100)
VAL_SEEDS = range(5000, 5050)
OFFLINE_SEEDS = range(1000, 2000)


def check_not_test_seeds(seeds) -> None:
    bad = sorted(set(int(s) for s in seeds) & set(TEST_SEEDS))
    if bad:
        raise ValueError(f"seeds {bad[:5]}... are reserved held-out TEST seeds (10000-10099)")


def state_slices() -> dict[str, slice]:
    out, i = {}, 0
    for name, n in STATE_LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


# ---------------------------------------------------------------- batched rotation helpers
def quat_to_mat_batch(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3)."""
    q = np.asarray(q, float).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def mat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,4) wxyz with w >= 0 (via MuJoCo, robust near 180 deg)."""
    import mujoco

    R = np.asarray(R, float).reshape(-1, 3, 3)
    out = np.zeros((len(R), 4))
    for i, r in enumerate(R):
        mujoco.mju_mat2Quat(out[i], r.reshape(9))
    out[out[:, 0] < 0] *= -1
    return out


def rotvec_to_mat_batch(v: np.ndarray) -> np.ndarray:
    """(N,3) axis-angle -> (N,3,3) (Rodrigues)."""
    v = np.asarray(v, float).reshape(-1, 3)
    th = np.linalg.norm(v, axis=1)
    k = np.where(th[:, None] > 1e-12, v / np.maximum(th, 1e-12)[:, None], 0.0)
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def mat_to_rot6(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,6) = [first column, second column]."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def rot6_to_mat(r6: np.ndarray) -> np.ndarray:
    """(N,6) -> (N,3,3) by Gram-Schmidt (Zhou et al. 2019, eq. for the 6D representation)."""
    r6 = np.asarray(r6, float).reshape(-1, 6)
    a, b = r6[:, :3], r6[:, 3:]
    c1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - np.sum(c1 * b, axis=1, keepdims=True) * c1
    c2 = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=2)


def mat_to_rpy_batch(R: np.ndarray) -> np.ndarray:
    """Inverse of transforms.rpy_to_mat (R = Rz(yaw) Ry(pitch) Rx(roll)), batched."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([roll, pitch, yaw], axis=1)


def rotation_angle_batch(R: np.ndarray) -> np.ndarray:
    """Geodesic angle of each rotation, in [0, pi] (same quantity as transforms.quat_angle)."""
    tr = np.trace(np.asarray(R, float).reshape(-1, 3, 3), axis1=1, axis2=2)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


# ---------------------------------------------------------------- state
def build_state(ee_p, ee_R, attached, panel_p, panel_R, targ_p, targ_R) -> np.ndarray:
    """All inputs batched ((N,3), (N,3,3), (N,)), base frame -> (N, 37) float32."""
    ee_p, panel_p, targ_p = (np.asarray(x, float).reshape(-1, 3) for x in (ee_p, panel_p, targ_p))
    ee_R, panel_R, targ_R = (np.asarray(x, float).reshape(-1, 3, 3) for x in (ee_R, panel_R, targ_R))
    att = np.asarray(attached, float).reshape(-1, 1)
    # panel pose in the target frame: p = R_t^T (p_p - p_t), R = R_t^T R_p
    RtT = np.transpose(targ_R, (0, 2, 1))
    rel_p = np.einsum("nij,nj->ni", RtT, panel_p - targ_p)
    rel_R = RtT @ panel_R
    s = np.concatenate([ee_p, mat_to_rot6(ee_R), att, panel_p, mat_to_rot6(panel_R),
                        targ_p, mat_to_rot6(targ_R), rel_p, mat_to_rot6(rel_R)], axis=1)
    assert s.shape[1] == STATE_DIM
    return s.astype(np.float32)


def state_from_obs(obs) -> np.ndarray:
    """37-D state from a live env observation, using ONLY the deployable group (the panel and
    target poses are the estimates, which carry observation noise if it is enabled)."""
    d = obs["deployable"] if "deployable" in obs else obs
    q = np.stack([d["ee_quat"], d["panel_quat_est"], d["target_quat_est"]])
    Rs = quat_to_mat_batch(q)
    return build_state(d["ee_pos"], Rs[0], d["attached"][0], d["panel_pos_est"], Rs[1],
                       d["target_pos_est"], Rs[2])[0]


# ---------------------------------------------------------------- UMI -> robot conversion
def world_to_base(T_base_world: np.ndarray, p: np.ndarray, R: np.ndarray):
    """Express world-frame poses (N,3),(N,3,3) in the robot base frame, T_base_world (4,4)."""
    Rbw, pbw = T_base_world[:3, :3], T_base_world[:3, 3]
    return np.einsum("ij,nj->ni", Rbw, p) + pbw, Rbw @ R


def resample_poses(t_src, pos, R, t_new):
    """Resample a pose stream to new timestamps: position linear, rotation slerp. Timestamps that
    coincide with a source sample (within 1 us) return that sample exactly."""
    t_src, t_new = np.asarray(t_src, float), np.asarray(t_new, float)
    q = mat_to_quat_batch(R)
    for i in range(1, len(q)):  # make the quaternion path continuous for slerp
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    idx = np.clip(np.searchsorted(t_src, t_new, side="right") - 1, 0, len(t_src) - 2)
    t0, t1 = t_src[idx], t_src[idx + 1]
    u = np.clip((t_new - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    u[np.abs(t_new - t0) < 1e-6] = 0.0
    u[np.abs(t_new - t1) < 1e-6] = 1.0
    p_new = pos[idx] * (1 - u)[:, None] + pos[idx + 1] * u[:, None]
    qa, qb = q[idx], q[idx + 1]
    dot = np.clip(np.sum(qa * qb, axis=1), -1.0, 1.0)
    th = np.arccos(dot)
    small = th < 1e-6
    sa = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, np.sin(th)))
    sb = np.where(small, u, np.sin(u * th) / np.where(small, 1, np.sin(th)))
    q_new = sa[:, None] * qa + sb[:, None] * qb
    return p_new, quat_to_mat_batch(q_new)


def resample_hold(t_src, values, t_new):
    """Zero-order hold (last sample at or before t): for discrete signals like the gripper."""
    idx = np.clip(np.searchsorted(np.asarray(t_src), np.asarray(t_new) + 1e-6, side="right") - 1,
                  0, len(t_src) - 1)
    return np.asarray(values)[idx]


def actions_from_poses(p, R, attached, cfg: EnvConfig, lag: int = 0):
    """Robot actions that move the EE along a base-frame pose trajectory (K rows -> K-1 actions).

    a_t[:3]  = (p_{t+1+lag} - p_{t+lag}) / max_pos_step
    a_t[3:6] = rpy(R_{t+1+lag} R_{t+lag}^T) / max_rot_step     (env applies R <- R_delta R)
    a_t[6]   = +1 if attached_{t+1} else -1                      (suction state after the step)
    lag shifts the motion target later to compensate tracking latency (UMI's latency matching);
    the last `lag` actions repeat the final pose (zero motion).
    Returns (actions (K-1,7) float32, raw_abs_max (K-1,) before clipping)."""
    K = len(p)
    i0 = np.minimum(np.arange(K - 1) + lag, K - 1)
    i1 = np.minimum(i0 + 1, K - 1)
    dp = p[i1] - p[i0]
    dR = R[i1] @ np.transpose(R[i0], (0, 2, 1))
    raw = np.concatenate([dp / cfg.max_pos_step, mat_to_rpy_batch(dR) / cfg.max_rot_step], axis=1)
    grip = np.where(np.asarray(attached[1:]) > 0.5, 1.0, -1.0)[:, None]
    a = np.concatenate([np.clip(raw, -1.0, 1.0), grip], axis=1)
    return a.astype(np.float32), np.abs(raw).max(axis=1)


def umi_reward_terms(next_states: np.ndarray, actions: np.ndarray, success_last: bool,
                     cfg: EnvConfig) -> dict[str, np.ndarray]:
    """The env's reward terms (env.py step 6) that UMI-style data supports, computed from s_{t+1}
    and a_t. collision / contact_force / drop need force or contact sensing: stored as 0."""
    w, sl = cfg.reward_weights, state_slices()
    n = len(actions)
    rel_p = next_states[:, sl["panel_in_target_pos"]].astype(float)
    rel_R = rot6_to_mat(next_states[:, sl["panel_in_target_rot6"]])
    a = actions[:, :6].astype(float)
    a_prev = np.vstack([np.zeros((1, 6)), a[:-1]])
    terms = {k: np.zeros(n) for k in REWARD_TERMS}
    terms["pos_error"] = -w["pos_error"] * np.linalg.norm(rel_p, axis=1)
    terms["ori_error"] = -w["ori_error"] * rotation_angle_batch(rel_R)
    terms["action"] = -w["action"] * np.sum(a * a, axis=1)
    terms["jerk"] = -w["jerk"] * np.sum((a - a_prev) ** 2, axis=1)
    if success_last:
        terms["success"][-1] = w["success"]
    return {k: v.astype(np.float32) for k, v in terms.items()}


# ---------------------------------------------------------------- file format
def save_transitions(path, episodes: list[dict], meta: dict) -> dict:
    """episodes: dicts with observations (T+1,37) [s_0..s_T], actions (T,7), reward_terms {term:(T,)},
    terminal (bool, last step), success (bool), seed (int). Writes the D4RL-style file."""
    obs, nxt, act, term, tout = [], [], [], [], []
    rterms = {k: [] for k in REWARD_TERMS}
    ends, succ, seeds = [], [], []
    info_keys = sorted({k for ep in episodes for k in ep.get("info", {})})
    infos = {k: [] for k in info_keys}
    for ep in episodes:
        for k in info_keys:
            infos[k].append(ep.get("info", {}).get(k))
        s, a = ep["observations"], ep["actions"]
        T = len(a)
        assert len(s) == T + 1 and T > 0
        obs.append(s[:-1])
        nxt.append(s[1:])
        act.append(a)
        t = np.zeros(T, bool)
        o = np.zeros(T, bool)
        (t if ep["terminal"] else o)[-1] = True
        term.append(t)
        tout.append(o)
        for k in REWARD_TERMS:
            rterms[k].append(np.asarray(ep["reward_terms"].get(k, np.zeros(T)), np.float32))
        ends.append(T)
        succ.append(bool(ep["success"]))
        seeds.append(int(ep["seed"]))
    r = {k: np.concatenate(v) for k, v in rterms.items()}
    included = meta.get("reward_terms_included", list(REWARD_TERMS))
    rewards = sum(r[k] for k in included).astype(np.float32)
    meta = {**meta, "state_layout": STATE_LAYOUT, "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "action_layout": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "attach"],
            "reward_terms_included": list(included)}
    arrays = {
        "observations": np.concatenate(obs).astype(np.float32),
        "actions": np.concatenate(act).astype(np.float32),
        "rewards": rewards,
        "next_observations": np.concatenate(nxt).astype(np.float32),
        "terminals": np.concatenate(term),
        "timeouts": np.concatenate(tout),
        "episode_ends": np.cumsum(ends).astype(np.int64),
        "episode_success": np.array(succ),
        "episode_seed": np.array(seeds, np.int64),
        "meta_json": np.str_(json.dumps(meta)),
        **{f"reward/{k}": v for k, v in r.items()},
        **{f"episode_info/{k}": np.array(v) for k, v in infos.items()},
    }
    np.savez_compressed(path, **arrays)
    return {"transitions": len(arrays["actions"]), "episodes": len(episodes)}


def load_transitions(path, reward_terms=None) -> dict:
    """Load a transition file into a dict of arrays. reward_terms (iterable) relabels `rewards`
    as the sum of those terms, e.g. UMI_REWARD_TERMS to make env data comparable with UMI data."""
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["meta"] = json.loads(str(z["meta_json"]))
    if reward_terms is not None:
        d["rewards"] = sum(d[f"reward/{k}"] for k in reward_terms).astype(np.float32)
        d["meta"]["reward_terms_included"] = list(reward_terms)
    return d


def episode_slices(episode_ends: np.ndarray) -> list[slice]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [slice(int(a), int(b)) for a, b in zip(starts, episode_ends)]
