"""Part B.B: convert UMI-style demonstrations into robot-compatible (state, action, reward) data.

Input: UMI's replay buffer (data/umi_sim.zarr.zip, see umi_zarr.py) -- absolute EE poses in the
SLAM/world frame, suction state as gripper_width, panel pose in the target frame, success labels --
plus the sim sidecar's target offsets (stand-in for a perception-detected target, LABNOTEBOOK
2026-11-17).

Steps per episode (each one is a requirement in the task doc, Part B.B):
 1. World -> robot base frame: p_b = R_bw p_w + t_bw, R_b = R_bw R_w (T_base_world from the cell
    calibration; the identity in this sim, but the code and the test use the general transform).
 2. Temporal resampling to the control rate (10 Hz): position linear, rotation slerp, gripper
    zero-order hold. The recording is already at 10 Hz, so this is exact here; it handles any rate.
 3. Absolute -> relative: actions are pose DELTAS between consecutive resampled poses, in the base
    frame, scaled by the env's per-step limits (2 cm, 3 deg) and clipped to [-1, 1].
 4. Gripper mapping: gripper_width (attached 0/1) -> attach bit +1 / -1 (state after the step).
 5. Workspace / reachability: EE positions outside the env's workspace box are counted and reported
    (the env would clip such commands); actions beyond the per-step limit are clipped and counted.
"""

from __future__ import annotations

import numpy as np

from .env import EnvConfig
from .transitions import (
    actions_from_poses,
    build_state,
    resample_hold,
    resample_poses,
    rotvec_to_mat_batch,
    umi_reward_terms,
    world_to_base,
)

LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width", "timestamp",
               "panel_pos_in_target", "panel_rot_axis_angle_in_target")


def load_umi_lowdim(zarr_path) -> dict:
    """Low-dim arrays + episode boundaries from UMI's zarr ZipStore (images are not read)."""
    import zarr

    store = zarr.ZipStore(str(zarr_path), mode="r")
    try:
        root = zarr.open_group(store=store, mode="r")
        out = {k: np.asarray(root["data"][k][:]) for k in LOWDIM_KEYS}
        out["episode_ends"] = np.asarray(root["meta"]["episode_ends"][:])
        out["episode_success"] = np.asarray(root["meta"]["episode_success"][:]).astype(bool)
    finally:
        store.close()
    return out


def cell_calibration(env):
    """T_base_world (4x4), the nominal target position and the target orientation (world).
    In a real cell these come from the robot-to-site calibration and the wall layout."""
    from .transforms import quat_to_mat

    bp, bq = env._base_pose()
    T_wb = np.eye(4)
    T_wb[:3, :3], T_wb[:3, 3] = quat_to_mat(bq), bp
    R_target = env.data.site_xmat[env._target_site].reshape(3, 3).copy()  # not randomized
    return np.linalg.inv(T_wb), env._nom_target_pos.copy(), R_target


def convert_episode(rows: dict, target_pos_w, target_R_w, T_base_world, cfg: EnvConfig,
                    success: bool, dt: float, lag: int = 0, state_dim: int = 37) -> dict:
    """One episode of UMI rows (world frame) -> states (T+1,37), actions (T,7), reward terms."""
    t = rows["timestamp"].astype(float)
    ee_R_w = rotvec_to_mat_batch(rows["robot0_eef_rot_axis_angle"])
    rel_R = rotvec_to_mat_batch(rows["panel_rot_axis_angle_in_target"])
    rel_p = rows["panel_pos_in_target"].astype(float)

    # panel pose in the world from the target pose and the panel-in-target pose
    n = len(t)
    tR = np.repeat(target_R_w[None], n, 0)
    tp = np.repeat(np.asarray(target_pos_w, float)[None], n, 0)
    panel_p_w = tp + np.einsum("nij,nj->ni", tR, rel_p)
    panel_R_w = tR @ rel_R

    # 1) world -> base
    ee_p, ee_R = world_to_base(T_base_world, rows["robot0_eef_pos"].astype(float), ee_R_w)
    pa_p, pa_R = world_to_base(T_base_world, panel_p_w, panel_R_w)
    ta_p, ta_R = world_to_base(T_base_world, tp, tR)

    # 2) resample to the control rate
    t_new = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-6)) + 1)
    ee_p, ee_R = resample_poses(t, ee_p, ee_R, t_new)
    pa_p, pa_R = resample_poses(t, pa_p, pa_R, t_new)
    ta_p, ta_R = resample_poses(t, ta_p, ta_R, t_new)
    att = resample_hold(t, rows["robot0_gripper_width"].reshape(-1), t_new)

    states = build_state(ee_p, ee_R, att, pa_p, pa_R, ta_p, ta_R, state_dim=state_dim)
    # 3) + 4) actions from consecutive poses, gripper mapping
    actions, raw_max = actions_from_poses(ee_p, ee_R, att, cfg, lag=lag)
    # 5) workspace check
    lo, hi = np.array(cfg.workspace_low), np.array(cfg.workspace_high)
    outside = np.any((ee_p < lo - 1e-9) | (ee_p > hi + 1e-9), axis=1)
    terms = umi_reward_terms(states[1:], actions, success, cfg)
    return {"observations": states, "actions": actions, "reward_terms": terms,
            "terminal": bool(success), "success": bool(success),
            "n_clipped": int(np.sum(raw_max > 1.0 + 1e-6)), "n_outside_workspace": int(outside.sum()),
            "ee_pos_base": ee_p}


def replay_actions(env, seed: int, actions: np.ndarray, recorded_ee_pos: np.ndarray | None = None):
    """Open-loop replay of converted actions in the env (validation of the conversion).
    Returns success / final errors and the mean EE distance to the recorded trajectory."""
    obs, _ = env.reset(seed=seed)
    errs, info, n = [], {}, 0
    for k, a in enumerate(actions):
        obs, _, term, trunc, info = env.step(a)
        n = k + 1
        if recorded_ee_pos is not None and k + 1 < len(recorded_ee_pos):
            errs.append(np.linalg.norm(obs["deployable"]["ee_pos"] - recorded_ee_pos[k + 1]))
        if term or trunc:
            break
    return {"success": bool(info.get("success", False)), "dropped": bool(info.get("dropped", False)),
            "pos_error": float(info.get("pos_error", np.nan)), "ori_error": float(info.get("ori_error", np.nan)),
            "steps": n, "ee_track_err": float(np.mean(errs)) if errs else np.nan}
