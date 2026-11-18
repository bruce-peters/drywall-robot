"""Policy evaluation on held-out seeds: the task doc's metrics (Part C.C).

    success rate, final position / orientation error, collision rate, episode duration
    (+ drop rate, peak panel-wall force, return).

`run_episode` is shared by eval_policy.py, the training-time validation and the policy viewer.
"""

from __future__ import annotations

import numpy as np


def run_episode(env, policy, seed: int, on_step=None) -> dict:
    """Roll out one episode; on_step(obs, action, info) is called after every step (viewer, video)."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    ret, collided, max_force, steps, ever_attached = 0.0, False, 0.0, 0, False
    terminated = truncated = False
    while not (terminated or truncated):
        a = policy.act(obs)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        steps += 1
        collided |= info["collision"]
        ever_attached |= info["attached"]
        max_force = max(max_force, info["max_panel_wall_force"])
        if on_step is not None:
            on_step(obs, a, info)
    row = {"seed": int(seed), "success": bool(info["success"]), "dropped": bool(info["dropped"]),
           "truncated": bool(truncated), "pos_error_mm": 1000 * info["pos_error"],
           "ori_error_deg": float(np.degrees(info["ori_error"])), "collision": bool(collided),
           "max_wall_force": float(max_force), "length": steps, "return": float(ret),
           "ever_attached": bool(ever_attached), "attached_at_end": bool(info["attached"]),
           "wall_dist_mm": 1000 * float(info["panel_wall_dist"])}
    row["failure"] = classify_failure(row, env.cfg)
    return row


FAILURE_TYPES = ("dropped", "never_grasped", "held_at_end", "released_off_wall", "released_misaligned", "other")


def classify_failure(row: dict, cfg) -> str | None:
    """Failure mode of one episode (None on success), from its end state:
    dropped            the panel fell (floor, or unsupported for 3 steps)
    never_grasped      suction never engaged (panel still on the rack)
    held_at_end        still holding the panel when time ran out
    released_off_wall  released, but the panel rests > wall_contact_tol from the wall (no fixture)
    released_misaligned released against the wall, but outside 20 mm / 5 deg
    """
    if row["success"]:
        return None
    if row["dropped"]:
        return "dropped"
    if not row["ever_attached"]:
        return "never_grasped"
    if row["attached_at_end"]:
        return "held_at_end"
    if row["wall_dist_mm"] > 1000 * cfg.wall_contact_tol:
        return "released_off_wall"
    if row["pos_error_mm"] > 1000 * cfg.success_pos_tol or row["ori_error_deg"] > np.degrees(cfg.success_ori_tol):
        return "released_misaligned"
    return "other"


def summarize(rows: list[dict]) -> dict:
    """Aggregate metrics. Errors are reported over all episodes and over successful ones."""
    if not rows:
        return {}
    f = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    succ = f("success").astype(bool)
    out = {
        "n": len(rows),
        "success_rate": float(succ.mean()),
        "drop_rate": float(f("dropped").mean()),
        "collision_rate": float(f("collision").mean()),
        "pos_error_mm_mean": float(f("pos_error_mm").mean()),
        "pos_error_mm_median": float(np.median(f("pos_error_mm"))),
        "ori_error_deg_mean": float(f("ori_error_deg").mean()),
        "length_mean": float(f("length").mean()),
        "length_success_mean": float(f("length")[succ].mean()) if succ.any() else None,
        "max_wall_force_mean": float(f("max_wall_force").mean()),
        "return_mean": float(f("return").mean()),
    }
    if "failure" in rows[0]:
        out["failures"] = {k: int(sum(r.get("failure") == k for r in rows)) for k in FAILURE_TYPES}
    # 95 % Wilson interval for the success rate (small-sample honest error bar)
    n, p, z = len(rows), out["success_rate"], 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    out["success_ci95"] = [float(max(0.0, c - h)), float(min(1.0, c + h))]
    return out


def evaluate(env, policy, seeds, progress=None) -> tuple[list[dict], dict]:
    rows = []
    for i, s in enumerate(seeds):
        rows.append(run_episode(env, policy, int(s)))
        if progress:
            progress(i, rows[-1])
    return rows, summarize(rows)
