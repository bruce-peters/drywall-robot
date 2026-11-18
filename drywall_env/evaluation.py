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
    ret, collided, max_force, steps = 0.0, False, 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        a = policy.act(obs)
        obs, r, terminated, truncated, info = env.step(a)
        ret += r
        steps += 1
        collided |= info["collision"]
        max_force = max(max_force, info["max_panel_wall_force"])
        if on_step is not None:
            on_step(obs, a, info)
    return {"seed": int(seed), "success": bool(info["success"]), "dropped": bool(info["dropped"]),
            "truncated": bool(truncated), "pos_error_mm": 1000 * info["pos_error"],
            "ori_error_deg": float(np.degrees(info["ori_error"])), "collision": bool(collided),
            "max_wall_force": float(max_force), "length": steps, "return": float(ret)}


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
