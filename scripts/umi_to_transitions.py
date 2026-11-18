"""Part B.B: convert the UMI-style zarr into a transition dataset (state, action, reward, next state).

    python scripts/umi_to_transitions.py                        # -> data/umi_transitions.npz
    python scripts/umi_to_transitions.py --replay 500           # + open-loop replay check

Output format: drywall_env/transitions.py. Conversion steps: drywall_env/umi_to_robot.py.
Rewards contain only the terms UMI-style data supports (no collision / contact force / drop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, EnvConfig  # noqa: E402
from drywall_env.transforms import quat_to_mat  # noqa: E402
from drywall_env.transitions import UMI_REWARD_TERMS, check_not_test_seeds, save_transitions  # noqa: E402
from drywall_env.umi_to_robot import (  # noqa: E402
    cell_calibration, convert_episode, load_umi_lowdim, replay_actions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/umi_sim.zarr.zip")
    ap.add_argument("--sidecar", default="data/umi_sim_sim_meta.npz",
                    help="sim metadata: target offsets (perception stand-in) and env seeds")
    ap.add_argument("--out", default="data/umi_transitions.npz")
    ap.add_argument("--lag", type=int, default=0, help="latency compensation in control steps")
    ap.add_argument("--replay", type=int, default=0, help="replay the first N episodes open-loop as a check")
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy")
    ap.add_argument("--region", choices=["full", "demo", "ood"], default="demo",
                    help="hard only: the region the recording used (needed to replay its layouts)")
    ap.add_argument("--state-dim", type=int, choices=[37, 46], default=None,
                    help="37 = state v1, 46 = v2 (+ panel in EE frame); default 37 easy / 46 hard")
    args = ap.parse_args()
    state_dim = args.state_dim or (46 if args.challenge == "hard" else 37)

    t0 = time.time()
    U = load_umi_lowdim(args.zarr)
    side = np.load(args.sidecar)
    seeds = side["seed"].astype(np.int64)
    check_not_test_seeds(seeds)
    ends = U["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    assert len(ends) == len(seeds), "zarr and sidecar disagree on the number of episodes"

    env = DrywallPlacementEnv(EnvConfig(challenge=args.challenge, region=args.region if args.challenge == "hard" else "full"))
    env.reset(seed=0)
    T_base_world, target_nom, target_R = cell_calibration(env)

    episodes, n_clip, n_out = [], 0, 0
    for i in range(len(ends)):
        sl = slice(starts[i], ends[i])
        rows = {k: U[k][sl] for k in U if k not in ("episode_ends", "episode_success")}
        if "target_pos_w" in side.files:  # recorder stored the full (perceived) target pose
            target_pos, target_Ri = side["target_pos_w"][i], quat_to_mat(side["target_quat_w"][i])
        else:  # older easy recordings: nominal target + its offsets
            target_pos = target_nom + np.array([0.0, side["target_dy"][i], side["target_dz"][i]])
            target_Ri = target_R
        ep = convert_episode(rows, target_pos, target_Ri, T_base_world, env.cfg,
                             bool(U["episode_success"][i]), env.dt, lag=args.lag, state_dim=state_dim)
        ep["seed"] = int(seeds[i])
        n_clip += ep["n_clipped"]
        n_out += ep["n_outside_workspace"]
        episodes.append(ep)

    n_trans = sum(len(e["actions"]) for e in episodes)
    summary = {
        "episodes": len(episodes), "transitions": n_trans,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "actions_clipped_frac": n_clip / n_trans,
        "ee_rows_outside_workspace": n_out,
        "lag": args.lag,
    }

    if args.replay:
        rep = []
        for i, ep in enumerate(episodes[: args.replay]):
            r = replay_actions(env, ep["seed"], ep["actions"], ep["ee_pos_base"])
            rep.append({**r, "recorded": ep["success"]})
            if (i + 1) % 50 == 0:
                print(f"  replayed {i + 1}/{args.replay}", flush=True)
        rs = np.array([r["success"] for r in rep])
        rec = np.array([r["recorded"] for r in rep])
        summary["replay"] = {
            "n": len(rep), "replay_success_rate": float(rs.mean()), "recorded_success_rate": float(rec.mean()),
            "outcome_agreement": float(np.mean(rs == rec)),
            "replay_success_given_recorded_success": float(rs[rec].mean()) if rec.any() else None,
            "mean_ee_tracking_error_mm": float(1000 * np.nanmean([r["ee_track_err"] for r in rep])),
        }

    meta = {"source": str(args.zarr), "sidecar": str(args.sidecar), "dt": env.dt, "challenge": args.challenge,
            "reward_terms_included": list(UMI_REWARD_TERMS),
            "notes": "UMI-style demos -> robot actions; target pose from sidecar (perception stand-in); "
                     "collision/contact_force/drop are 0 (not observable in UMI data)",
            "summary": summary}
    for ep in episodes:
        ep.pop("ee_pos_base")
    save_transitions(args.out, episodes, meta)
    env.close()
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
