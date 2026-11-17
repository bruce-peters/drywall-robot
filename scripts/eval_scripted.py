"""Evaluate the scripted controller over N seeds.

    python scripts/eval_scripted.py --n 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, ScriptedController  # noqa: E402


def run_episode(env, ctrl, seed):
    obs, info = env.reset(seed=seed)
    ctrl.reset()
    collided = False
    max_force = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        collided |= info["collision"]
        max_force = max(max_force, info["max_panel_wall_force"])
    return {
        "seed": seed,
        "success": info["success"],
        "dropped": info["dropped"],
        "truncated": truncated,
        "pos_error": info["pos_error"],
        "ori_error": info["ori_error"],
        "collision": collided,
        "max_force": max_force,
        "length": info["steps"],
        **info["randomization"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of episodes (seeds 0..n-1)")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--action-noise", type=float, default=0.0, help="scripted controller action noise std")
    args = ap.parse_args()

    env = DrywallPlacementEnv()
    ctrl = ScriptedController(env.cfg, action_noise_std=args.action_noise, seed=args.start_seed)
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.n):
        r = run_episode(env, ctrl, seed)
        rows.append(r)
        print(f"seed {seed:3d}: success={r['success']!s:5} dropped={r['dropped']!s:5} "
              f"pos_err={r['pos_error'] * 1000:6.2f} mm ori_err={np.degrees(r['ori_error']):5.2f} deg "
              f"collision={r['collision']!s:5} max_wall_F={r['max_force']:6.1f} N len={r['length']}")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n=== scripted controller, %d episodes ===" % len(rows))
    print(f"success rate          : {mean('success') * 100:.1f} %")
    print(f"drop rate             : {mean('dropped') * 100:.1f} %")
    print(f"mean final pos error  : {mean('pos_error') * 1000:.2f} mm")
    print(f"mean final ori error  : {np.degrees(mean('ori_error')):.3f} deg")
    print(f"collision rate        : {mean('collision') * 100:.1f} %  (episodes with >= 1 collision step)")
    print(f"mean episode length   : {mean('length'):.1f} steps")
    print(f"max panel-wall force  : mean {mean('max_force'):.1f} N, worst {max(r['max_force'] for r in rows):.1f} N")
    env.close()


if __name__ == "__main__":
    main()
