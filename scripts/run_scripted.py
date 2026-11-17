"""Run one scripted episode and save a video.

    python scripts/run_scripted.py --seed 0 --video out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, ScriptedController  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None, help="output .mp4 (omit for no video)")
    args = ap.parse_args()

    env = DrywallPlacementEnv(render_mode="rgb_array" if args.video else None)
    ctrl = ScriptedController(env.cfg)
    obs, info = env.reset(seed=args.seed)
    print("randomization:", {k: round(v, 4) for k, v in info["randomization"].items()})

    frames = [env.render()] if args.video else []
    phase, ret, totals = ctrl.phase, 0.0, {}
    terminated = truncated = False
    any_collision = False
    while not (terminated or truncated):
        action = ctrl.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        any_collision |= info["collision"]
        for k, v in info["reward_terms"].items():
            totals[k] = totals.get(k, 0.0) + v
        if ctrl.phase != phase:
            print(f"step {info['steps']:3d}: {phase} -> {ctrl.phase}  "
                  f"(pos_err {info['pos_error'] * 1000:.1f} mm, ori_err {np.degrees(info['ori_error']):.2f} deg)")
            phase = ctrl.phase
        if args.video:
            frames.append(env.render())

    print(f"\nterminated={terminated} truncated={truncated} steps={info['steps']} "
          f"success={info['success']} dropped={info['dropped']} collision={any_collision}")
    print(f"final pos_err {info['pos_error'] * 1000:.2f} mm, ori_err {np.degrees(info['ori_error']):.3f} deg, "
          f"return {ret:.2f}")
    print("reward terms (episode sums):", {k: round(v, 3) for k, v in totals.items()})

    if args.video:
        fps = env.metadata["render_fps"]
        # hold the last frame for 1 s so the final placement is visible
        frames += [frames[-1]] * fps
        imageio.mimsave(args.video, frames, fps=fps, macro_block_size=16)
        print(f"saved {args.video} ({len(frames)} frames @ {fps} fps)")
    env.close()


if __name__ == "__main__":
    main()
