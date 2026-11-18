"""Run one scripted episode and save a video.

    python scripts/run_scripted.py --seed 0 --video out.mp4
    python scripts/run_scripted.py --challenge hard --seed 1 --video media/hard_challenge/scripted_hard_seed1.mp4

--label burns a caption into every frame (default: "Scripted controller (no learning) | ...").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, EnvConfig  # noqa: E402
from drywall_env.scripted_hard import make_scripted  # noqa: E402
from drywall_env.video import label_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None, help="output .mp4 (omit for no video)")
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy")
    ap.add_argument("--region", choices=["full", "demo", "ood"], default="full", help="hard only")
    ap.add_argument("--mode", choices=["demo", "explore"], default="demo", help="hard only: controller mode")
    ap.add_argument("--label", type=str, default=None, help="caption burned into the video ('' = none)")
    args = ap.parse_args()

    cfg = EnvConfig(challenge=args.challenge, region=args.region if args.challenge == "hard" else "full")
    env = DrywallPlacementEnv(cfg, render_mode="rgb_array" if args.video else None)
    ctrl = make_scripted(env.cfg, mode=args.mode, seed=args.seed)
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
        label = args.label
        if label is None and args.challenge == "hard":
            outcome = "SUCCESS" if info["success"] else ("DROPPED" if info["dropped"] else "FAILED")
            label = (f"Scripted controller (no learning) | hard challenge, {args.mode} mode | seed {args.seed} | "
                     f"{outcome}")
        if label:
            frames = [label_frame(f, label) for f in frames]
        # hold the last frame for 1 s so the final placement is visible
        frames += [frames[-1]] * fps
        imageio.mimsave(args.video, frames, fps=fps, macro_block_size=16)
        print(f"saved {args.video} ({len(frames)} frames @ {fps} fps)")
    env.close()


if __name__ == "__main__":
    main()
