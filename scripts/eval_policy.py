"""Evaluate policies on the held-out TEST seeds (10000-10099) and print the task doc's metrics table.

    python scripts/eval_policy.py --policy scripted checkpoints/bc_umi.pt
    python scripts/eval_policy.py --policy checkpoints/bc_umi.pt --video media/bc_umi

--video PREFIX saves PREFIX_success.mp4 and PREFIX_failure.mp4 (first test seed of each kind, if any),
with the policy name, seed and outcome burned into the frames. Results go to results/<name>.json.
Test seeds are used ONLY here and in the policy viewer -- never for data or training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv  # noqa: E402
from drywall_env.evaluation import evaluate, run_episode  # noqa: E402
from drywall_env.policies import ScriptedPolicy, load_policy  # noqa: E402
from drywall_env.transitions import TEST_SEEDS  # noqa: E402

LABELS = {"scripted": "Scripted controller (no learning)"}


def make_policy(spec: str, cfg):
    if spec == "scripted":
        return ScriptedPolicy(cfg)
    return load_policy(spec)


def label_frame(frame: np.ndarray, text: str, color) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=font)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    d.text((12, 10), text, fill=color, font=font)
    return np.asarray(img)


def save_video(policy, seed: int, path: str, name: str):
    import imageio.v2 as imageio

    env = DrywallPlacementEnv(render_mode="rgb_array")
    frames, state = [], {}

    def on_step(obs, a, info):
        state["info"] = info
        frames.append(env.render())

    env.reset(seed=seed)
    frames.append(env.render())
    row = run_episode(env, policy, seed, on_step=on_step)
    outcome = "SUCCESS" if row["success"] else ("DROPPED" if row["dropped"] else "FAILED (time limit)")
    color = (90, 230, 90) if row["success"] else (255, 110, 90)
    text = f"{name} | test seed {seed} | {outcome} | pos err {row['pos_error_mm']:.0f} mm"
    frames = [label_frame(f, text, color) for f in frames]
    fps = env.metadata["render_fps"]
    frames += [frames[-1]] * fps
    imageio.mimsave(path, frames, fps=fps, macro_block_size=16)
    env.close()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", required=True, help="'scripted' or checkpoint paths")
    ap.add_argument("--n", type=int, default=len(TEST_SEEDS), help="number of test seeds (from 10000)")
    ap.add_argument("--video", default=None, help="prefix for success / failure videos (single policy)")
    ap.add_argument("--name", default=None, help="display name in the video label")
    ap.add_argument("--failure-kind", default=None,
                    help="failure video shows the first episode of this mode (e.g. released_off_wall)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    seeds = list(TEST_SEEDS)[: args.n]
    env = DrywallPlacementEnv()
    table = []
    for spec in args.policy:
        pol = make_policy(spec, env.cfg)
        name = Path(spec).stem if spec != "scripted" else "scripted"
        t0 = time.time()
        rows, s = evaluate(env, pol, seeds)
        s["seconds"] = time.time() - t0
        Path(args.results_dir).mkdir(exist_ok=True)
        with open(Path(args.results_dir) / f"{name}.json", "w") as f:
            json.dump({"policy": spec, "seeds": [seeds[0], seeds[-1]], "summary": s, "episodes": rows}, f, indent=1)
        table.append((name, s))
        print(f"{name}: success {100 * s['success_rate']:.1f} % "
              f"(95% CI {100 * s['success_ci95'][0]:.0f}-{100 * s['success_ci95'][1]:.0f}), {s['seconds']:.0f} s",
              flush=True)
        if args.video:
            label = args.name or LABELS.get(name, name)
            for kind in ("success", "failure"):
                pick = [r["seed"] for r in rows if r["success"] == (kind == "success")
                        and (kind == "success" or args.failure_kind in (None, r.get("failure")))]
                if pick:
                    save_video(pol, pick[0], f"{args.video}_{kind}.mp4", label)
                    print(f"  saved {args.video}_{kind}.mp4 (seed {pick[0]})")
    env.close()

    print(f"\n| policy | success | 95% CI | pos err (mm) | ori err (deg) | collision | drop | length (steps) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in table:
        print(f"| {name} | {100 * s['success_rate']:.0f} % | {100 * s['success_ci95'][0]:.0f}-"
              f"{100 * s['success_ci95'][1]:.0f} % | {s['pos_error_mm_mean']:.1f} | {s['ori_error_deg_mean']:.2f} | "
              f"{100 * s['collision_rate']:.0f} % | {100 * s['drop_rate']:.0f} % | {s['length_mean']:.0f} |")
    print("\nfailure modes (episodes):")
    for name, s in table:
        print(f"  {name:<18} " + "  ".join(f"{k} {v}" for k, v in s.get("failures", {}).items() if v))


if __name__ == "__main__":
    main()
