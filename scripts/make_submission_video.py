"""Assemble the 2-5 minute submission video from the labelled clips in media/, with title cards that
explain each clip and interpret the failures.

    python scripts/make_submission_video.py            # -> media/submission_video.mp4

Clips play at 2x (every second frame at 10 fps); title cards are 3.5 s. Every clip also carries its own
burned-in label (policy, test seed, outcome), except the original Part A clip, which the card introduces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FPS = 10
W, H = 1280, 480

# (clip, speed-up, title-card lines). The first line is the heading.
STORY = [
    (None, 1, ["Learning drywall panel placement",
               "UMI-style demonstrations  +  simulation-based offline RL",
               "Franka Panda with a suction plate, MuJoCo. Clips play at 2x speed; labels are burned in."]),
    ("media/scripted_seed0.mp4", 2,
     ["1. The environment (easy challenge) - scripted controller, no learning",
      "Pick the panel off the rack, rotate it, press it flush onto the wall target, release onto the ledge.",
      "This controller also plays the 'human' that records the UMI-style demonstrations."]),
    ("media/hard_challenge/scripted_hard_seed1.mp4", 2,
     ["2. Hard challenge - wall and rack anywhere around the robot",
      "Same scripted demonstrator, now swinging around the base (rack behind the robot here).",
      "Demos cover only part of these layouts; simulation covers all of them."]),
    ("media/easy_challenge/bc_umi_h8_success.mp4", 2,
     ["3. Learned policy: behaviour cloning on the UMI demos (easy challenge)",
      "MLP with 8-step action chunks; 80 % success on 100 held-out test layouts."]),
    ("media/easy_challenge/bc_umi_h8_failure.mp4", 2,
     ["FAILURE: behaviour cloning stops just above the panel and never grasps",
      "Near the goal the demonstrated motions are tiny, so the regressed motion averages to ~0",
      "before the state where the demos switch suction on - a BC fixed point."]),
    ("media/easy_challenge/iql_umi_sim_b3_success.mp4", 2,
     ["4. Offline RL: IQL (implemented from scratch) on UMI + simulated data",
      "Removes most grasp failures, but 71 % overall - not better than BC (80 %)."]),
    ("media/easy_challenge/iql_umi_sim_b3_failure.mp4", 2,
     ["FAILURE: IQL places the panel but releases it more than 2 mm off the wall",
      "RL changed how the policy fails more than how often: release timing is one bit,",
      "judged by a sparse success reward several steps later."]),
    ("media/hard_challenge/bc_umi_h8_in_failure.mp4", 2,
     ["5. Hard challenge: BC trained on the demos alone",
      "6 % inside the demos' region, 0 % outside it (never grasps a rack it has not seen).",
      "FAILURE shown: suction released mid-air -> the panel drops."]),
    ("media/hard_challenge/bc_umi_sim_h8_ood_success.mp4", 2,
     ["6. Adding simulated data extends the policy beyond the demos",
      "BC on UMI + sim successes: 55 % inside, 20 % OUTSIDE the demos' region (vs 0 %).",
      "This layout was never covered by a demonstration."]),
    ("media/hard_challenge/iql_umi_sim_fix_b3_in_success.mp4", 2,
     ["7. Best offline-RL policy on the hard challenge, after diagnosing it",
      "Rescaled rewards, 50/50 demo/sim batches, up-weighted release, 1M steps:",
      "IQL 39 -> 65 % inside the demos' region (BC: 55 %), 22 -> 26 % outside."]),
    ("media/hard_challenge/iql_umi_sim_fix_b3_in_failure.mp4", 2,
     ["FAILURE: the panel is carried but never placed ('held at end')",
      "It ends a median ~13 cm from the target: an approach-precision failure, not reluctance to release.",
      "Training longer alone did not help (39 -> 33 %); outside the demos' region all learned policies stay below 35 %."]),
    (None, 1, ["Takeaways",
               "- Demos teach the motion; simulation is what covers layouts the demos never saw.",
               "- Tuned offline RL (IQL) matched or beat BC on the same data, but not significantly.",
               "- Every learned policy fails at the last centimetres and the release decision:",
               "  deploy the learned policy for gross motion + a force-controlled, verified contact/release.",
               "Code, data pipeline, results and lab notebook: see README.md / docs/technical_report.md"]),
]


def card(lines: list[str], seconds: float = 3.5) -> list[np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(img)

    def font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    y = 150 - 18 * max(0, len(lines) - 3)
    for i, line in enumerate(lines):
        f = font(34 if i == 0 else 23)
        color = (255, 120, 100) if line.startswith("FAILURE") else ((240, 240, 240) if i == 0 else (190, 200, 215))
        w = d.textlength(line, font=f)
        d.text(((W - w) / 2, y), line, fill=color, font=f)
        y += 58 if i == 0 else 36
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def clip(path: str, speed: int) -> list[np.ndarray]:
    frames = [f[..., :3] for f in imageio.get_reader(path)]
    frames = frames[::speed]
    out = []
    for f in frames:
        if f.shape[:2] != (H, W):  # pad or crop to the common size
            g = np.zeros((H, W, 3), np.uint8)
            h, w = min(H, f.shape[0]), min(W, f.shape[1])
            g[:h, :w] = f[:h, :w]
            f = g
        out.append(f)
    return out + [out[-1]] * 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media/submission_video.mp4")
    args = ap.parse_args()
    frames = []
    for path, speed, lines in STORY:
        frames += card(lines, 3.5 if path else 5.0)
        if path:
            if not Path(path).exists():
                raise SystemExit(f"missing clip {path}")
            frames += clip(path, speed)
    imageio.mimsave(args.out, frames, fps=FPS, macro_block_size=16)
    print(f"wrote {args.out}: {len(frames)} frames, {len(frames) / FPS / 60:.1f} min")


if __name__ == "__main__":
    main()
