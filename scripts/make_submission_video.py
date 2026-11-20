"""Assemble the 2-5 minute submission video from the labelled clips in media/, with title cards that
explain each clip and interpret the failures.

    python scripts/make_submission_video.py            # -> media/submission_video.mp4

Clips play at 2-4x (every n-th frame at 10 fps); title cards stay up long enough to read. Every clip also carries its own
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

# (clip, speed-up, title-card lines). The first line is the heading; long lines wrap on the card.
STORY = [
    (None, 1, ["Learning drywall panel placement",
               "Behaviour cloning and offline RL from UMI-style demos and simulation",
               "Everything here runs in MuJoCo, and the clips are sped up 2-4x."]),
    ("media/scripted_seed0.mp4", 2,
     ["1. The task (easy version)",
      "The robot picks the panel off the rack, turns it, presses it flat on the wall target and lets go onto the ledge.",
      "This is my scripted controller, not a learned policy. It also stands in for the human recording the UMI demos."]),
    ("media/hard_challenge/scripted_hard_seed1.mp4", 2,
     ["2. The hard version",
      "Here the wall and the rack can be anywhere around the robot (in this one the rack is behind it).",
      "The demos only cover part of these layouts, so anything outside that has to come from simulation."]),
    ("media/easy_challenge/bc_umi_h8_success.mp4", 2,
     ["3. Behaviour cloning on the UMI demos (easy version)",
      "A small network that predicts 8 actions at a time. It succeeds on 80 out of 100 test layouts."]),
    ("media/easy_challenge/bc_umi_h8_failure.mp4", 3,
     ["FAILURE: it stops just above the panel and never grabs it",
      "Close to the panel the demo motions are tiny, so the network's average motion there is basically zero,",
      "and it never reaches the point where the demos turn the suction on."]),
    ("media/easy_challenge/iql_umi_sim_b3_success.mp4", 2,
     ["4. Offline RL (IQL, which I implemented from scratch) on demos plus sim data",
      "It mostly fixes the grabbing problem, but overall it gets 71 % (BC got 80 %)."]),
    ("media/easy_challenge/iql_umi_sim_b3_failure.mp4", 3,
     ["FAILURE: the panel is in the right spot, but it lets go more than 2 mm off the wall",
      "RL changed how the policy fails more than how often. Letting go is a single on/off decision,",
      "and the success reward only shows up a few steps after it."]),
    ("media/hard_challenge/bc_umi_h8_in_failure.mp4", 2,
     ["5. Hard version: behaviour cloning on the demos alone",
      "6 % inside the region the demos cover and 0 % outside it (it never even grabs a rack it hasn't seen).",
      "FAILURE shown here: it turns the suction off in mid-air and drops the panel."]),
    ("media/hard_challenge/bc_umi_sim_h8_ood_success.mp4", 2,
     ["6. Adding sim data lets the policy work outside the demos",
      "With sim data, BC gets 55 % inside the demo region and 20 % outside it (up from 0 %).",
      "No demo ever covered this layout."]),
    ("media/hard_challenge/iql_umi_sim_fix_b3_in_success.mp4", 2,
     ["7. IQL again, once I figured out what was wrong with it",
      "I shrank the rewards, gave every batch half demos and half sim data, weighted the let-go steps more and trained longer.",
      "That took IQL from 39 % to 65 % inside the demo region (BC was at 55 %)."]),
    ("media/hard_challenge/iql_umi_sim_fix_b3_in_failure.mp4", 4,
     ["FAILURE: it carries the panel over but never places it",
      "These episodes end a median of about 13 cm from the target, so it's an approach problem, not the policy refusing to let go.",
      "Training longer on its own didn't help (it went from 39 % down to 33 %)."]),
    ("media/hard_challenge/bc_umi_sim_h8_wall_gmm_ood_success.mp4", 2,
     ["8. What actually made the big difference",
      "I rotated the policy's inputs and outputs into the wall's frame, and gave it a mixture output so it can commit to one way "
      "around the robot instead of averaging two.",
      "BC with both gets 93 % inside the demo region and 74 % outside it. This layout is one of the outside ones."]),
    ("media/hard_challenge/bc_umi_sim_h8_wall_gmm_in_failure.mp4", 4,
     ["FAILURE: so close, but 25 mm off target (the limit is 20 mm)",
      "It lets go flush on the ledge, just a bit too far to one side. Six of its seven failures inside the demo region "
      "look like this, and they're all off to the same side.",
      "This last bit of precision is where every learned policy I trained struggles."]),
    (None, 1, ["Takeaways",
               "- Demos teach the motion, and simulation covers the layouts the demos never saw.",
               "- How the policy sees the task (wall frame, mixture output) mattered more than BC vs offline RL.",
               "- With the same setup IQL and BC came out even (63 % vs 63 % inside the demo region).",
               "- On a real robot I'd use the learned policy for the big motions and a force-controlled,",
               "  checked contact and release for the last few centimetres.",
               "Code, results and the full lab notebook: github.com/bruce-peters/drywall-robot"]),
]


def card(lines: list[str], seconds: float | None = None) -> list[np.ndarray]:
    """A title card. Lines wider than the frame wrap; it stays up long enough to read (~4.5 words/s)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(img)

    def font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    rows = []  # (text, is_heading, colour)
    for i, line in enumerate(lines):
        f = font(34 if i == 0 else 23)
        color = (255, 120, 100) if line.startswith("FAILURE") else ((240, 240, 240) if i == 0 else (190, 200, 215))
        cur = ""
        for word in line.split(" "):
            if cur and d.textlength(cur + " " + word, font=f) > W - 100:
                rows.append((cur, i == 0, color))
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        rows.append((cur, i == 0, color))
    height = sum(58 if head else 36 for _, head, _ in rows)
    y = max(20, (H - height) / 2)
    for text, head, color in rows:
        f = font(34 if head else 23)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, fill=color, font=f)
        y += 58 if head else 36
    if seconds is None:
        seconds = max(3.5, sum(len(t.split()) for t in lines) / 4.5)
    return [np.asarray(img)] * int(seconds * FPS)


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
        frames += card(lines)
        if path:
            if not Path(path).exists():
                raise SystemExit(f"missing clip {path}")
            frames += clip(path, speed)
    imageio.mimsave(args.out, frames, fps=FPS, macro_block_size=16)
    print(f"wrote {args.out}: {len(frames)} frames, {len(frames) / FPS / 60:.1f} min")


if __name__ == "__main__":
    main()
