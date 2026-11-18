"""Record N scripted demonstrations as a UMI-style dataset (.npz). Schema: drywall_env/umi_recorder.py.

    python scripts/record_umi.py --n 500 --out data/umi_sim.npz

The scripted controller stands in for the human demonstrator. Demo quality varies per episode:
action-noise std ~ U[0, --max-noise] (logged in the sidecar), which gives both successes and
failures. Sim-only per-episode metadata (seed, noise std, randomization, how it ended) goes to
<out>_sim_meta.npz, NOT into the UMI file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, ScriptedController  # noqa: E402
from drywall_env.umi_recorder import (  # noqa: E402
    FISHEYE_FOV_DEG, IMAGE_HZ, POSE_HZ, SCHEMA_VERSION, NpzStreamWriter, UMIRecorder,
)


def record_episode(env, rec, seed, noise_std, t0):
    obs, info = env.reset(seed=seed)
    ctrl = ScriptedController(env.cfg, action_noise_std=noise_std, seed=seed)  # fresh per episode
    rec.start(t0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        rec.on_control_step()
    ep = rec.finish(success=bool(info["success"]))
    meta = {"seed": seed, "action_noise_std": noise_std, "dropped": bool(info["dropped"]),
            "truncated": bool(truncated), "n_control_steps": int(info["steps"]), **info["randomization"]}
    return ep, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=str, default="data/umi_sim.npz")
    ap.add_argument("--start-seed", type=int, default=0, help="episode i uses env seed start_seed + i")
    ap.add_argument("--max-noise", type=float, default=0.2, help="per-episode action-noise std ~ U[0, max]")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--check-after", type=int, default=5, help="project final size after this many episodes")
    ap.add_argument("--append-calibration", action="store_true",
                    help="only add meta/ camera calibration to an existing --out file (recorded before it existed)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.append_calibration:
        env = DrywallPlacementEnv()
        env.reset(seed=0)
        rec = UMIRecorder(env, image_size=args.image_size)
        w = NpzStreamWriter(out, mode="a")
        w.put_group("meta", rec.calibration())
        w.close()
        rec.close()
        env.close()
        print(f"added camera calibration to {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_name(out.stem + "_sim_meta.npz")
    noise_rng = np.random.default_rng(args.start_seed)

    env = DrywallPlacementEnv()
    rec = UMIRecorder(env, image_size=args.image_size)
    writer = NpzStreamWriter(out)
    writer.put("meta/schema_version", np.str_(SCHEMA_VERSION))
    writer.put("meta/n_episodes", np.int64(args.n))
    writer.put("meta/pose_hz", np.float64(POSE_HZ))
    writer.put("meta/image_hz", np.float64(IMAGE_HZ))
    writer.put("meta/fisheye_fov_deg", np.float64(FISHEYE_FOV_DEG))
    writer.put_group("meta", rec.calibration())

    metas, successes, t0, start = [], [], 0.0, time.time()
    try:
        for i in range(args.n):
            seed = args.start_seed + i
            noise = float(noise_rng.uniform(0.0, args.max_noise))
            ep, meta = record_episode(env, rec, seed, noise, t0)
            writer.put_group(f"episode_{i:03d}", ep)
            metas.append(meta)
            successes.append(bool(ep["success"]))
            t0 = float(ep["end_time"]) + 1.0 / POSE_HZ  # next episode starts on the next frame
            print(f"ep {i:3d} seed {seed:3d} noise {noise:.3f} success={bool(ep['success'])!s:5} "
                  f"poses {len(ep['timestamp']):4d} images {len(ep['wrist_rgb']):3d} "
                  f"| {writer.bytes_written() / 1e9:.2f} GB, {time.time() - start:.0f} s", flush=True)
            if i + 1 == args.check_after and args.n > args.check_after:
                projected = writer.bytes_written() / (i + 1) * args.n
                free = shutil.disk_usage(out.parent).free
                print(f"projected size {projected / 1e9:.1f} GB, free {free / 1e9:.1f} GB", flush=True)
                if projected > free - 1e9:
                    raise SystemExit("not enough disk space for the full run; aborting")
    finally:
        writer.close()
        rec.close()
        env.close()

    keys = metas[0].keys() if metas else []
    np.savez(meta_path, **{k: np.array([m[k] for m in metas]) for k in keys})
    print(f"\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB) and {meta_path}")
    print(f"{len(metas)} episodes, success rate {np.mean(successes) * 100:.1f} %, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
