"""Convert data/umi_sim.npz to UMI's replay-buffer format (.zarr.zip). Schema: drywall_env/umi_zarr.py.

    python scripts/convert_to_umi_zarr.py --npz data/umi_sim.npz --out data/umi_sim.zarr.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env.umi_zarr import convert_npz_to_umi_zarr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/umi_sim.npz")
    ap.add_argument("--out", default="data/umi_sim.zarr.zip")
    ap.add_argument("--no-mirror-swap", action="store_true",
                    help="keep the raw (physically flipped) mirror regions in camera0_rgb")
    args = ap.parse_args()
    t = time.time()

    def progress(i, n):
        if i % 25 == 0 or i == n - 1:
            print(f"episode {i + 1}/{n}  {time.time() - t:.0f} s", flush=True)

    r = convert_npz_to_umi_zarr(args.npz, args.out, progress=progress, do_mirror_swap=not args.no_mirror_swap)
    size = Path(args.out).stat().st_size / 1e9
    print(f"wrote {args.out}: {r['episodes']} episodes, {r['rows']} rows, {size:.2f} GB, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
