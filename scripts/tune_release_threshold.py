"""Tune a learned policy's suction-release threshold on VALIDATION seeds (never test seeds).

    python scripts/tune_release_threshold.py --challenge easy checkpoints/easy_challenge/iql_umi_sim_b3.pt

While suction is on, the policy keeps it on only if p(attach) > threshold (grasping always uses 0.5).
The release is ~1 in 150-220 suction-on states in the data, so the plain 0.5 classifier is biased
towards "keep holding" (lower threshold = hold longer, higher = release sooner). Each threshold is
run on the same validation seeds (easy 5000-5049, hard 25000-25049 on the full region); the best
success rate wins, ties go to the threshold closest to 0.5. The chosen value is then evaluated ONCE
on the test seeds with `eval_policy.py --release-threshold`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env.evaluation import evaluate  # noqa: E402
from drywall_env.policies import load_policy  # noqa: E402
from drywall_env.training_utils import make_env, val_seeds_for  # noqa: E402

THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--n", type=int, default=50, help="validation episodes per threshold")
    ap.add_argument("--results-dir", default=None, help="default results/<challenge>_challenge")
    args = ap.parse_args()

    env = make_env(args.challenge)
    seeds = list(val_seeds_for(env))[: args.n]
    out_dir = Path(args.results_dir or f"results/{args.challenge}_challenge")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ck in args.checkpoints:
        rows = []
        for thr in args.thresholds:
            _, s = evaluate(env, load_policy(ck, release_threshold=thr), seeds)
            rows.append({"threshold": thr, "success_rate": s["success_rate"], "failures": s["failures"]})
            print(f"{Path(ck).stem:28s} thr {thr:.2f}: val success {100 * s['success_rate']:5.1f} %  "
                  + " ".join(f"{k} {v}" for k, v in s["failures"].items() if v), flush=True)
        best = max(rows, key=lambda r: (r["success_rate"], -abs(r["threshold"] - 0.5)))
        print(f"  -> best threshold {best['threshold']} ({100 * best['success_rate']:.0f} % on validation)")
        with open(out_dir / f"tune_rt_{Path(ck).stem}.json", "w") as f:
            json.dump({"checkpoint": ck, "seeds": [seeds[0], seeds[-1]], "rows": rows,
                       "best_threshold": best["threshold"]}, f, indent=1)
    env.close()


if __name__ == "__main__":
    main()
