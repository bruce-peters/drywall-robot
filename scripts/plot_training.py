"""Training charts (matplotlib) for BC and IQL runs, and the final comparison chart.

    python scripts/plot_training.py runs/bc_umi                 # -> media/training_bc_umi.png
    python scripts/plot_training.py runs/iql_sim runs/iql_umi_sim
    python scripts/plot_training.py --compare results/*.json    # -> media/comparison.png

Each run directory holds log.jsonl written by train_bc.py / train_iql.py. The run type is read
from the log. Validation-rollout points are measured on the validation seeds (5000-5049 easy, 25000-25049 hard; never trained on, never
used to pick a checkpoint); the comparison chart uses the held-out test seeds 10000-10099.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
C = {"blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a", "red": "#c53030", "grey": "#718096",
     "purple": "#6b46c1"}


def val_seed_text(L) -> str:
    v = L.get("val_rollout", [])
    lo, hi = (v[0].get("seeds") if v and v[0].get("seeds") else [5000, 5049])
    return f"seeds {lo}-{hi}"


def kfmt(fig):
    """Gradient-step axes as 50k / 100k instead of 50000 / 100000."""
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda x, _: f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}")
    for ax in fig.axes:
        if ax.get_xlabel() == "gradient step":
            ax.xaxis.set_major_formatter(f)


def read_log(run_dir):
    recs = [json.loads(line) for line in open(Path(run_dir) / "log.jsonl", encoding="utf-8")]
    by = {}
    for r in recs:
        by.setdefault(r["kind"], []).append(r)
    return by


def series(recs, key):
    recs = [r for r in recs if key in r and r[key] is not None]
    return np.array([r["step"] for r in recs]), np.array([r[key] for r in recs], float)


def plot_val_rollouts(ax, L):
    v = L.get("val_rollout", [])
    if not v:
        ax.text(0.5, 0.5, "no validation rollouts", ha="center", transform=ax.transAxes)
        return
    n = int(v[0]["n"])
    lo, hi = v[0].get("seeds", [5000, 5000 + n - 1])
    title = f"Validation rollouts ({n} episodes, seeds {lo}-{hi})"
    st = np.array([r["step"] for r in v])
    sr = 100 * np.array([r["success_rate"] for r in v])
    lo = 100 * np.array([r["success_ci95"][0] for r in v])
    hi = 100 * np.array([r["success_ci95"][1] for r in v])
    ax.fill_between(st, lo, hi, color=C["green"], alpha=0.15, label="95 % Wilson interval")
    ax.plot(st, sr, "o-", color=C["green"], label="success rate")
    ax.plot(st, 100 * np.array([r["collision_rate"] for r in v]), "s--", color=C["red"], ms=4,
            label="collision rate")
    ax.set_ylim(-3, 103)
    ax.set_ylabel("% of episodes")
    ax.set_xlabel("gradient step")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(st, [r["pos_error_mm_median"] for r in v], "^:", color=C["grey"], ms=4, label="median final pos error")
    ax2.set_ylabel("median final position error (mm)", color=C["grey"])
    ax2.set_yscale("log")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7)


def plot_bc(L, name, out):
    cfg = L["config"][0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for key, col, lab in (("loss", C["blue"], "total"), ("mse", C["orange"], "motion MSE"), ("bce", C["purple"], "attach BCE")):
        x, y = series(L["train"], key)
        ax.plot(x, y, color=col, alpha=0.85, label=f"train {lab}")
        x, y = series(L.get("val_loss", []), key)
        ax.plot(x, y, "o--", color=col, ms=3, label=f"held-out demos {lab}")
    ax.set_yscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Behaviour-cloning loss")
    ax.legend(fontsize=7)
    ax = axes[1]
    x, y = series(L["train"], "lr")
    ax.plot(x, y, color=C["grey"])
    ax.set_title("Learning rate (cosine decay)")
    ax.set_xlabel("gradient step")
    plot_val_rollouts(axes[2], L)
    fig.suptitle(f"{name}: behaviour cloning on {int(cfg['n_train_episodes'])} demonstration episodes "
                 f"({int(cfg['n_train_transitions'])} transitions), action chunk H={int(cfg['chunk'])}", fontsize=11)
    fig.text(0.01, 0.005, "Train loss = running mean over 200 steps. Held-out loss = demo episodes withheld from "
             "training. Rollouts = closed-loop runs of the current policy in the env (measurement only; the "
             "final checkpoint is always the last step).", fontsize=7, color=C["grey"])
    kfmt(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def plot_iql(L, name, out):
    cfg = L["config"][0]
    tr = L["train"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    for key, col, lab in (("q_loss", C["blue"], "Q loss, eq. (6): TD error²"),
                          ("v_loss", C["orange"], "V loss, eq. (5): expectile")):
        ax.plot(*series(tr, key), color=col, label=lab)
    ax.set_yscale("log")
    ax.set_title("Critic losses")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(*series(tr, "q_mean"), color=C["blue"], label="mean target Q(s,a) on batch")
    ax.plot(*series(tr, "v_mean"), color=C["orange"], label="mean V(s) on batch")
    ax.set_title(f"Value estimates (rewards × {cfg['reward_scale']})")
    ax.legend(fontsize=7)
    ax = axes[0, 2]
    x, m = series(tr, "adv_mean")
    _, s = series(tr, "adv_std")
    ax.plot(x, m, color=C["purple"], label="mean A = Q - V")
    ax.fill_between(x, m - s, m + s, color=C["purple"], alpha=0.15, label="± 1 std")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Advantage (expectile τ={cfg['tau']} pushes mean A below 0)")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    ax.plot(*series(tr, "actor_nll"), color=C["grey"], label="unweighted −log π (BC loss)")
    ax.plot(*series(tr, "actor_loss"), color=C["green"], label="weighted actor loss, eq. (7)")
    ax.set_yscale("log")
    ax.set_title("Actor (advantage-weighted regression)")
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(*series(tr, "weight_mean"), ":", color=C["orange"], label=f"mean exp(β·A), β={cfg['beta']}")
    ax2.set_ylabel("mean weight", color=C["orange"])
    ax2.grid(False)
    ax = axes[1, 1]
    hist = L.get("adv_hist", [])
    if hist:
        pick = [hist[i] for i in sorted({0, len(hist) // 2, len(hist) - 1})]
        allv = np.concatenate([h["values"] for h in pick])
        bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 60)
        for h, col in zip(pick, (C["grey"], C["blue"], C["green"])):
            ax.hist(h["values"], bins=bins, histtype="step", lw=1.4, color=col, label=f"step {h['step']}")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_title("Advantage distribution on 4096 dataset (s, a)")
    ax.set_xlabel("A(s, a)")
    plot_val_rollouts(axes[1, 2], L)
    for a in axes.flat:
        if not a.get_xlabel():
            a.set_xlabel("gradient step")
    data = ", ".join(Path(p).name for p in cfg["data"])
    fig.suptitle(f"{name}: IQL on {int(cfg['n_transitions'])} transitions ({data}); reward terms: "
                 f"{', '.join(cfg['reward_terms_included'])}", fontsize=11)
    fig.text(0.01, 0.005, f"Training uses only the fixed dataset. Validation rollouts ({val_seed_text(L)}) are measurement "
             "only: not added to the data, not used for gradients or checkpoint choice. Equations: Kostrikov et al. "
             "2021, arXiv:2110.06169.", fontsize=7, color=C["grey"])
    kfmt(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def plot_compare(paths, out):
    res = [json.load(open(p)) for p in paths]
    names = [Path(p).stem.removesuffix("_in").removesuffix("_ood") for p in paths]
    S = [r["summary"] for r in res]
    fig, axes = plt.subplots(1, 5, figsize=(20, 6.2))
    x = np.arange(len(names))
    sr = np.array([100 * s["success_rate"] for s in S])
    lo = np.clip(sr - np.array([100 * s["success_ci95"][0] for s in S]), 0, None)
    hi = np.clip(np.array([100 * s["success_ci95"][1] for s in S]) - sr, 0, None)  # float noise at 100 %
    axes[0].bar(x, sr, yerr=[lo, hi], capsize=4, color=C["green"])
    axes[0].set_title("Success rate (95 % Wilson CI)")
    axes[0].set_ylim(0, 105)
    for i, r in enumerate(res):
        e = r["episodes"]
        axes[1].scatter(np.full(len(e), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(e)),
                        [max(ep["pos_error_mm"], 0.1) for ep in e], s=6, alpha=0.5,
                        c=[C["green"] if ep["success"] else C["red"] for ep in e])
    axes[1].axhline(20, color="k", lw=0.8, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_title("Final position error per episode (mm)\n(green = success, dashed = 20 mm tolerance)")
    axes[2].bar(x, [100 * s["collision_rate"] for s in S], color=C["red"])
    axes[2].set_title("Collision rate (% episodes)")
    axes[3].bar(x, [s["length_mean"] for s in S], color=C["blue"])
    axes[3].set_title("Mean episode length (steps; time-out 300 easy / 500 hard)")
    kinds = [("never_grasped", C["grey"]), ("held_at_end", C["purple"]), ("released_off_wall", C["orange"]),
             ("released_misaligned", "#d69e2e"), ("dropped", C["red"]), ("other", "#a0aec0")]
    bottom = np.zeros(len(S))
    for k, col in kinds:
        v = np.array([s.get("failures", {}).get(k, 0) for s in S], float)
        if v.any():
            axes[4].bar(x, v, bottom=bottom, color=col, label=k.replace("_", " "))
            bottom += v
    axes[4].set_title("Failure modes (episodes)")
    axes[4].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ranges = sorted({f"{r['seeds'][0]}-{r['seeds'][1]}" for r in res})
    which = {"in": "hard challenge, IN the demos' region", "ood": "hard challenge, OUTSIDE the demos' region",
             "test": "easy challenge"}.get(res[0].get("test", "test"), "")
    fig.suptitle(f"Policy comparison on held-out test seeds {', '.join(ranges)} ({which}; {S[0]['n']} episodes each)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--compare", nargs="*", default=None)
    ap.add_argument("--out-dir", default="media")
    ap.add_argument("--compare-name", default="comparison", help="file name of the comparison chart")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)
    for run in args.runs:
        L = read_log(run)
        name = Path(run).name
        out = Path(args.out_dir) / f"training_{name}.png"
        (plot_iql if "q_loss" in L["train"][0] else plot_bc)(L, name, out)
        print("wrote", out)
    if args.compare:
        out = Path(args.out_dir) / f"{args.compare_name}.png"
        plot_compare(args.compare, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
