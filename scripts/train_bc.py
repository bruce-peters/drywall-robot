"""Part B.C: behaviour cloning on the converted UMI demonstrations.

    python scripts/train_bc.py                       # -> checkpoints/bc_umi.pt, runs/bc_umi/log.jsonl

Objective (BC as maximum likelihood): min_theta E_{(s,a)~D} [ ||mu_theta(s) - a_motion||^2
    + BCE(sigmoid(l_theta(s)), 1[a_attach > 0]) ], i.e. a fixed-variance Gaussian on the 6 motion
dims and a Bernoulli on the suction bit. With --success-only (default) D is the successful
demonstrations only ("filtered BC"): failed demos would teach the policy to fail.

The same network is IQL's actor (train_iql.py), so the comparison isolates the learning signal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env.policies import ActorNet, action_chunks, save_actor  # noqa: E402
from drywall_env.training_utils import (  # noqa: E402
    JsonlLogger, default_log_path, make_env, set_seed, snapshot_path, validation_rollouts,
)
from drywall_env.transitions import UMI_REWARD_TERMS, episode_slices, load_many  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/umi_transitions.npz"])
    ap.add_argument("--out", default="checkpoints/bc_umi.pt")
    ap.add_argument("--log", default=None, help="default runs/<checkpoint subfolder>/<out stem>/log.jsonl")
    ap.add_argument("--all-episodes", action="store_true", help="also train on failed demos")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--chunk", type=int, default=1, help="action chunk length H")
    ap.add_argument("--n-exec", type=int, default=None, help="actions executed per chunk (default H)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out demo episodes for the val loss")
    ap.add_argument("--val-loss-every", type=int, default=1000)
    ap.add_argument("--rollout-every", type=int, default=5000, help="validation rollouts (0 = off)")
    ap.add_argument("--rollout-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy",
                    help="env used for the validation rollouts (hard: seeds 25000-25049, full region)")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = args.log or default_log_path(args.out)
    D = load_many(args.data, reward_terms=UMI_REWARD_TERMS)  # refuses test seeds; rewards unused by BC

    # episodes -> train / val split (by episode, so the val loss measures generalisation)
    eps = episode_slices(D["episode_ends"])
    keep = [i for i in range(len(eps)) if args.all_episodes or D["episode_success"][i]]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(keep)
    n_val = max(1, int(round(args.val_frac * len(keep))))
    val_eps, train_eps = sorted(keep[:n_val]), sorted(keep[n_val:])
    chunks = action_chunks(D["actions"], D["episode_ends"], args.chunk)
    idx = lambda ids: np.concatenate([np.arange(eps[i].start, eps[i].stop) for i in ids])  # noqa: E731
    tr, va = idx(train_eps), idx(val_eps)
    S = torch.as_tensor(D["observations"], device=dev)
    A = torch.as_tensor(chunks, device=dev)
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)

    actor = ActorNet(state_dim=D["observations"].shape[1], hidden=args.hidden, chunk=args.chunk).to(dev)
    actor.norm.fit(D["observations"][tr])
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    log = JsonlLogger(log_path)
    log.log("config", 0, **vars(args), n_train_episodes=len(train_eps), n_val_episodes=len(val_eps),
            n_train_transitions=len(tr), device=dev)
    print(f"BC on {len(train_eps)} train / {len(val_eps)} val episodes ({len(tr)} transitions), "
          f"device {dev}", flush=True)

    env = make_env(args.challenge) if args.rollout_every else None
    t0, acc = time.time(), []
    for step in range(1, args.steps + 1):
        b = tr_t[torch.randint(len(tr_t), (args.batch,), device=dev)]
        loss, mse, bce = actor.loss_per_sample(S[b], A[b])
        loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc.append([loss.item(), mse.mean().item(), bce.mean().item()])
        if step % 200 == 0:
            m = np.mean(acc, 0)
            log.log("train", step, loss=m[0], mse=m[1], bce=m[2], lr=sched.get_last_lr()[0])
            acc = []
        if step % args.val_loss_every == 0 or step == args.steps:
            actor.eval()
            with torch.no_grad():
                vl, vm, vb = (x.mean().item() for x in actor.loss_per_sample(S[va_t], A[va_t]))
            actor.train()
            log.log("val_loss", step, loss=vl, mse=vm, bce=vb)
            print(f"step {step:6d}  val loss {vl:.4f} (mse {vm:.4f}, bce {vb:.4f})  "
                  f"{time.time() - t0:.0f} s", flush=True)
        if args.rollout_every and (step % args.rollout_every == 0 or step == args.steps):
            actor.eval()
            s = validation_rollouts(actor, env, args.rollout_episodes, n_exec=args.n_exec)
            save_actor(snapshot_path(args.out, step), actor, extra={"algo": "bc", "snapshot_step": step,
                                                                    "n_exec": args.n_exec or args.chunk})
            actor.train()
            log.log("val_rollout", step, **s)
            print(f"step {step:6d}  VAL ROLLOUTS success {100 * s['success_rate']:.0f} %  "
                  f"pos {s['pos_error_mm_mean']:.1f} mm  collisions {100 * s['collision_rate']:.0f} %", flush=True)

    actor.eval()
    save_actor(args.out, actor, extra={"algo": "bc", "data": args.data, "n_exec": args.n_exec or args.chunk,
                                       "args": vars(args), "train_episodes": len(train_eps)})
    log.log("done", args.steps, seconds=time.time() - t0, checkpoint=args.out)
    log.close()
    print(f"saved {args.out}  ({time.time() - t0:.0f} s)")
    if env:
        env.close()


if __name__ == "__main__":
    main()
