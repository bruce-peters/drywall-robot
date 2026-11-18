"""Part C.B: Implicit Q-Learning (IQL), written from scratch for this project.

    python scripts/train_iql.py --data data/offline_sim.npz --out checkpoints/iql_sim.pt
    python scripts/train_iql.py --data data/umi_transitions.npz data/offline_sim.npz \
        --reward-terms umi --out checkpoints/iql_umi_sim.pt

Reference: I. Kostrikov, A. Nair, S. Levine, "Offline Reinforcement Learning with Implicit
Q-Learning", ICLR 2022, arXiv:2110.06169. Equation numbers below refer to that paper. The CORL
single-file implementation (Tarasov et al., NeurIPS 2023 Datasets & Benchmarks,
github.com/tinkoff-ai/CORL) was used only as a cross-check of details (twin critics, target
network, advantage clipping); no code was copied.

Why IQL is OFFLINE RL
---------------------
Standard Q-learning bootstraps from max_a' Q(s', a'). On a fixed dataset that max is taken over
actions the data never tried, whose Q-values are pure extrapolation and are systematically
over-estimated (distribution shift, Kumar et al. 2020 / CQL). IQL never evaluates Q at an action
outside the dataset. It learns a state value V(s) as an *upper expectile* of Q(s, a) over the
dataset's own actions, which approximates "the value of the best action the data supports" without
ever querying an unseen action. Training reads only the fixed dataset; no env.step is called for
learning (the optional validation rollouts only measure, see training_utils.py).

The three losses (one gradient step each per iteration)
------------------------------------------------------
Value, eq. (5):    L_V(psi)   = E_{(s,a)~D} [ L2^tau( Q_target(s,a) - V_psi(s) ) ]
                   with the expectile loss  L2^tau(u) = | tau - 1(u < 0) | * u^2.
                   tau = 0.5 gives the mean (SARSA-like value of the data policy); tau -> 1
                   approaches the max over in-distribution actions. Q_target = min of two target
                   critics (clipped double-Q).
Q-function, eq. (6): L_Q(theta) = E_{(s,a,r,s')~D} [ ( r + gamma * (1 - done) * V_psi(s')
                   - Q_theta(s,a) )^2 ]  -- the bootstrap uses V(s'), never max_a' Q(s', a').
                   `done` is a true terminal (success / drop); time-outs DO bootstrap.
Policy, eq. (7) (advantage-weighted regression, Peng et al. 2019):
                   L_pi(phi) = - E_{(s,a)~D} [ exp( beta * (Q_target(s,a) - V_psi(s)) ) * log pi_phi(a|s) ]
                   The weight exp(beta * A) is clipped at 100 (as in the paper's code). Our actor
                   is the BC network (policies.ActorNet): -log pi = MSE(motion) + BCE(attach) up to
                   constants (fixed-variance Gaussian x Bernoulli), so IQL's actor loss is exactly
                   BC's loss with a per-sample weight. beta -> 0 recovers plain BC on all data.
Target critics: theta_target <- (1 - alpha) * theta_target + alpha * theta  (Polyak, alpha = 0.005).

Defaults follow the paper's MuJoCo locomotion settings: tau = 0.7, beta = 3, gamma = 0.99,
lr = 3e-4, batch 256, cosine-decayed actor learning rate. Rewards are multiplied by
--reward-scale (default 0.1) so that Q stays O(10) here (step costs ~ -0.5, success +100); the
paper likewise rescales rewards per domain.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env.policies import ActorNet, Normalizer, action_chunks, mlp, save_actor  # noqa: E402
from drywall_env.training_utils import (  # noqa: E402
    JsonlLogger, default_log_path, make_env, set_seed, snapshot_path, validation_rollouts,
)
from drywall_env.transitions import (  # noqa: E402
    REWARD_TERMS, STATE_DIM, UMI_REWARD_TERMS, load_many,
)

EXP_ADV_MAX = 100.0


def expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """L2^tau(u) = |tau - 1(u < 0)| u^2  (IQL eq. 5). Under-estimates (u > 0: Q above V) are
    weighted by tau, over-estimates by 1 - tau, so for tau > 0.5 V settles above the mean of Q."""
    weight = torch.abs(tau - (u < 0).float())
    return weight * u.pow(2)


class TwinQ(nn.Module):
    """Two independent Q(s, a) networks (clipped double-Q, Fujimoto et al. 2018)."""

    def __init__(self, norm: Normalizer, hidden, state_dim: int = STATE_DIM):
        super().__init__()
        self.norm = norm
        self.q1 = mlp(state_dim + 7, 1, hidden)
        self.q2 = mlp(state_dim + 7, 1, hidden)

    def both(self, s, a):
        x = torch.cat([self.norm(s), a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, s, a):
        return torch.min(*self.both(s, a))


class ValueNet(nn.Module):
    def __init__(self, norm: Normalizer, hidden, state_dim: int = STATE_DIM):
        super().__init__()
        self.norm = norm
        self.v = mlp(state_dim, 1, hidden)

    def forward(self, s):
        return self.v(self.norm(s)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None, help="default runs/<checkpoint subfolder>/<out stem>/log.jsonl")
    ap.add_argument("--reward-terms", choices=["stored", "umi", "all"], default="stored",
                    help="stored = as saved in the file; umi = only the terms UMI data supports")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--tau", type=float, default=0.7, help="expectile (eq. 5)")
    ap.add_argument("--beta", type=float, default=3.0, help="inverse temperature (eq. 7)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--polyak", type=float, default=0.005)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=1, help="actor action chunk (weighted by A(s_t, a_t))")
    ap.add_argument("--n-exec", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--rollout-every", type=int, default=20_000, help="validation rollouts (0 = off)")
    ap.add_argument("--rollout-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy",
                    help="env used for the validation rollouts (hard: seeds 25000-25049, full region)")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    terms = {"stored": None, "umi": UMI_REWARD_TERMS, "all": REWARD_TERMS}[args.reward_terms]
    D = load_many(args.data, terms)  # refuses test seeds; one reward definition
    N = len(D["actions"])
    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
    S, S2 = T(D["observations"]), T(D["next_observations"])
    A = T(D["actions"])
    A_chunk = T(action_chunks(D["actions"], D["episode_ends"], args.chunk))
    R = T(D["rewards"]) * args.reward_scale
    DONE = T(D["terminals"].astype(np.float32))

    # Networks. One state normalizer (fit on the dataset) shared by actor, critics and value.
    sdim = D["observations"].shape[1]  # 37 (easy, state v1) or 46 (hard, state v2)
    norm = Normalizer(sdim).fit(D["observations"]).to(dev)
    actor = ActorNet(state_dim=sdim, hidden=args.hidden, chunk=args.chunk).to(dev)
    actor.norm.load_state_dict(norm.state_dict())
    qf = TwinQ(norm, args.hidden, sdim).to(dev)
    q_target = copy.deepcopy(qf).requires_grad_(False)
    vf = ValueNet(norm, args.hidden, sdim).to(dev)
    # the normalizer is fixed (fit once), so it is excluded from the optimizers
    params = lambda m: [p for n, p in m.named_parameters() if not n.startswith("norm.")]  # noqa: E731
    q_opt = torch.optim.Adam(params(qf), lr=args.lr)
    v_opt = torch.optim.Adam(params(vf), lr=args.lr)
    a_opt = torch.optim.Adam(params(actor), lr=args.lr)
    a_sched = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, args.steps)

    log = JsonlLogger(args.log or default_log_path(args.out))
    log.log("config", 0, **vars(args), n_transitions=N, n_episodes=len(D["episode_ends"]),
            reward_terms_included=D["reward_terms_included"] if terms is None else list(terms),
            terminal_frac=float(D["terminals"].mean()), device=dev)
    print(f"IQL on {N} transitions / {len(D['episode_ends'])} episodes from {args.data}; "
          f"reward terms {terms or D['reward_terms_included']}; device {dev}", flush=True)

    env = make_env(args.challenge) if args.rollout_every else None
    acc, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        b = torch.randint(N, (args.batch,), device=dev)
        s, a, r, s2, done = S[b], A[b], R[b], S2[b], DONE[b]

        # --- value: eq. (5), expectile regression of V(s) towards min-target-Q(s, a)
        with torch.no_grad():
            q_t = q_target(s, a)
        v = vf(s)
        adv = q_t - v
        v_loss = expectile_loss(adv, args.tau).mean()
        v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        v_opt.step()

        # --- Q: eq. (6), TD target r + gamma (1 - done) V(s'), using the just-updated V
        with torch.no_grad():
            target = r + args.gamma * (1.0 - done) * vf(s2)
        q1, q2 = qf.both(s, a)
        q_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()

        # --- policy: eq. (7), advantage-weighted regression with the BC loss as -log pi
        w = torch.exp(args.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        nll, _, _ = actor.loss_per_sample(s, A_chunk[b])
        a_loss = (w * nll).mean()
        a_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        a_opt.step()
        a_sched.step()

        # --- target critics: Polyak averaging
        with torch.no_grad():
            for p, pt in zip(qf.parameters(), q_target.parameters()):
                pt.lerp_(p, args.polyak)

        acc.append(torch.stack([v_loss, q_loss / 2, a_loss, nll.mean(), q_t.mean(), v.mean().detach(),
                                adv.mean().detach(), adv.std().detach(), w.mean(),
                                (w >= EXP_ADV_MAX).float().mean()]).detach())
        if step % args.log_every == 0:
            m = torch.stack(acc).mean(0).cpu().numpy()
            acc = []
            keys = ["v_loss", "q_loss", "actor_loss", "actor_nll", "q_mean", "v_mean",
                    "adv_mean", "adv_std", "weight_mean", "weight_clipped_frac"]
            log.log("train", step, **dict(zip(keys, m)), lr_actor=a_sched.get_last_lr()[0])
            if step % (args.log_every * 20) == 0:
                print(f"step {step:7d}  V {m[0]:.4f}  Q {m[1]:.4f}  actor {m[2]:.3f} (nll {m[3]:.3f})  "
                      f"Q {m[4]:.2f}  V {m[5]:.2f}  A {m[6]:+.3f}±{m[7]:.3f}  w {m[8]:.2f}  "
                      f"{time.time() - t0:.0f} s", flush=True)
        if step % 10_000 == 0:  # advantage histogram snapshot for the charts
            with torch.no_grad():
                bb = torch.randint(N, (4096,), device=dev)
                advs = (q_target(S[bb], A[bb]) - vf(S[bb])).cpu().numpy()
            log.log("adv_hist", step, values=np.round(advs, 4).tolist())
        if args.rollout_every and (step % args.rollout_every == 0 or step == args.steps):
            actor.eval()
            sm = validation_rollouts(actor, env, args.rollout_episodes, n_exec=args.n_exec)
            save_actor(snapshot_path(args.out, step), actor, extra={"algo": "iql", "snapshot_step": step,
                                                                    "n_exec": args.n_exec or args.chunk})
            actor.train()
            log.log("val_rollout", step, **sm)
            print(f"step {step:7d}  VAL ROLLOUTS success {100 * sm['success_rate']:.0f} %  "
                  f"pos {sm['pos_error_mm_mean']:.1f} mm  collisions {100 * sm['collision_rate']:.0f} %", flush=True)

    actor.eval()
    save_actor(args.out, actor, extra={"algo": "iql", "data": args.data, "n_exec": args.n_exec or args.chunk,
                                       "args": vars(args)})
    torch.save({"q": qf.state_dict(), "v": vf.state_dict()}, Path(args.out).with_suffix(".critics.pt"))
    log.log("done", args.steps, seconds=time.time() - t0, checkpoint=args.out)
    log.close()
    print(f"saved {args.out}  ({time.time() - t0:.0f} s)")
    if env:
        env.close()


if __name__ == "__main__":
    main()
