"""Part C.A: generate a FIXED offline dataset of simulated trajectories (seeds 1000-1999).

    python scripts/generate_offline_data.py --bc checkpoints/bc_umi.pt     # -> data/offline_sim.npz

Behaviour policies (mixed per episode, chosen from the episode seed):
  * scripted controller with action noise std ~ U[0, --max-noise]
  * the Part B behaviour-cloning policy with Gaussian action noise std ~ U[0, --max-noise]
    (omit --bc for scripted-only data)
Randomized factors (doc lists >= 2): panel position and yaw on the rack, target offset on the wall,
panel mass, panel friction (all from the env seed) and robot action noise (per episode).

Stored as transitions (state, action, reward, next state, terminal, timeout) in the shared format
(drywall_env/transitions.py) with the env's FULL reward, every term stored separately. The action
stored is the one actually executed (after noise and clipping to [-1, 1]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WORKER = {}


def _init_worker(bc_path):
    import torch

    torch.set_num_threads(1)
    from drywall_env import DrywallPlacementEnv
    from drywall_env.policies import load_policy

    _WORKER["env"] = DrywallPlacementEnv()
    _WORKER["bc"] = load_policy(bc_path) if bc_path else None


def _episode(job):
    from drywall_env.policies import ScriptedPolicy
    from drywall_env.transitions import REWARD_TERMS, state_from_obs

    seed, behaviour, noise = job
    env = _WORKER["env"]
    rng = np.random.default_rng(seed + 7_000_000)
    pol = ScriptedPolicy(env.cfg) if behaviour == "scripted" else _WORKER["bc"]
    obs, info = env.reset(seed=seed)
    pol.reset()
    states, actions = [state_from_obs(obs)], []
    terms = {k: [] for k in REWARD_TERMS}
    terminated = truncated = False
    while not (terminated or truncated):
        a = np.asarray(pol.act(obs), np.float64).copy()
        if noise > 0:
            a[:6] += rng.normal(0.0, noise, 6)
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(a)
        states.append(state_from_obs(obs))
        actions.append(a)
        for k in REWARD_TERMS:
            terms[k].append(info["reward_terms"][k])
    return {"observations": np.array(states), "actions": np.array(actions),
            "reward_terms": {k: np.array(v, np.float32) for k, v in terms.items()},
            "terminal": bool(terminated), "success": bool(info["success"]), "seed": int(seed),
            "info": {"behaviour": behaviour, "noise_std": float(noise), "dropped": bool(info["dropped"]),
                     "collision_steps": int(sum(1 for c in terms["collision"] if c < 0))}}


def main():
    from drywall_env.transitions import OFFLINE_SEEDS, REWARD_TERMS, check_not_test_seeds, save_transitions

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/offline_sim.npz")
    ap.add_argument("--bc", default=None, help="BC checkpoint used as the second behaviour policy")
    ap.add_argument("--n", type=int, default=len(OFFLINE_SEEDS))
    ap.add_argument("--start-seed", type=int, default=OFFLINE_SEEDS[0])
    ap.add_argument("--bc-frac", type=float, default=0.5, help="fraction of episodes from the BC policy")
    ap.add_argument("--max-noise", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n))
    check_not_test_seeds(seeds)
    jobs = []
    for s in seeds:
        r = np.random.default_rng(s)
        beh = "bc" if (args.bc and r.uniform() < args.bc_frac) else "scripted"
        jobs.append((s, beh, float(r.uniform(0.0, args.max_noise))))

    t0 = time.time()
    episodes = []
    with Pool(args.workers, initializer=_init_worker, initargs=(args.bc,)) as pool:
        for i, ep in enumerate(pool.imap(_episode, jobs, chunksize=4)):
            episodes.append(ep)
            if (i + 1) % 100 == 0:
                sr = np.mean([e["success"] for e in episodes])
                print(f"  {i + 1}/{len(jobs)} episodes, success so far {100 * sr:.1f} %, "
                      f"{time.time() - t0:.0f} s", flush=True)

    summary = {"episodes": len(episodes), "transitions": int(sum(len(e["actions"]) for e in episodes))}
    for beh in ("scripted", "bc"):
        es = [e for e in episodes if e["info"]["behaviour"] == beh]
        if es:
            summary[beh] = {"episodes": len(es), "success_rate": float(np.mean([e["success"] for e in es])),
                            "drop_rate": float(np.mean([e["info"]["dropped"] for e in es])),
                            "collision_episode_rate": float(np.mean([e["info"]["collision_steps"] > 0 for e in es]))}
    summary["success_rate"] = float(np.mean([e["success"] for e in episodes]))
    meta = {"source": "env rollouts (generate_offline_data.py)", "bc_checkpoint": args.bc,
            "max_noise": args.max_noise, "bc_frac": args.bc_frac, "seeds": [seeds[0], seeds[-1]],
            "reward_terms_included": list(REWARD_TERMS), "summary": summary}
    save_transitions(args.out, episodes, meta)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
