"""Small helpers shared by train_bc.py and train_iql.py: JSONL logging and validation rollouts.

Validation rollouts (LABNOTEBOOK 2026-11-17): every so often the CURRENT actor is run for 50
episodes on VAL_SEEDS (5000-5049). This is measurement only -- the episodes are never added to the
dataset, never used for a gradient step and never used to choose a checkpoint (the saved model is
always the last step) -- so training stays offline. Test seeds (10000-10099) are never touched here.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate
from .policies import TorchPolicy
from .transitions import HARD_VAL_SEEDS, VAL_SEEDS


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")
        self.t0 = time.time()

    def log(self, kind: str, step: int, **values):
        rec = {"kind": kind, "step": int(step), "time_s": round(time.time() - self.t0, 2)}
        def clean(v):
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                return float(v)
            return v

        rec.update({k: clean(v) for k, v in values.items()})
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def val_seeds_for(env):
    """Validation seeds: 5000-5049 (easy) or 25000-25049 (hard, full region)."""
    return HARD_VAL_SEEDS if env.cfg.challenge == "hard" else VAL_SEEDS


def validation_rollouts(actor, env, n: int = 50, n_exec=None) -> dict:
    """Run a CPU copy of the actor on the first n validation seeds; returns the summary."""
    pol = TorchPolicy(copy.deepcopy(actor).cpu(), n_exec=n_exec, device="cpu")
    seeds = list(val_seeds_for(env))[:n]
    with torch.no_grad():
        _, summary = evaluate(env, pol, seeds)
    summary["seeds"] = [seeds[0], seeds[-1]]
    return summary


def make_env(challenge: str, region: str = "full"):
    from .env import DrywallPlacementEnv, EnvConfig

    return DrywallPlacementEnv(EnvConfig(challenge=challenge, region=region if challenge == "hard" else "full"))


def default_log_path(checkpoint) -> Path:
    """runs/<same subfolder as the checkpoint>/<stem>/log.jsonl, e.g.
    checkpoints/hard_challenge/bc.pt -> runs/hard_challenge/bc/log.jsonl."""
    ck = Path(checkpoint)
    parts = ck.parent.parts
    sub = Path(*parts[parts.index("checkpoints") + 1:]) if "checkpoints" in parts else Path()
    return Path("runs") / sub / ck.stem / "log.jsonl"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def snapshot_path(checkpoint, step: int) -> Path:
    """checkpoints/<sub>/snapshots/<stem>/step_XXXXXX.pt: the actor at a validation step, saved for
    VIEWING only (view_policy.py --snapshots). The reported model is always the final checkpoint."""
    ck = Path(checkpoint)
    return ck.parent / "snapshots" / ck.stem / f"step_{step:06d}.pt"
