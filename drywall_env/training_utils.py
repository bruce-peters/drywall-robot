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
from .transitions import VAL_SEEDS


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


def validation_rollouts(actor, env, n: int = 50, n_exec=None) -> dict:
    """Run a CPU copy of the actor on the first n validation seeds; returns the summary."""
    pol = TorchPolicy(copy.deepcopy(actor).cpu(), n_exec=n_exec, device="cpu")
    with torch.no_grad():
        _, summary = evaluate(env, pol, list(VAL_SEEDS)[:n])
    return summary


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
