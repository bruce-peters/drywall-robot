"""Learned policies (PyTorch) and a common `act(obs) -> action` interface.

Every policy here maps the 37-D state (transitions.state_from_obs, deployable observations only)
to the env's 7-D action. The same actor network is used by
  * behaviour cloning (scripts/train_bc.py):  loss = MSE(motion) + BCE(attach)
  * IQL (scripts/train_iql.py):               the same loss, weighted per sample by exp(beta * A)
so BC and IQL differ only in the training signal, not in the architecture.

Action head: 6 motion dims through tanh (the env's [-1, 1] range) and 1 attach logit
(attach = +1 if sigmoid(logit) > 0.5). Optional action chunking: the network predicts the next
H actions; at run time the first `n_exec` of them are executed before re-planning (receding
horizon, as in Diffusion Policy / ACT). H = 1 is plain single-step BC.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .transitions import STATE_DIM, state_from_obs


class Normalizer(nn.Module):
    """x -> (x - mean) / std with statistics from the training data. Constant dimensions (for
    example the target orientation, which never changes in this task) get std = 1."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: np.ndarray):
        x = np.asarray(x, np.float64)
        std = x.std(0)
        self.mean.copy_(torch.as_tensor(x.mean(0), dtype=torch.float32))
        self.std.copy_(torch.as_tensor(np.where(std < 1e-6, 1.0, std), dtype=torch.float32))
        return self

    def forward(self, x):
        return (x - self.mean) / self.std


def mlp(inp: int, out: int, hidden=(256, 256, 256), layer_norm: bool = True) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if layer_norm else []) + [nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden=(256, 256, 256), chunk: int = 1):
        super().__init__()
        self.state_dim, self.hidden, self.chunk = state_dim, tuple(hidden), chunk
        self.norm = Normalizer(state_dim)
        self.net = mlp(state_dim, 7 * chunk, hidden)

    def forward(self, s):
        """s (B, 37) -> motion (B, H, 6) in [-1, 1], attach logits (B, H)."""
        out = self.net(self.norm(s)).view(-1, self.chunk, 7)
        return torch.tanh(out[..., :6]), out[..., 6]

    def loss_per_sample(self, s, a_chunk, attach_weight: float = 1.0, bce_weight=None):
        """Per-sample negative log-likelihood surrogate: MSE on motion + BCE on attach,
        averaged over the chunk. a_chunk (B, H, 7) with attach in {-1, +1}. bce_weight (B, H),
        optional: per-element weights on the attach BCE (e.g. to up-weight the rare release)."""
        motion, logit = self(s)
        mse = ((motion - a_chunk[..., :6]) ** 2).sum(-1).mean(-1)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logit, (a_chunk[..., 6] > 0).float(), reduction="none")
        bce = (bce * bce_weight).mean(-1) if bce_weight is not None else bce.mean(-1)
        return mse + attach_weight * bce, mse, bce

    def config(self) -> dict:
        return {"state_dim": self.state_dim, "hidden": list(self.hidden), "chunk": self.chunk}


def action_chunks(actions: np.ndarray, episode_ends: np.ndarray, chunk: int) -> np.ndarray:
    """(N,7) actions -> (N,H,7) targets a_t..a_{t+H-1}, padded with the episode's last action."""
    if chunk == 1:
        return actions[:, None, :]
    out = np.empty((len(actions), chunk, actions.shape[1]), actions.dtype)
    start = 0
    for end in episode_ends:
        idx = np.arange(start, end)
        for h in range(chunk):
            out[start:end, h] = actions[np.minimum(idx + h, end - 1)]
        start = end
    return out


class TorchPolicy:
    """Runs an ActorNet in the env: obs -> 37-D state -> action, with receding-horizon chunks."""

    def __init__(self, actor: ActorNet, n_exec: int | None = None, device="cpu", name="policy",
                 release_threshold: float = 0.5):
        self.actor = actor.to(device).eval()
        self.device = device
        self.n_exec = n_exec or actor.chunk
        self.name = name
        # While suction is on, keep it on only if p(attach) > release_threshold (grasping always
        # uses 0.5). 0.5 = the plain classifier; tuned on validation seeds only.
        self.release_threshold = release_threshold
        self._queue: deque = deque()

    def reset(self):
        self._queue.clear()

    @torch.no_grad()
    def predict(self, states: np.ndarray) -> np.ndarray:
        """Batched: (B, 37) -> (B, H, 7) actions."""
        from .transitions import state_slices

        s = torch.as_tensor(np.asarray(states, np.float32), device=self.device).view(-1, self.actor.state_dim)
        motion, logit = self.actor(s)
        attached = s[:, state_slices(self.actor.state_dim)["attached"]].view(-1, 1) > 0.5
        rel = torch.logit(torch.tensor(float(self.release_threshold), device=s.device))
        thr = torch.where(attached, rel, torch.zeros((), device=s.device))
        attach = torch.where(logit > thr, 1.0, -1.0).unsqueeze(-1)
        return torch.cat([motion, attach], -1).cpu().numpy()

    def act(self, obs) -> np.ndarray:
        if not self._queue:
            # the state version (37-D v1 / 46-D v2) follows the network's input size
            chunk = self.predict(state_from_obs(obs, self.actor.state_dim)[None])[0]
            self._queue.extend(chunk[: self.n_exec])
        return self._queue.popleft().astype(np.float32)


class ScriptedPolicy:
    """The scripted controller behind the same interface (for eval / the policy viewer)."""

    def __init__(self, cfg=None, action_noise_std: float = 0.0, seed=None):
        from .scripted import ScriptedController

        self.ctrl = ScriptedController(cfg, action_noise_std=action_noise_std, seed=seed)
        self.name = "scripted"

    def reset(self):
        self.ctrl.reset()

    def act(self, obs):
        return self.ctrl.act(obs)

    @property
    def phase(self):
        return self.ctrl.phase


def save_actor(path, actor: ActorNet, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": actor.config(), "state_dict": actor.state_dict(), "extra": extra or {}}, path)


def load_policy(path, device="cpu", n_exec: int | None = None, release_threshold: float = 0.5) -> TorchPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = ActorNet(**ck["config"])
    actor.load_state_dict(ck["state_dict"])
    pol = TorchPolicy(actor, n_exec=n_exec or ck["extra"].get("n_exec"), device=device,
                      name=Path(path).stem, release_threshold=release_threshold)
    pol.extra = ck["extra"]
    return pol
