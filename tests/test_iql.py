import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expectile_loss():
    iql = _load_script("train_iql")
    u = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    np.testing.assert_allclose(iql.expectile_loss(u, 0.7).numpy(), [0.3 * 4, 0.3 * 1, 0.7 * 1, 0.7 * 4])
    # tau = 0.5 is half the squared error; the minimiser of E[L2^tau(x - v)] is the tau-expectile
    x = torch.tensor([0.0, 1.0, 2.0, 10.0])
    v = torch.zeros((), requires_grad=True)
    opt = torch.optim.SGD([v], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        iql.expectile_loss(x - v, 0.9).mean().backward()
        opt.step()
    assert v.item() > x.mean().item()  # upper expectile lies above the mean
