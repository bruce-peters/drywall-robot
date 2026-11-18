import numpy as np
import pytest
import torch

from drywall_env import DrywallPlacementEnv
from drywall_env.evaluation import summarize
from drywall_env.policies import ActorNet, TorchPolicy, action_chunks, load_policy, save_actor
from drywall_env.transitions import STATE_DIM


def test_action_chunks_pad_within_episode():
    a = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    c = action_chunks(a, np.array([3, 5]), chunk=3)
    assert c.shape == (5, 3, 7)
    np.testing.assert_array_equal(c[0, :, 0], a[[0, 1, 2], 0])
    np.testing.assert_array_equal(c[1, :, 0], a[[1, 2, 2], 0])  # padded with the episode's last action
    np.testing.assert_array_equal(c[3, :, 0], a[[3, 4, 4], 0])  # never crosses into the next episode


def test_actor_output_range_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    actor = ActorNet(chunk=4)
    actor.norm.fit(np.random.default_rng(0).normal(size=(100, STATE_DIM)))
    path = tmp_path / "a.pt"
    save_actor(path, actor, extra={"n_exec": 2})
    pol = load_policy(path)
    s = np.random.default_rng(1).normal(size=(6, STATE_DIM)).astype(np.float32)
    out = pol.predict(s)
    assert out.shape == (6, 4, 7)
    assert np.all(np.abs(out[..., :6]) <= 1.0) and set(np.unique(out[..., 6])) <= {-1.0, 1.0}
    np.testing.assert_allclose(out, TorchPolicy(actor.eval()).predict(s))
    assert pol.n_exec == 2


def test_receding_horizon_replans_every_n_exec():
    actor = ActorNet(chunk=4).eval()
    pol = TorchPolicy(actor, n_exec=2)
    calls = []
    orig = pol.predict
    pol.predict = lambda s: calls.append(1) or orig(s)
    env = DrywallPlacementEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(6):
        obs, *_ = env.step(pol.act(obs))
    env.close()
    assert len(calls) == 3  # 6 steps, 2 executed per prediction


def test_summarize_wilson_interval():
    rows = [{"success": i < 8, "dropped": False, "collision": False, "pos_error_mm": 3.0,
             "ori_error_deg": 0.1, "length": 170, "max_wall_force": 10.0, "return": 0.0} for i in range(10)]
    s = summarize(rows)
    assert s["success_rate"] == pytest.approx(0.8)
    lo, hi = s["success_ci95"]
    assert 0.4 < lo < 0.8 < hi < 1.0
