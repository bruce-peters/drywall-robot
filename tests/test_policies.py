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


def test_release_threshold_only_affects_attached_states():
    """release_threshold changes the suction decision only while suction is on; grasping uses 0.5."""
    from drywall_env.transitions import state_slices

    torch.manual_seed(0)
    actor = ActorNet(chunk=1)
    s = np.zeros((2, STATE_DIM), np.float32)
    s[1, state_slices(STATE_DIM)["attached"]] = 1.0  # row 0 detached, row 1 attached
    with torch.no_grad():  # force p(attach) = sigmoid(0.4) ~ 0.6 for every state
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.zero_()
        actor.net[-1].bias[6] = 0.4
    assert np.all(TorchPolicy(actor).predict(s)[:, 0, 6] == 1.0)  # default 0.5: hold / attach
    out = TorchPolicy(actor, release_threshold=0.7).predict(s)[:, 0, 6]
    assert out[0] == 1.0 and out[1] == -1.0  # detached still grasps; attached now releases


def test_bce_weight_scales_attach_loss():
    torch.manual_seed(0)
    actor = ActorNet(chunk=2)
    s = torch.randn(3, STATE_DIM)
    a = torch.zeros(3, 2, 7)
    a[..., 6] = -1.0
    _, _, bce1 = actor.loss_per_sample(s, a)
    _, _, bce3 = actor.loss_per_sample(s, a, bce_weight=torch.full((3, 2), 3.0))
    torch.testing.assert_close(bce3, 3 * bce1)


def test_wall_frame_policy_is_rotation_equivariant():
    """Rotating the whole scene about the base z-axis rotates a wall-frame policy's actions with it,
    and the wall frame puts the target straight ahead."""
    from drywall_env import EnvConfig
    from drywall_env.transitions import STATE_DIM_V2, rotate_actions_z, rotate_states_z, state_from_obs, wall_yaw

    env = DrywallPlacementEnv(EnvConfig(challenge="hard"))
    s = np.stack([state_from_obs(env.reset(seed=k)[0], STATE_DIM_V2) for k in range(4)])
    env.close()
    np.testing.assert_allclose(wall_yaw(rotate_states_z(s, -wall_yaw(s))), 0.0, atol=1e-6)
    torch.manual_seed(0)
    for head in ("mse", "gmm"):
        actor = ActorNet(state_dim=STATE_DIM_V2, chunk=2, frame="wall", head=head)
        actor.norm.fit(s)
        pol = TorchPolicy(actor)
        theta = np.radians(37.0)
        a = pol.predict(s)
        a_rot = pol.predict(rotate_states_z(s, theta))
        np.testing.assert_allclose(a_rot, rotate_actions_z(a, theta), atol=1e-4)


def test_gmm_head_shapes_and_loss_prefers_a_data_mode():
    torch.manual_seed(0)
    actor = ActorNet(chunk=3, head="gmm", n_modes=4)
    s = torch.randn(5, STATE_DIM)
    motion, logit = actor(s)
    assert motion.shape == (5, 3, 6) and logit.shape == (5, 3)
    assert torch.all(motion.abs() <= 1.0)
    a = torch.zeros(5, 3, 7)
    a[..., :6] = motion.detach()  # targets at the chosen mode's mean ...
    near, _, _ = actor.loss_per_sample(s, a)
    a[..., :6] = -motion.detach()  # ... are more likely than far-away targets
    far, _, _ = actor.loss_per_sample(s, a)
    assert torch.all(near < far)
