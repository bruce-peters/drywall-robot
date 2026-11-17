import numpy as np
import pytest

from drywall_env import DrywallPlacementEnv, EnvConfig, ScriptedController


@pytest.fixture(scope="module")
def env():
    e = DrywallPlacementEnv()
    yield e
    e.close()


def rollout(env, seed, n_steps=60, action_seed=123):
    """Fixed random-action rollout; returns stacked policy obs, rewards and infos."""
    rng = np.random.default_rng(action_seed)
    obs, info = env.reset(seed=seed)
    obs_list, rewards, rand = [obs["policy"].copy()], [], info["randomization"]
    for _ in range(n_steps):
        a = rng.uniform(-1, 1, 7).astype(np.float32)
        a[6] = 1.0
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(obs["policy"].copy())
        rewards.append(r)
        if term or trunc:
            break
    return np.array(obs_list), np.array(rewards), rand


def test_same_seed_identical_rollout(env):
    o1, r1, rand1 = rollout(env, seed=7)
    o2, r2, rand2 = rollout(env, seed=7)
    assert rand1 == rand2
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)


def test_same_seed_identical_scripted_rollout(env):
    def run():
        obs, _ = env.reset(seed=3)
        ctrl = ScriptedController(env.cfg)
        traj = [obs["policy"].copy()]
        for _ in range(80):
            obs, *_ = env.step(ctrl.act(obs))
            traj.append(obs["policy"].copy())
        return np.array(traj)

    np.testing.assert_array_equal(run(), run())


def test_different_seeds_differ(env):
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=2)
    assert info_a["randomization"] != info_b["randomization"]


def test_randomization_within_ranges(env):
    cfg = env.cfg
    for seed in range(10):
        _, info = env.reset(seed=seed)
        r = info["randomization"]
        assert abs(r["panel_dx"]) <= cfg.panel_pos_range and abs(r["panel_dy"]) <= cfg.panel_pos_range
        assert abs(r["panel_dyaw"]) <= cfg.panel_yaw_range
        assert abs(r["target_dy"]) <= cfg.target_offset_range and abs(r["target_dz"]) <= cfg.target_offset_range
        assert cfg.panel_mass_range[0] <= r["panel_mass"] <= cfg.panel_mass_range[1]
        assert cfg.panel_friction_range[0] <= r["panel_friction"] <= cfg.panel_friction_range[1]
        # sampled values are actually applied to the model
        assert env.model.body_mass[env._panel_body] == pytest.approx(r["panel_mass"])
        assert env.model.geom_friction[env._panel_geom, 0] == pytest.approx(r["panel_friction"])


def test_action_clipping_reported_and_bounded(env):
    obs, _ = env.reset(seed=0)
    p0 = obs["deployable"]["ee_pos"].copy()
    _, _, _, _, info = env.step(np.array([5.0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    assert info["action_clipped"]
    # commanded move is at most max_pos_step even though the action asked for 5x
    assert np.linalg.norm(env._cmd_pos - p0) <= env.cfg.max_pos_step + 1e-9

    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    assert not info["action_clipped"]
    assert not info["workspace_clipped"]


def test_workspace_clipping(env):
    env.reset(seed=0)
    info = None
    for _ in range(60):  # push straight up: 60 x 2 cm would leave the workspace box
        _, _, term, trunc, info = env.step(np.array([0, 0, 1.0, 0, 0, 0, -1], dtype=np.float32))
    assert info["workspace_clipped"]
    assert env._cmd_pos[2] <= env.cfg.workspace_high[2] + 1e-12


def test_attach_refused_when_far(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32))
    assert info["attach_failed"] and not info["attached"]


def test_observation_groups(env):
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"deployable", "privileged", "policy"}
    assert env.observation_space.contains(obs)
    # with zero noise the estimates equal the privileged truth
    np.testing.assert_allclose(obs["deployable"]["panel_pos_est"], obs["privileged"]["panel_pos"])
    np.testing.assert_allclose(obs["deployable"]["target_pos_est"], obs["privileged"]["target_pos"])


def test_observation_noise():
    e = DrywallPlacementEnv(EnvConfig(pos_noise_std=0.01, rot_noise_std=0.01))
    obs, _ = e.reset(seed=0)
    assert not np.allclose(obs["deployable"]["panel_pos_est"], obs["privileged"]["panel_pos"])
    e.close()


def test_reward_terms_sum_to_reward(env):
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert r == pytest.approx(sum(info["reward_terms"].values()))
    assert set(info["reward_terms"]) == {"pos_error", "ori_error", "collision", "contact_force",
                                         "action", "jerk", "drop", "success"}


def test_scripted_success_rate():
    """Scripted controller succeeds on >= 90 % of 20 seeds."""
    e = DrywallPlacementEnv()
    ctrl = ScriptedController(e.cfg)
    successes = 0
    for seed in range(20):
        obs, _ = e.reset(seed=seed)
        ctrl.reset()
        term = trunc = False
        while not (term or trunc):
            obs, _, term, trunc, info = e.step(ctrl.act(obs))
        successes += bool(info["success"])
        assert term or not info["success"]
    e.close()
    assert successes >= 18, f"only {successes}/20 successes"
