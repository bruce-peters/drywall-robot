"""Hard challenge: layout sampling, reward weights, state v2, scripted controller."""

import numpy as np
import pytest

from drywall_env import DrywallPlacementEnv, EnvConfig
from drywall_env.scripted_hard import make_scripted
from drywall_env.transforms import pose_inv, pose_mul, quat_to_mat
from drywall_env.transitions import STATE_DIM, STATE_DIM_V2, rot6_to_mat, state_from_obs, state_slices


def _in_demo(cfg, r):
    return (cfg.hard_demo_wall_angle[0] <= r["wall_angle"] <= cfg.hard_demo_wall_angle[1]
            and cfg.hard_demo_rack_gap[0] <= r["rack_angle"] - r["wall_angle"] <= cfg.hard_demo_rack_gap[1])


@pytest.mark.parametrize("region", ["full", "demo", "ood"])
def test_hard_layout_sampling(region):
    env = DrywallPlacementEnv(EnvConfig(challenge="hard", region=region))
    cfg = env.cfg
    for seed in range(25):
        _, info = env.reset(seed=seed)
        r = info["randomization"]
        gap = abs(r["rack_angle"] - r["wall_angle"])
        assert abs(r["rack_angle"]) <= cfg.hard_rack_angle_limit + 1e-9
        assert cfg.hard_rack_gap[0] - 1e-9 <= gap <= cfg.hard_rack_gap[1] + 1e-9 or region == "demo"
        assert cfg.hard_rack_radius[0] <= r["rack_radius"] <= cfg.hard_rack_radius[1]
        assert cfg.hard_target_z[0] <= r["target_z"] <= cfg.hard_target_z[1]
        if region == "demo":
            assert _in_demo(cfg, r)
        if region == "ood":
            assert not _in_demo(cfg, r)
        # the target really sits on the rotated wall at the sampled height
        tp, _ = env._target_pose_world()
        assert tp[2] == pytest.approx(r["target_z"], abs=1e-6)
        assert np.arctan2(tp[1], tp[0]) == pytest.approx(
            r["wall_angle"] + np.arctan2(r["target_dy"], env._nom_target_pos[0]), abs=1e-6)
    env.close()


def test_success_weight_per_challenge():
    assert EnvConfig().reward_weights["success"] == 100.0  # easy results stay reproducible
    assert EnvConfig(challenge="hard").reward_weights["success"] == 1000.0


def test_state_v2_panel_in_ee():
    env = DrywallPlacementEnv(EnvConfig(challenge="hard"))
    obs, _ = env.reset(seed=3)
    s1, s2 = state_from_obs(obs), state_from_obs(obs, STATE_DIM_V2)
    assert s1.shape == (STATE_DIM,) and s2.shape == (STATE_DIM_V2,)
    np.testing.assert_array_equal(s2[:STATE_DIM], s1)  # v2 extends v1
    d = obs["deployable"]
    rel_p, rel_q = pose_mul(*pose_inv(d["ee_pos"], d["ee_quat"]), d["panel_pos_est"], d["panel_quat_est"])
    sl = state_slices(STATE_DIM_V2)
    np.testing.assert_allclose(s2[sl["panel_in_ee_pos"]], rel_p, atol=1e-5)
    np.testing.assert_allclose(rot6_to_mat(s2[sl["panel_in_ee_rot6"]])[0], quat_to_mat(rel_q), atol=1e-5)
    env.close()


def test_hard_scripted_demo_succeeds_with_rack_behind():
    """Seed 1 (full region): wall at 4 deg, rack behind the robot at 149 deg."""
    env = DrywallPlacementEnv(EnvConfig(challenge="hard"))
    obs, info = env.reset(seed=1)
    assert abs(info["randomization"]["rack_angle"]) > np.radians(120)
    ctrl = make_scripted(env.cfg, mode="demo", seed=1)
    done = False
    while not done:
        obs, _, te, tr, info = env.step(ctrl.act(obs))
        done = te or tr
    assert info["success"]
    env.close()
