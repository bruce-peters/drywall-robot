import numpy as np
import pytest

from drywall_env import DrywallPlacementEnv, ScriptedController
from drywall_env.transforms import quat_to_mat, quat_to_rotvec, rpy_to_mat
from drywall_env.transitions import (
    STATE_DIM, TEST_SEEDS, UMI_REWARD_TERMS, actions_from_poses, check_not_test_seeds,
    load_transitions, mat_to_rot6, quat_to_mat_batch, resample_poses, rot6_to_mat,
    save_transitions, state_from_obs, umi_reward_terms,
)
from drywall_env.umi_to_robot import cell_calibration, convert_episode


@pytest.fixture(scope="module")
def env():
    e = DrywallPlacementEnv()
    yield e
    e.close()


def random_rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return quat_to_mat_batch(q)


def test_rot6_roundtrip():
    R = random_rotations(50)
    np.testing.assert_allclose(rot6_to_mat(mat_to_rot6(R)), R, atol=1e-12)


def test_state_from_obs_shape_and_truth(env):
    obs, _ = env.reset(seed=0)
    s = state_from_obs(obs)
    assert s.shape == (STATE_DIM,) and s.dtype == np.float32
    np.testing.assert_allclose(s[:3], obs["deployable"]["ee_pos"])  # ee_pos
    np.testing.assert_allclose(s[10:13], obs["privileged"]["panel_pos"], atol=1e-6)  # panel_pos (no noise)


def test_resample_exact_and_midpoint():
    t = np.arange(5) * 0.1
    p = np.cumsum(np.ones((5, 3)), axis=0)
    R = random_rotations(5, seed=1)
    p2, R2 = resample_poses(t, p, R, t)  # same timestamps -> exact
    np.testing.assert_allclose(p2, p)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    # midpoint between two rotations a known angle apart (90 deg about z) -> 45 deg about z
    Ra = np.eye(3)
    Rb = rpy_to_mat(0, 0, np.pi / 2)
    _, Rm = resample_poses([0.0, 1.0], np.zeros((2, 3)), np.stack([Ra, Rb]), [0.5])
    np.testing.assert_allclose(Rm[0], rpy_to_mat(0, 0, np.pi / 4), atol=1e-9)


def test_actions_reproduce_next_pose(env):
    """Integrating the derived actions the way env.step does gives back the next pose."""
    rng = np.random.default_rng(0)
    p = np.cumsum(rng.uniform(-0.01, 0.01, (20, 3)), axis=0)
    R = [np.eye(3)]
    for _ in range(19):
        R.append(rpy_to_mat(*rng.uniform(-0.03, 0.03, 3)) @ R[-1])
    R = np.array(R)
    a, raw = actions_from_poses(p, R, np.ones(20), env.cfg)
    assert np.all(raw <= 1.0)
    for k in range(19):
        np.testing.assert_allclose(p[k] + a[k, :3] * env.cfg.max_pos_step, p[k + 1], atol=1e-6)
        R_next = rpy_to_mat(*(a[k, 3:6] * env.cfg.max_rot_step)) @ R[k]
        np.testing.assert_allclose(R_next, R[k + 1], atol=1e-5)
    assert np.all(a[:, 6] == 1.0)


def _umi_rows_from_env(env, seed, n_steps=120, T_world_base=None):
    """Run the scripted controller and record UMI-style rows (world frame) at every control step.
    T_world_base optionally re-expresses the rows in a different 'SLAM map' world frame."""
    obs, _ = env.reset(seed=seed)
    ctrl = ScriptedController(env.cfg)
    rows = {k: [] for k in ("timestamp", "robot0_eef_pos", "robot0_eef_rot_axis_angle",
                            "robot0_gripper_width", "panel_pos_in_target", "panel_rot_axis_angle_in_target")}
    states = []
    T = np.eye(4) if T_world_base is None else T_world_base
    from drywall_env.transforms import mat_to_quat, pose_inv, pose_mul

    def record(obs):
        ee_p, ee_q = env._ee_pose_world()
        tp, tq = env._target_pose_world()
        pp, pq = env._body_pose(env._panel_body)
        rel_p, rel_q = pose_mul(*pose_inv(tp, tq), pp, pq)
        R = T[:3, :3] @ quat_to_mat(ee_q)
        rows["timestamp"].append(env.data.time)
        rows["robot0_eef_pos"].append(T[:3, :3] @ ee_p + T[:3, 3])
        rows["robot0_eef_rot_axis_angle"].append(quat_to_rotvec(mat_to_quat(R)))
        rows["robot0_gripper_width"].append([float(env.attached)])
        rows["panel_pos_in_target"].append(rel_p)
        rows["panel_rot_axis_angle_in_target"].append(quat_to_rotvec(rel_q))
        states.append(state_from_obs(obs))

    record(obs)
    for _ in range(n_steps):
        obs, *_ = env.step(ctrl.act(obs))
        record(obs)
    return {k: np.array(v) for k, v in rows.items()}, np.array(states)


@pytest.mark.parametrize("map_frame", [False, True])
def test_umi_conversion_matches_env_states(env, map_frame):
    """UMI rows -> 37-D states equal the env's own states, also when the UMI 'world' is a
    different (SLAM map) frame and T_base_world is not the identity."""
    T_map_world = np.eye(4)
    if map_frame:
        T_map_world[:3, :3] = rpy_to_mat(0.1, -0.2, 0.7)
        T_map_world[:3, 3] = [1.5, -2.0, 0.3]
    rows, env_states = _umi_rows_from_env(env, seed=4, T_world_base=T_map_world)
    env.reset(seed=4)
    T_base_world, target_nom, target_R = cell_calibration(env)
    # the target is known in the robot's world; express it in the map frame like the EE rows
    tp_w = env._target_pose_world()[0]
    tp_map = T_map_world[:3, :3] @ tp_w + T_map_world[:3, 3]
    tR_map = T_map_world[:3, :3] @ target_R
    T_base_map = T_base_world @ np.linalg.inv(T_map_world)
    ep = convert_episode(rows, tp_map, tR_map, T_base_map, env.cfg, success=False, dt=env.dt)
    np.testing.assert_allclose(ep["observations"], env_states, atol=2e-5)
    assert ep["actions"].shape == (len(env_states) - 1, 7)


def test_umi_reward_terms_match_env(env):
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(0)
    states, acts, env_terms = [state_from_obs(obs)], [], []
    for _ in range(15):
        a = rng.uniform(-1, 1, 7).astype(np.float32)
        obs, _, _, _, info = env.step(a)
        states.append(state_from_obs(obs))
        acts.append(a)
        env_terms.append(info["reward_terms"])
    terms = umi_reward_terms(np.array(states[1:]), np.array(acts), success_last=False, cfg=env.cfg)
    for k in ("pos_error", "ori_error", "action", "jerk"):
        np.testing.assert_allclose(terms[k], [t[k] for t in env_terms], rtol=1e-4, atol=1e-5)


def test_test_seeds_are_refused():
    check_not_test_seeds(range(0, 2000))
    with pytest.raises(ValueError):
        check_not_test_seeds([5, TEST_SEEDS[0]])


def test_save_load_roundtrip_and_relabel(tmp_path):
    rng = np.random.default_rng(0)
    eps = []
    for i, (T, succ) in enumerate([(5, True), (3, False)]):
        terms = {k: rng.normal(size=T).astype(np.float32) for k in ("pos_error", "collision", "success")}
        eps.append({"observations": rng.normal(size=(T + 1, STATE_DIM)).astype(np.float32),
                    "actions": rng.uniform(-1, 1, (T, 7)).astype(np.float32), "reward_terms": terms,
                    "terminal": succ, "success": succ, "seed": i})
    path = tmp_path / "t.npz"
    save_transitions(path, eps, {"reward_terms_included": ["pos_error", "collision", "success"]})
    d = load_transitions(path)
    assert d["observations"].shape == (8, STATE_DIM)
    np.testing.assert_array_equal(d["next_observations"][:5], eps[0]["observations"][1:])
    assert d["terminals"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
    assert d["timeouts"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert d["episode_ends"].tolist() == [5, 8]
    np.testing.assert_allclose(d["rewards"], d["reward/pos_error"] + d["reward/collision"] + d["reward/success"])
    d2 = load_transitions(path, reward_terms=UMI_REWARD_TERMS)
    np.testing.assert_allclose(d2["rewards"], d["reward/pos_error"] + d["reward/success"])
