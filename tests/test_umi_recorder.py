import sys
from pathlib import Path

import numpy as np

from drywall_env import DrywallPlacementEnv
from drywall_env.umi_recorder import NpzStreamWriter, UMIRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_umi import record_episode  # noqa: E402


def test_umi_episode_schema_and_timing(tmp_path):
    env = DrywallPlacementEnv()
    rec = UMIRecorder(env, image_size=64)
    try:
        ep, meta = record_episode(env, rec, seed=0, noise_std=0.0, t0=5.0)
    finally:
        rec.close()
        env.close()

    T, K = len(ep["timestamp"]), len(ep["wrist_rgb"])
    assert T == 6 * meta["n_control_steps"] + 1 and K == meta["n_control_steps"] + 1
    np.testing.assert_allclose(np.diff(ep["timestamp"]), 1 / 60, atol=1e-9)  # 60 Hz poses
    np.testing.assert_allclose(ep["image_timestamp"], ep["timestamp"][::6])  # 10 Hz images
    assert ep["start_time"] == 5.0 and ep["end_time"] == ep["timestamp"][-1]
    assert ep["ee_pos"].shape == (T, 3) and ep["ee_quat"].shape == (T, 4)
    assert ep["panel_pos_in_target"].shape == (T, 3) and ep["panel_quat_in_target"].shape == (T, 4)
    assert ep["wrist_rgb"].shape == ep["external_rgb"].shape == (K, 64, 64, 3)
    assert ep["wrist_rgb"][:, 0, 0].max() == 0  # fisheye corners are outside the image circle
    # Clean scripted demo: attaches once, releases once, succeeds with the panel on target.
    assert np.count_nonzero(np.diff(ep["attached"].astype(int))) == 2
    assert bool(ep["success"]) and np.linalg.norm(ep["panel_pos_in_target"][-1]) < 0.02
    assert env.substep_hook is None

    # Round trip through the streaming .npz writer.
    path = tmp_path / "d.npz"
    w = NpzStreamWriter(path)
    w.put_group("episode_000", ep)
    w.close()
    d = np.load(path)
    for k, v in ep.items():
        np.testing.assert_array_equal(d[f"episode_000/{k}"], v)


def _seg_hits(m, d, origin, R_cv, dirs_cv, project):
    """Ray-cast camera-frame directions; return the fraction whose projected pixel shows the hit geom."""
    import mujoco

    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # rendered groups only (no collision meshes)
    gid = np.array([-1], dtype=np.int32)
    ok = n = 0
    for dc in dirs_cv:
        dw = R_cv @ dc
        dist = mujoco.mj_ray(m, d, origin, dw, groups, 1, -1, gid)
        if gid[0] < 0:
            continue
        u, v, seg = project(R_cv.T @ (dw * dist))
        n += 1
        ok += int(seg[int(round(v)), int(round(u))] == gid[0])
    return ok / max(n, 1), n


def test_camera_calibration_matches_render():
    import mujoco

    from drywall_env.umi_recorder import FISHEYE_FOV_DEG, fisheye_maps

    env = DrywallPlacementEnv()
    env.reset(seed=0)
    for _ in range(40):  # move away from home so the wrist camera sees the rack and panel
        env.step(np.array([0.3, -0.6, -0.5, 0, 0, 0, -1.0]))
    n = 224
    rec = UMIRecorder(env, image_size=n)
    cal = rec.calibration()
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    try:
        # External pinhole camera.
        r = mujoco.Renderer(m, n, n)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=rec._ext_cam)
        seg = r.render()[..., 0]
        K, T = cal["external_K"], cal["T_world_external"]

        def proj_pin(pc):
            return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2], seg

        dirs = np.column_stack([rng.uniform(-0.38, 0.38, (300, 2)), np.ones(300)])
        frac, cnt = _seg_hits(m, d, T[:3, 3], T[:3, :3], dirs / np.linalg.norm(dirs, axis=1, keepdims=True), proj_pin)
        assert cnt > 250 and frac > 0.95, (frac, cnt)

        # Wrist fisheye: segmentation through the same remap, pose = EE pose @ T_ee_wrist.
        r.close()
        r = mujoco.Renderer(m, 768, 768)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=rec._wrist_cam)
        rows, cols, valid = fisheye_maps(n, FISHEYE_FOV_DEG, 768)
        segw = np.where(valid, r.render()[..., 0][rows, cols], -2)
        r.close()
        ee_p, ee_q = env._ee_pose_world()
        from drywall_env.transforms import quat_to_mat
        T_w_ee = np.eye(4)
        T_w_ee[:3, :3], T_w_ee[:3, 3] = quat_to_mat(ee_q), ee_p
        Tw = T_w_ee @ cal["T_ee_wrist"]
        Kw = cal["wrist_K"]

        def proj_fish(pc):
            th, ph = np.arctan2(np.hypot(pc[0], pc[1]), pc[2]), np.arctan2(pc[1], pc[0])
            return Kw[0, 2] + Kw[0, 0] * th * np.cos(ph), Kw[1, 2] + Kw[1, 1] * th * np.sin(ph), segw

        th = np.radians(rng.uniform(0, 70, 300))
        ph = rng.uniform(-np.pi, np.pi, 300)
        dirs = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        frac, cnt = _seg_hits(m, d, Tw[:3, 3], Tw[:3, :3], dirs, proj_fish)
        assert cnt > 250 and frac > 0.95, (frac, cnt)
        assert len(set(segw[valid].tolist())) > 3  # the view is not just one surface
    finally:
        rec.close()
        env.close()


def test_relative_trajectory():
    from drywall_env.umi_zarr import mat_to_pose, pose_to_mat, relative_trajectory

    rng = np.random.default_rng(1)
    pos, rv = rng.normal(size=(10, 3)), rng.uniform(-1, 1, (10, 3))
    rp, rr = relative_trajectory(pos, rv, base_index=-1)
    np.testing.assert_allclose(rp[-1], 0, atol=1e-9)
    np.testing.assert_allclose(rr[-1], 0, atol=1e-9)
    # Relative poses do not depend on the world frame (UMI's calibration-free property).
    G = pose_to_mat([0.3, -1.0, 2.0], [0.2, -0.4, 0.9])[0]
    p2, r2 = mat_to_pose(G @ pose_to_mat(pos, rv))
    rp2, rr2 = relative_trajectory(p2, r2, base_index=-1)
    np.testing.assert_allclose(rp2, rp, atol=1e-9)
    np.testing.assert_allclose(pose_to_mat(rp2, rr2), pose_to_mat(rp, rr), atol=1e-9)


def test_convert_to_umi_zarr(tmp_path):
    import zarr

    from drywall_env.umi_zarr import convert_npz_to_umi_zarr, mirror_swap

    env = DrywallPlacementEnv()
    rec = UMIRecorder(env, image_size=64)
    try:
        eps = [record_episode(env, rec, seed=s, noise_std=0.0, t0=0.0)[0] for s in (0, 1)]
        cal = rec.calibration()
    finally:
        rec.close()
        env.close()
    npz = tmp_path / "d.npz"
    w = NpzStreamWriter(npz)
    w.put_group("meta", cal)
    for i, ep in enumerate(eps):
        w.put_group(f"episode_{i:03d}", ep)
    w.close()

    out = tmp_path / "d.zarr.zip"
    convert_npz_to_umi_zarr(npz, out)
    root = zarr.open(zarr.ZipStore(str(out), mode="r"), mode="r")
    ks = [len(ep["wrist_rgb"]) for ep in eps]
    np.testing.assert_array_equal(root["meta/episode_ends"][:], np.cumsum(ks))
    N = sum(ks)
    for key, shape in {"camera0_rgb": (N, 64, 64, 3), "robot0_eef_pos": (N, 3),
                       "robot0_eef_rot_axis_angle": (N, 3), "robot0_gripper_width": (N, 1),
                       "robot0_demo_start_pose": (N, 6), "robot0_demo_end_pose": (N, 6)}.items():
        assert root["data"][key].shape == shape, key
    np.testing.assert_allclose(root["data/robot0_eef_pos"][: ks[0]], eps[0]["ee_pos"][::6])
    np.testing.assert_array_equal(root["data/robot0_gripper_width"][: ks[0], 0], eps[0]["attached"][::6])
    np.testing.assert_allclose(root["data/robot0_demo_start_pose"][ks[0], :3], eps[1]["ee_pos"][0])
    img = root["data/camera0_rgb"][5].astype(int)
    swapped = mirror_swap(eps[0]["wrist_rgb"][5])  # UMI's digital mirror reflection (default on)
    assert np.abs(img - swapped).mean() < 2  # JpegXl level 99: near lossless
    assert not np.array_equal(swapped, eps[0]["wrist_rgb"][5])
    assert root.attrs["mirror_swap"] is True
    assert root["meta/episode_success"][:].tolist() == [True, True]
    np.testing.assert_allclose(root.attrs["camera_calibration"]["wrist_K"], cal["wrist_K"])


def test_side_mirrors_are_planar_reflections():
    """Mirror pixels must show what a ray reflected off the stored mirror plane hits."""
    import mujoco

    from drywall_env.transforms import quat_to_mat
    from drywall_env.umi_mirrors import SIDES, fisheye_dirs, reflect
    from drywall_env.umi_recorder import mirror_mask

    env = DrywallPlacementEnv()
    env.reset(seed=0)
    for _ in range(40):
        env.step(np.array([0.3, -0.6, -0.5, 0, 0, 0, -1.0]))
    n = 224
    rec = UMIRecorder(env, image_size=n, supersample=1)
    try:
        cal = rec.calibration()
        seg = rec._wrist_image(segmentation=True)
        m, d = env.model, env.data
        ee_p, ee_q = env._ee_pose_world()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_to_mat(ee_q), ee_p
        T = T @ cal["T_ee_wrist"]
        groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        gid = np.array([-1], dtype=np.int32)
        rng = np.random.default_rng(0)
        seen = set()
        for i, side in enumerate(SIDES):
            nrm, pt = cal["mirror_normals"][i], cal["mirror_points"][i]
            # Inside the fisheye image circle only: UMI's polygons come from a 4:3 GoPro frame and
            # their outer corners (4.6 %) fall just outside our inscribed 155 deg circle (black).
            vv, uu = np.mgrid[0:n, 0:n]
            in_circle = np.hypot(uu - (n - 1) / 2, vv - (n - 1) / 2) <= n / 2
            rows, cols = np.nonzero(mirror_mask(n, side) & in_circle)
            pick = rng.choice(len(rows), 200, replace=False)
            ok = cnt = 0
            for r, c in zip(rows[pick], cols[pick]):
                u = fisheye_dirs(np.array([[(c + 0.5) / n, (r + 0.5) / n]]))[0]
                hit = u * (pt @ nrm) / (u @ nrm)  # on the mirror plane (camera frame)
                ref = reflect(u, nrm)
                dist = mujoco.mj_ray(m, d, T[:3, :3] @ hit + T[:3, 3], T[:3, :3] @ ref, groups, 1, -1, gid)
                if gid[0] < 0 or dist < 0:
                    continue
                cnt += 1
                ok += int(seg[r, c] == gid[0])
                seen.add(int(gid[0]))
            assert cnt > 150 and ok / cnt > 0.95, (side, ok, cnt)
        assert len(seen) >= 2  # the mirrors show a scene, not a single surface
        assert seg[mirror_mask(n, "left")].tolist() != seg[mirror_mask(n, "right")][::-1].tolist()
    finally:
        rec.close()
        env.close()
