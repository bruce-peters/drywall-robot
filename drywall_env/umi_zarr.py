"""Convert the UMI-style .npz (see umi_recorder.py) into UMI's own replay-buffer format.

UMI (real-stanford/universal_manipulation_interface, scripts_slam_pipeline/07_generate_replay_buffer.py)
stores a zarr v2 ZipStore with one row per camera frame:

  data/camera0_rgb                (N, 224, 224, 3) uint8    JpegXl level 99, chunks (1, H, W, 3)
  data/robot0_eef_pos             (N, 3) float32            absolute, world frame
  data/robot0_eef_rot_axis_angle  (N, 3) float32            absolute, world frame
  data/robot0_gripper_width       (N, 1) float32            HERE: suction attached as 0.0 / 1.0
  data/robot0_demo_start_pose     (N, 6) float64            pose (pos + axis-angle) at episode start
  data/robot0_demo_end_pose       (N, 6) float64            pose at episode end
  meta/episode_ends               (E,) int64                cumulative row counts

Rows are at 10 Hz (the image rate): row k = pose sample 6k of the 60 Hz stream. In UMI's training
config use down_sample_steps = 1 for 10 Hz.

Extras (not in UMI; UMI's loader only reads the keys named in its shape_meta, so they are ignored):
  data/external_rgb, data/timestamp, data/panel_pos_in_target, data/panel_rot_axis_angle_in_target,
  meta/episode_success (E,) bool, and root attrs "camera_calibration" (JSON, OpenCV conventions)
  and "mirror_swap" (bool).

camera0_rgb has UMI's mirror swap applied by default (as UMI's pipeline does with --mirror_swap):
the side-mirror crops are digitally reflected. NOTE: the source .npz keeps the RAW view, so
camera0_rgb differs from npz wrist_rgb inside the two mirror regions.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .transforms import quat_to_mat, quat_to_rotvec, rotvec_to_quat

IMAGE_EVERY = 6  # 60 Hz poses / 10 Hz images


def _rotvecs(quats: np.ndarray) -> np.ndarray:
    return np.array([quat_to_rotvec(q) for q in np.asarray(quats, float)]).reshape(-1, 3)


def pose_to_mat(pos: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """(…,3) positions + (…,3) axis-angle -> (…,4,4) homogeneous transforms."""
    pos, rotvec = np.asarray(pos, float).reshape(-1, 3), np.asarray(rotvec, float).reshape(-1, 3)
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = [quat_to_mat(rotvec_to_quat(r)) for r in rotvec]
    T[:, :3, 3] = pos
    return T


def mat_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    T = np.asarray(T, float).reshape(-1, 4, 4)
    rv = []
    for R in T[:, :3, :3]:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(9))
        rv.append(quat_to_rotvec(q))
    return T[:, :3, 3].copy(), np.array(rv)


def relative_trajectory(pos: np.ndarray, rotvec: np.ndarray, base_index: int = -1):
    """UMI's relative-trajectory representation (paper PD2; UMI code: pose_repr 'relative').

    Every pose is expressed relative to the pose at `base_index` (UMI: the current / latest
    observed EE pose): T_rel[t] = inv(T[base]) @ T[t]. Returns (pos_rel, rotvec_rel). Invariant to
    the choice of world frame, which is why UMI needs no robot-to-SLAM-map calibration.
    """
    T = pose_to_mat(pos, rotvec)
    return mat_to_pose(np.linalg.inv(T[base_index]) @ T)


def mirror_swap(frames: np.ndarray) -> np.ndarray:
    """UMI's digital reflection (07_generate_replay_buffer.py, --mirror_swap): inside both mirror
    regions, replace pixels with the horizontally flipped frame. The regions are left/right
    symmetric, so each mirror's crop is reflected into the opposite region, undoing the mirror flip."""
    from .umi_recorder import mirror_mask

    h, w = frames.shape[-3:-1]
    assert h == w
    mask = mirror_mask(h, "left") | mirror_mask(h, "right")
    out = frames.copy()
    out[..., mask, :] = frames[..., :, ::-1, :][..., mask, :]
    return out


def convert_npz_to_umi_zarr(npz_path, out_path, compression_level: int = 99, progress=None,
                            do_mirror_swap: bool = True):
    """Write UMI's replay-buffer .zarr.zip from our .npz. Images are written one chunk per frame,
    each exactly once (a ZipStore cannot overwrite); low-dim arrays are written once at the end.
    do_mirror_swap applies UMI's digital mirror reflection to camera0_rgb (raw stays in the npz)."""
    import zarr
    from imagecodecs.numcodecs import Jpegxl, register_codecs

    register_codecs(verbose=False)  # registers "imagecodecs_jpegxl", the same codec id UMI's loader uses
    src = np.load(npz_path)
    episodes = sorted({k.split("/")[0] for k in src.files if k.startswith("episode_")})
    lengths = [len(src[f"{e}/image_timestamp"]) for e in episodes]
    n_total = int(sum(lengths))
    h, w = src[f"{episodes[0]}/wrist_rgb"].shape[1:3]

    store = zarr.ZipStore(str(out_path), mode="w")
    root = zarr.group(store=store)
    data, meta = root.require_group("data"), root.require_group("meta")
    img_kw = dict(shape=(n_total, h, w, 3), chunks=(1, h, w, 3), dtype=np.uint8,
                  compressor=Jpegxl(level=compression_level, numthreads=1))
    cam0 = data.create_dataset("camera0_rgb", **img_kw)
    ext = data.create_dataset("external_rgb", **img_kw)

    low = {k: [] for k in ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width",
                           "robot0_demo_start_pose", "robot0_demo_end_pose", "timestamp",
                           "panel_pos_in_target", "panel_rot_axis_angle_in_target")}
    success, row = [], 0
    pool = ThreadPoolExecutor(max_workers=os.cpu_count())
    for i, (e, k) in enumerate(zip(episodes, lengths)):
        g = lambda name: src[f"{e}/{name}"]  # noqa: E731
        sel = slice(None, None, IMAGE_EVERY)
        pos, rv = g("ee_pos")[sel].astype(np.float32), _rotvecs(g("ee_quat")[sel]).astype(np.float32)
        assert len(pos) == k, f"{e}: {len(pos)} poses at image times vs {k} images"
        start = np.concatenate([g("ee_pos")[0], _rotvecs(g("ee_quat")[:1])[0]])
        end = np.concatenate([g("ee_pos")[-1], _rotvecs(g("ee_quat")[-1:])[0]])
        low["robot0_eef_pos"].append(pos)
        low["robot0_eef_rot_axis_angle"].append(rv)
        low["robot0_gripper_width"].append(g("attached")[sel].astype(np.float32)[:, None])
        low["robot0_demo_start_pose"].append(np.tile(start, (k, 1)))
        low["robot0_demo_end_pose"].append(np.tile(end, (k, 1)))
        low["timestamp"].append(g("timestamp")[sel])
        low["panel_pos_in_target"].append(g("panel_pos_in_target")[sel].astype(np.float32))
        low["panel_rot_axis_angle_in_target"].append(
            _rotvecs(g("panel_quat_in_target")[sel]).astype(np.float32))
        for arr, name in ((cam0, "wrist_rgb"), (ext, "external_rgb")):
            frames = g(name)
            if name == "wrist_rgb" and do_mirror_swap:
                frames = mirror_swap(frames)
            # Encode on a thread pool (imagecodecs releases the GIL), then store each frame's
            # chunk directly: key "<array path>/<row>.0.0.0" with chunks (1, H, W, 3). Same bytes
            # as arr[row:row+k] = frames, just parallel.
            encoded = list(pool.map(arr.compressor.encode, frames))
            for j, buf in enumerate(encoded):
                store[f"{arr.path}/{row + j}.0.0.0"] = buf
        success.append(bool(g("success")))
        row += k
        if progress:
            progress(i, len(episodes))

    pool.shutdown()
    for name, parts in low.items():
        arr = np.concatenate(parts)
        data.array(name, arr, chunks=arr.shape, compressor=None)  # UMI: low-dim uncompressed
    meta.array("episode_ends", np.cumsum(lengths).astype(np.int64), compressor=None)
    meta.array("episode_success", np.array(success), compressor=None)
    calib = {k.split("/", 1)[1]: src[k].tolist() for k in src.files
             if k.startswith("meta/") and any(w in k for w in ("wrist", "external", "mirror"))}
    # One attrs write: a ZipStore cannot overwrite .zattrs.
    root.attrs.put({"camera_calibration": json.loads(json.dumps(calib)),
                    "mirror_swap": bool(do_mirror_swap),
                    "source": "drywall-robot sim, converted from " + str(npz_path)})
    store.close()
    return {"episodes": len(episodes), "rows": n_total}
