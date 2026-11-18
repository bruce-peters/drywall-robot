"""UMI-style demonstration recorder: writes episodes as if they came from a UMI hand-held gripper.

What UMI records (Chi et al. 2024, arXiv:2402.10329) and how we mimic it:
  * wrist GoPro, 155 deg fisheye, policy images 224x224 at 10 Hz  -> `wrist_rgb` (rendered
    pinhole at 155 deg, remapped to an equidistant fisheye; no side mirrors)
  * absolute 6-DoF pose from visual-inertial SLAM, 60 Hz, world (map) frame -> `ee_pos/ee_quat`
    (exact sim pose: no SLAM drift)
  * gripper state                                                -> `attached` (exact: perfect
    gripper tracking; suction is binary, so there is no width)
Extra streams the task asked for: `panel_pos_in_target/panel_quat_in_target` (60 Hz) and an
external camera `external_rgb` (10 Hz, for the CV stretch). No actions or rewards are stored;
UMI does not have them. Sim-only metadata (seed, noise, randomization) goes to a sidecar file.

Single .npz, one group per episode (key = "<group>/<name>"):

  meta/schema_version          str
  meta/n_episodes              int
  meta/pose_hz, meta/image_hz  float (60, 10)
  meta/fisheye_fov_deg         float (155)
  episode_XXX/timestamp            (T,)        float64  s, session clock (episodes back to back)
  episode_XXX/ee_pos               (T, 3)      float32  m, world frame (end-effector = suction face)
  episode_XXX/ee_quat              (T, 4)      float32  wxyz, world frame
  episode_XXX/attached             (T,)        uint8    1 = suction holding the panel
  episode_XXX/panel_pos_in_target  (T, 3)      float32  m, panel pose in the target frame
  episode_XXX/panel_quat_in_target (T, 4)      float32  wxyz
  episode_XXX/image_timestamp      (K,)        float64  s, image k = pose sample 6k
  episode_XXX/wrist_rgb            (K, H, W, 3) uint8   fisheye, black outside the image circle
  episode_XXX/external_rgb         (K, H, W, 3) uint8   fixed overview camera
  episode_XXX/start_time           ()          float64  = timestamp[0]
  episode_XXX/end_time             ()          float64  = timestamp[-1]
  episode_XXX/success              ()          bool

Camera calibration (OpenCV conventions: camera x right, y down, z forward; pixel centres at
integer coordinates; T_a_b is the 4x4 pose of frame b in frame a, i.e. maps b-points into a):
  meta/wrist_model                 str      "opencv_fisheye" (Kannala-Brandt; D = 0 -> equidistant)
  meta/wrist_K, meta/wrist_D       (3,3), (4,)
  meta/wrist_image_size            (2,)     [width, height]
  meta/T_ee_wrist                  (4,4)    wrist camera in the EE frame (fixed mount)
  meta/external_model              str      "opencv_pinhole" (no distortion)
  meta/external_K                  (3,3)
  meta/external_image_size         (2,)
  meta/T_world_external            (4,4)    external camera in the world frame (static)
  meta/mirror_sides                (2,)     "left", "right"
  meta/mirror_polygons             (2,6,2)  mirror image regions, normalised (u, v) (UMI's polygons)
  meta/mirror_normals, mirror_points, mirror_virtual_origins  (2,3)  in the wrist camera frame

Side mirrors (schema umi-sim-2): wrist_rgb contains UMI's two side mirrors, rendered as exact
planar reflections (see umi_mirrors.py). Content inside the mirror regions is physically
mirror-flipped, as on the real GoPro; the zarr converter applies UMI's "mirror swap".
"""

from __future__ import annotations

import zipfile

import mujoco
import numpy as np

from PIL import Image, ImageDraw

from .transforms import pose_inv, pose_mul, quat_to_mat
from .umi_mirrors import FISHEYE_FOV_DEG, MIRROR_POLYGONS, SIDES, fisheye_dirs, mirror_geometry, reflect

SCHEMA_VERSION = "umi-sim-2"  # 2: side mirrors in wrist_rgb
POSE_HZ = 60.0
IMAGE_HZ = 10.0


def _homog(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def mirror_mask(size: int, side: str) -> np.ndarray:
    """Boolean (size, size) mask of a UMI mirror region (pixel i covers [i, i+1))."""
    im = Image.new("1", (size, size), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in MIRROR_POLYGONS[side] * size], fill=1)
    return np.array(im, dtype=bool)


def fisheye_maps(out_size: int, fov_deg: float, src_size: int):
    """Index maps that remap a square pinhole image (fovy = fov_deg) to an equidistant fisheye
    (r = f * theta) whose inscribed image circle spans fov_deg. Returns (rows, cols, valid)."""
    half = np.radians(fov_deg / 2)
    f_fish = (out_size / 2) / half
    f_pin = (src_size / 2) / np.tan(half)
    c_out, c_src = (out_size - 1) / 2, (src_size - 1) / 2
    v, u = np.mgrid[0:out_size, 0:out_size].astype(float)
    dx, dy = u - c_out, v - c_out
    r = np.hypot(dx, dy)
    theta = r / f_fish
    valid = theta <= half
    rp = f_pin * np.tan(np.minimum(theta, half))
    scale = np.divide(rp, r, out=np.zeros_like(r), where=r > 0)
    cols = np.clip(np.rint(c_src + dx * scale), 0, src_size - 1).astype(np.intp)
    rows = np.clip(np.rint(c_src + dy * scale), 0, src_size - 1).astype(np.intp)
    return rows, cols, valid


class UMIRecorder:
    """Samples a DrywallPlacementEnv at 60 Hz (poses) and 10 Hz (images).

    Usage per episode: start(t0) right after env.reset(); then env.step() as usual (the
    substep hook samples poses); call on_control_step() after every env.step(); finish().
    """

    def __init__(self, env, image_size: int = 224, pinhole_size: int = 768, supersample: int = 2,
                 mirror_size: int = 320):
        self.env = env
        self.image_size = image_size
        self.ss = supersample
        m = env.model
        if env.cfg.substeps % 6:
            raise ValueError("substeps per control step must be a multiple of 6 for 60 Hz sampling")
        self.sample_every = env.cfg.substeps // 6  # 10 Hz control -> 60 Hz poses
        if abs(m.opt.timestep * self.sample_every - 1.0 / POSE_HZ) > 1e-9:
            raise ValueError("physics timestep * sample_every must equal 1/60 s")
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, pinhole_size)
        m.vis.global_.offheight = max(m.vis.global_.offheight, pinhole_size)
        self._sizes = (pinhole_size, mirror_size)
        self._wrist_r = mujoco.Renderer(m, height=pinhole_size, width=pinhole_size)
        self._mirror_r = mujoco.Renderer(m, height=mirror_size, width=mirror_size)
        self._ext_r = mujoco.Renderer(m, height=image_size, width=image_size)
        self._seg_r = None  # lazily created segmentation renderers (tests)
        self._wrist_cam = m.camera("panda/wrist").id
        N = image_size * supersample
        self._maps = fisheye_maps(N, FISHEYE_FOV_DEG, pinhole_size)
        # Side mirrors: for every fisheye pixel inside a mirror region, reflect its ray off the
        # mirror plane and look it up in the virtual (reflected) camera's pinhole render.
        self._mirrors = []
        for side in SIDES:
            g = mirror_geometry(side)
            cam = m.camera(f"panda/wrist_mirror_{side}").id
            assert abs(m.cam_fovy[cam] - g["fovy_deg"]) < 1e-6
            mask = mirror_mask(N, side) & self._maps[2]
            idx = np.flatnonzero(mask)
            rows, cols = np.divmod(idx, N)
            q = reflect(fisheye_dirs(np.column_stack([(cols + 0.5) / N, (rows + 0.5) / N])), g["normal"]) @ g["R"]
            f = (mirror_size / 2) / np.tan(np.radians(g["fovy_deg"] / 2))
            c = (mirror_size - 1) / 2
            mr = np.rint(c + f * q[:, 1] / q[:, 2]).astype(np.intp)
            mc = np.rint(c + f * q[:, 0] / q[:, 2]).astype(np.intp)
            assert mr.min() >= 0 and mc.min() >= 0 and max(mr.max(), mc.max()) < mirror_size
            self._mirrors.append((cam, idx, mr, mc))
        # Same overview viewpoint as env.render() (behind-left of the robot, looking at the wall).
        self._ext_cam = mujoco.MjvCamera()
        if getattr(env.cfg, "challenge", "easy") == "hard":
            # Hard challenge: wall and rack can be anywhere around the robot, so the static external
            # camera is mounted high above the cell looking down (ceiling RGB-D), covering the ring.
            self._ext_cam.lookat[:] = [0.0, 0.0, 0.3]
            self._ext_cam.distance, self._ext_cam.azimuth, self._ext_cam.elevation = 3.0, 180.0, -72.0
        else:
            self._ext_cam.lookat[:] = [0.45, -0.15, 0.5]
            self._ext_cam.distance, self._ext_cam.azimuth, self._ext_cam.elevation = 2.2, 35.0, -25.0

    # ------------------------------------------------------------------ calibration
    def calibration(self) -> dict:
        """Intrinsics/extrinsics of both cameras in OpenCV conventions (see module docstring)."""
        env, n = self.env, self.image_size
        m, d = env.model, env.data
        mujoco.mj_forward(m, d)
        flip = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera (looks -z, y up) -> OpenCV (z fwd, y down)
        c = (n - 1) / 2

        # Wrist: equidistant fisheye with the same f and centre as fisheye_maps().
        f_w = (n / 2) / np.radians(FISHEYE_FOV_DEG / 2)
        ee_p, ee_q = env._ee_pose_world()
        T_w_ee = _homog(quat_to_mat(ee_q), ee_p)
        T_w_cam = _homog(d.cam_xmat[self._wrist_cam].reshape(3, 3) @ flip, d.cam_xpos[self._wrist_cam])

        # External: free camera; read the pose and frustum MuJoCo actually renders with.
        self._ext_r.update_scene(d, camera=self._ext_cam)
        # scene.camera[0/1] are the two stereo eyes (offset by +/- ipd/2); mono rendering uses
        # their average.
        gl, gl1 = self._ext_r.scene.camera[0], self._ext_r.scene.camera[1]
        fwd, up = np.array(gl.forward), np.array(gl.up)
        eye = (np.array(gl.pos) + np.array(gl1.pos)) / 2
        R_ext = np.column_stack([np.cross(fwd, up), -up, fwd])  # x right, y down, z forward
        f_e = (n / 2) * gl.frustum_near / gl.frustum_top
        return {
            "wrist_model": np.str_("opencv_fisheye"),
            "wrist_K": np.array([[f_w, 0, c], [0, f_w, c], [0, 0, 1.0]]),
            "wrist_D": np.zeros(4),
            "wrist_image_size": np.array([n, n]),
            "T_ee_wrist": np.linalg.inv(T_w_ee) @ T_w_cam,
            "external_model": np.str_("opencv_pinhole"),
            "external_K": np.array([[f_e, 0, c], [0, f_e, c], [0, 0, 1.0]]),
            "external_image_size": np.array([n, n]),
            "T_world_external": _homog(R_ext, eye),
            # Side mirrors (wrist camera OpenCV frame, metres): plane through point with normal;
            # virtual camera centre = lens reflected across the plane. Polygons: normalised (u, v).
            "mirror_sides": np.array(SIDES),
            "mirror_polygons": np.stack([MIRROR_POLYGONS[s] for s in SIDES]),
            "mirror_normals": np.stack([mirror_geometry(s)["normal"] for s in SIDES]),
            "mirror_points": np.stack([mirror_geometry(s)["point"] for s in SIDES]),
            "mirror_virtual_origins": np.stack([mirror_geometry(s)["origin"] for s in SIDES]),
        }

    # ------------------------------------------------------------------ sampling
    def _sample_pose(self):
        env = self.env
        ee_p, ee_q = env._ee_pose_world()
        tp, tq = env._target_pose_world()
        pp, pq = env._body_pose(env._panel_body)
        ip, iq = pose_inv(tp, tq)
        rel_p, rel_q = pose_mul(ip, iq, pp, pq)
        self._buf["timestamp"].append(self._t0 + env.data.time)
        self._buf["ee_pos"].append(ee_p)
        self._buf["ee_quat"].append(ee_q)
        self._buf["attached"].append(env.attached)
        self._buf["panel_pos_in_target"].append(rel_p)
        self._buf["panel_quat_in_target"].append(rel_q)

    def _wrist_image(self, segmentation: bool = False):
        """Fisheye wrist image with both side mirrors composited in. segmentation=True returns
        per-pixel geom ids (-1 = none) at the supersampled resolution instead (for tests)."""
        d = self.env.data
        if segmentation:
            if self._seg_r is None:
                self._seg_r = [mujoco.Renderer(self.env.model, height=s, width=s) for s in self._sizes]
                for r in self._seg_r:
                    r.enable_segmentation_rendering()
            main_r, mirror_r = self._seg_r
        else:
            main_r, mirror_r = self._wrist_r, self._mirror_r
        main_r.update_scene(d, camera=self._wrist_cam)
        pin = main_r.render()
        pin = pin[..., :1].astype(np.int32) if segmentation else pin
        rows, cols, valid = self._maps
        img = np.where(valid[..., None], pin[rows, cols], -1 if segmentation else 0).astype(pin.dtype)
        flat = img.reshape(-1, img.shape[-1])
        for cam, idx, mr, mc in self._mirrors:
            mirror_r.update_scene(d, camera=cam)
            mp = mirror_r.render()
            mp = mp[..., :1].astype(np.int32) if segmentation else mp
            flat[idx] = mp[mr, mc]
        if segmentation:
            return img[..., 0]
        if self.ss > 1:  # box-filter the supersampled fisheye down to the output size
            s, n = self.ss, self.image_size
            img = img.reshape(n, s, n, s, 3).mean(axis=(1, 3)).round().astype(np.uint8)
        return img

    def _sample_images(self):
        self._ext_r.update_scene(self.env.data, camera=self._ext_cam)
        self._buf["image_timestamp"].append(self._t0 + self.env.data.time)
        self._buf["wrist_rgb"].append(self._wrist_image())
        self._buf["external_rgb"].append(self._ext_r.render().copy())

    def _hook(self, env, i):
        if (i + 1) % self.sample_every == 0:
            self._sample_pose()

    # ------------------------------------------------------------------ episode API
    def start(self, t0: float):
        """Call right after env.reset(); t0 = session-clock time of the first sample."""
        self._t0 = t0
        self._buf = {k: [] for k in ("timestamp", "ee_pos", "ee_quat", "attached", "panel_pos_in_target",
                                     "panel_quat_in_target", "image_timestamp", "wrist_rgb", "external_rgb")}
        self._sample_pose()
        self._sample_images()
        self.env.substep_hook = self._hook

    def on_control_step(self):
        self._sample_images()

    def finish(self, success: bool) -> dict:
        self.env.substep_hook = None
        b = self._buf
        f32 = lambda k: np.asarray(b[k], dtype=np.float32)  # noqa: E731
        ts = np.asarray(b["timestamp"], dtype=np.float64)
        return {
            "timestamp": ts,
            "ee_pos": f32("ee_pos"),
            "ee_quat": f32("ee_quat"),
            "attached": np.asarray(b["attached"], dtype=np.uint8),
            "panel_pos_in_target": f32("panel_pos_in_target"),
            "panel_quat_in_target": f32("panel_quat_in_target"),
            "image_timestamp": np.asarray(b["image_timestamp"], dtype=np.float64),
            "wrist_rgb": np.stack(b["wrist_rgb"]),
            "external_rgb": np.stack(b["external_rgb"]),
            "start_time": np.float64(ts[0]),
            "end_time": np.float64(ts[-1]),
            "success": np.bool_(success),
        }

    def close(self):
        self.env.substep_hook = None
        for r in [self._wrist_r, self._mirror_r, self._ext_r] + (self._seg_r or []):
            r.close()


class NpzStreamWriter:
    """Writes arrays one at a time into a compressed .npz (np.load-compatible), so the whole
    dataset never has to sit in memory."""

    def __init__(self, path, mode: str = "w"):
        self.zf = zipfile.ZipFile(path, mode, compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    def put(self, name: str, arr):
        if name + ".npy" in self.zf.namelist():
            raise KeyError(f"{name} already in the archive")
        with self.zf.open(name + ".npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)

    def put_group(self, group: str, arrays: dict):
        for k, v in arrays.items():
            self.put(f"{group}/{k}", v)

    def bytes_written(self) -> int:
        return sum(i.compress_size for i in self.zf.infolist())

    def close(self):
        self.zf.close()
