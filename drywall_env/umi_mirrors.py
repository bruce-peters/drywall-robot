"""UMI side mirrors as planar optics (pure numpy; shared by model.py and the recorder).

UMI puts two flat stick-on mirrors in the periphery of the GoPro's 155 deg fisheye view. A flat
mirror is equivalent to a virtual camera reflected across the mirror plane (paper HD3, Fig. 4), so
the mirror regions of the image show the scene from two extra viewpoints: implicit stereo.

Image regions come from UMI's code (umi/common/cv_util.py, get_mirror_canonical_polygon): pixel
polygons on the 2704 x 2028 GoPro frame, the right one being the left one mirrored in x. UMI resizes
that frame to 224 x 224, so we use the same normalised (u, v) coordinates on our square image.
The 3D mirror placement is in UMI's Onshape CAD, which we do not have, so it is chosen here:
  * each mirror's centre lies on the ray through its region's centroid, MIRROR_DIST from the lens;
  * its normal is chosen so that ray reflects to a forward view tilted MIRROR_INWARD_DEG inward.
Every pixel of a mirror region is then rendered exactly: its ray is reflected off the plane and
the scene is sampled along the reflected ray from the virtual camera centre (the lens reflected
across the plane). Mirror content is therefore physically mirror-flipped, as on the real GoPro.
All vectors are in the wrist camera's OpenCV frame (x right, y down, z forward).
"""

from __future__ import annotations

import numpy as np

FISHEYE_FOV_DEG = 155.0
MIRROR_DIST = 0.06  # m, lens -> mirror centre
MIRROR_INWARD_DEG = 15.0  # reflected central ray: forward, tilted this much towards the image centre
_UMI_W, _UMI_H = 2704, 2028
_UMI_LEFT_PX = [(540, 1700), (680, 1450), (590, 1070), (290, 1130), (290, 1770), (550, 1770)]
MIRROR_POLYGONS = {  # normalised image coords (u right, v down, in [0, 1])
    "left": np.array([(x / _UMI_W, y / _UMI_H) for x, y in _UMI_LEFT_PX]),
    "right": np.array([(1 - x / _UMI_W, y / _UMI_H) for x, y in _UMI_LEFT_PX]),
}
SIDES = ("left", "right")


def fisheye_dirs(uv: np.ndarray, fov_deg: float = FISHEYE_FOV_DEG) -> np.ndarray:
    """Normalised image coords (N, 2) -> unit ray directions (N, 3) for the equidistant fisheye
    used by the recorder (image circle inscribed in the square image, spanning fov_deg)."""
    d = (np.asarray(uv, float).reshape(-1, 2) - 0.5) * 2  # in units of the image half-size
    th = np.hypot(d[:, 0], d[:, 1]) * np.radians(fov_deg / 2)
    ph = np.arctan2(d[:, 1], d[:, 0])
    return np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v - 2 * (v @ n)[..., None] * n


def _polygon_samples(poly: np.ndarray, per_edge: int = 20) -> np.ndarray:
    nxt = np.roll(poly, -1, axis=0)
    t = np.linspace(0, 1, per_edge, endpoint=False)[:, None, None]
    return (poly + t * (nxt - poly)).reshape(-1, 2)


def mirror_geometry(side: str) -> dict:
    """Plane, virtual camera pose and the pinhole fovy needed to cover the mirror region."""
    poly = MIRROR_POLYGONS[side]
    u0 = fisheye_dirs(poly).mean(axis=0)
    u0 /= np.linalg.norm(u0)
    a = np.radians(MIRROR_INWARD_DEG)
    r0 = np.array([-np.sign(u0[0]) * np.sin(a), 0.0, np.cos(a)])  # inward = towards the centre
    n = (u0 - r0) / np.linalg.norm(u0 - r0)  # reflect(u0, n) == r0
    point = MIRROR_DIST * u0
    origin = 2 * (point @ n) * n  # lens (at 0) reflected across the plane
    # Virtual camera axes (OpenCV): z = central reflected ray, y as close to image-down as possible.
    y = np.array([0.0, 1.0, 0.0]) - r0[1] * r0
    y /= np.linalg.norm(y)
    R = np.column_stack([np.cross(y, r0), y, r0])
    # Pinhole fovy covering every reflected ray of the region (+ margin).
    q = reflect(fisheye_dirs(_polygon_samples(poly)), n) @ R
    assert np.all(q[:, 2] > 0.2), "mirror region not in front of its virtual camera"
    half = np.arctan(np.max(np.abs(q[:, :2] / q[:, 2:3])))
    # Every region ray must actually hit the mirror plane in front of the lens.
    assert np.all(fisheye_dirs(_polygon_samples(poly)) @ n > 0)
    return {"normal": n, "point": point, "origin": origin, "R": R,
            "fovy_deg": float(np.degrees(2 * half) + 4.0)}
