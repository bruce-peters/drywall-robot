"""Builds the MuJoCo model: work-cell scene + Menagerie Panda (no hand) + suction plate."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .umi_mirrors import SIDES, mirror_geometry

ASSETS = Path(__file__).parent / "assets"
SCENE_XML = ASSETS / "scene.xml"

# Suction / panel-handling plate mounted on the Panda flange ("attachment" body).
PLATE_HALF = (0.12, 0.09, 0.005)  # 24 x 18 x 1 cm
PLATE_MASS = 0.5
# The end-effector frame ("ee" site) sits at the centre of the plate's outer face,
# same orientation as the flange: z points out of the plate (towards the panel).
EE_OFFSET = 2 * PLATE_HALF[2]
# Wrist camera: 5 cm past the panel's long edge (panel half-height 0.2 m along plate y) and
# 5 cm behind the plate face. On the +y edge (below the panel at the wall, so it sees the ledge):
# the -y edge went 5 cm into the wall at the home pose. Clearance to wall/rack/ledge/floor
# >= 2.6 cm over test episodes. Rendered as a 155 deg pinhole, then remapped to a fisheye.
WRIST_CAM_POS = (0.0, 0.25, EE_OFFSET - 0.05)
WRIST_CAM_FOVY = 155.0

ARM_JOINTS = [f"panda/joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"panda/actuator{i}" for i in range(1, 8)]
HOME_QPOS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]


def panda_xml_path() -> Path:
    """Path to the Menagerie Panda without hand (fetched/pinned by robot_descriptions)."""
    from robot_descriptions import panda_mj_description

    return Path(panda_mj_description.PACKAGE_PATH) / "panda_nohand.xml"


def build_spec() -> mujoco.MjSpec:
    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    panda = mujoco.MjSpec.from_file(str(panda_xml_path()))

    # Drop the Panda's own light/keyframe; the scene provides lights.
    for light in list(panda.lights):
        panda.delete(light)
    for key in list(panda.keys):
        panda.delete(key)

    # Gravity compensation on all arm links (mirrors the real Panda's controller).
    for body in panda.bodies:
        if body.name != "world":
            body.gravcomp = 1.0

    flange = panda.body("attachment")
    plate = flange.add_body(name="plate", gravcomp=1.0)
    plate.add_geom(
        name="plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(PLATE_HALF),
        pos=[0, 0, PLATE_HALF[2]],
        mass=PLATE_MASS,
        rgba=[0.15, 0.35, 0.75, 1],
    )
    plate.add_site(name="ee", pos=[0, 0, EE_OFFSET], size=[0.01, 0, 0])
    # UMI-style wrist camera (no mass, no collision). Mounted past the held panel's edge so the
    # panel does not fill the view: WRIST_CAM_POS is in the plate frame. The mast is not modelled. It looks along the
    # approach direction (+z of the plate): a MuJoCo camera looks along its own -z, hence 180 deg
    # about x, which also makes image-up = plate -y (world up when the panel is at the wall).
    # The pinhole fovy here is what we render; scripts/record_umi.py remaps it to a 155 deg fisheye.
    plate.add_camera(name="wrist", pos=list(WRIST_CAM_POS), quat=[0, 1, 0, 0], fovy=WRIST_CAM_FOVY)
    # UMI side mirrors = virtual cameras (the lens reflected across each mirror plane), rigid on the
    # plate like the real mirrors; see umi_mirrors.py. With quat [0, 1, 0, 0] the wrist camera's
    # OpenCV frame (x right, y down, z forward) has the same axes as the plate frame.
    for side in SIDES:
        g = mirror_geometry(side)
        R_mj = g["R"] @ np.diag([1.0, -1.0, -1.0])  # OpenCV axes -> MuJoCo camera axes
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_mj.reshape(9))
        plate.add_camera(name=f"wrist_mirror_{side}", pos=list(np.array(WRIST_CAM_POS) + g["origin"]),
                         quat=list(q), fovy=g["fovy_deg"])

    scene.attach(panda, prefix="panda/", frame=scene.frame("robot_base"))
    return scene


def build_model() -> mujoco.MjModel:
    return build_spec().compile()
