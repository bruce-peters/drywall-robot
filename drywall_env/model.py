"""Builds the MuJoCo model: work-cell scene + Menagerie Panda (no hand) + suction plate."""

from __future__ import annotations

from pathlib import Path

import mujoco

ASSETS = Path(__file__).parent / "assets"
SCENE_XML = ASSETS / "scene.xml"

# Suction / panel-handling plate mounted on the Panda flange ("attachment" body).
PLATE_HALF = (0.12, 0.09, 0.005)  # 24 x 18 x 1 cm
PLATE_MASS = 0.5
# The end-effector frame ("ee" site) sits at the centre of the plate's outer face,
# same orientation as the flange: z points out of the plate (towards the panel).
EE_OFFSET = 2 * PLATE_HALF[2]

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

    scene.attach(panda, prefix="panda/", frame=scene.frame("robot_base"))
    return scene


def build_model() -> mujoco.MjModel:
    return build_spec().compile()
