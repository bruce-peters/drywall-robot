"""Gymnasium-style MuJoCo environment: pick a drywall panel off a rack and place it on a wall."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .ik import DLSIK
from .model import ARM_ACTUATORS, ARM_JOINTS, HOME_QPOS, build_model
from .transforms import (
    canonical,
    mat_to_quat,
    pose_inv,
    pose_mul,
    quat_angle,
    quat_conj,
    quat_mul,
    quat_to_mat,
    rotvec_to_quat,
    rpy_to_mat,
)

PANEL_HALF = np.array([0.3, 0.2, 0.005])


@dataclass
class EnvConfig:
    # Which task: "easy" = Parts A-C (fixed wall in front, rack at (0.35, -0.5), small offsets).
    # "hard" = wall AND rack placed anywhere around the robot every episode (see the hard_* fields;
    # ranges from the IK reachability sweep, LABNOTEBOOK 2026-11-18).
    challenge: str = "easy"
    # hard only: "full" (whole distribution), "demo" (the limited region the UMI demos cover) or
    # "ood" (full distribution minus the demo region)
    region: str = "full"
    # Timing
    substeps: int = 60  # physics dt 1/600 s (scene.xml) -> 10 Hz control; 60 Hz = every 10 substeps
    max_steps: int | None = None  # None -> 300 (easy) / 500 (hard: long swings around the base)
    # Action scaling (per control step, robot base frame)
    max_pos_step: float = 0.02  # m
    max_rot_step: float = np.radians(3.0)  # rad, per roll/pitch/yaw component
    # Commanded EE position is clipped to this box (base frame), around reachable space.
    # None -> easy: (0.10, -0.75, 0.15)-(0.80, 0.60, 1.00); hard: +-0.85 m in x and y.
    workspace_low: tuple | None = None
    workspace_high: tuple | None = None
    # Hard challenge (angles in rad, about the robot's base z axis; 0 = the easy wall direction)
    hard_wall_angle: tuple = (np.radians(-150.0), np.radians(150.0))  # joint1 is +-166 deg
    hard_demo_wall_angle: tuple = (np.radians(-45.0), np.radians(45.0))
    hard_rack_radius: tuple = (0.60, 0.70)  # m; closer racks behind the robot are unreachable
    hard_rack_gap: tuple = (np.radians(60.0), np.radians(150.0))  # |rack angle - wall angle|
    hard_demo_rack_gap: tuple = (np.radians(-120.0), np.radians(-60.0))  # demo: rack on the wall's right
    hard_rack_angle_limit: float = np.radians(160.0)  # |rack angle| (joint1 limit)
    # panel yaw relative to the rack's radial direction; +15..+75 deg is a wrist-limit band (sweep)
    hard_panel_rel_yaw: tuple = (np.radians(-135.0), np.radians(-15.0))
    hard_target_dy: tuple = (-0.20, 0.20)  # m along the wall
    hard_target_z: tuple = (0.55, 0.75)  # m, target centre height (wall-facing reach band)

    def __post_init__(self):
        hard = self.challenge == "hard"
        if self.challenge not in ("easy", "hard") or self.region not in ("full", "demo", "ood"):
            raise ValueError(f"bad challenge/region {self.challenge}/{self.region}")
        if self.max_steps is None:
            self.max_steps = 500 if hard else 300
        if self.workspace_low is None:
            self.workspace_low = (-0.85, -0.85, 0.15) if hard else (0.10, -0.75, 0.15)
        if self.workspace_high is None:
            self.workspace_high = (0.85, 0.85, 1.00) if hard else (0.80, 0.60, 1.00)
        if hard:
            self.reward_weights = {**self.reward_weights, "success": self.hard_success_weight}
    # Suction attachment: plate face within attach_dist of the panel face, plate centre over
    # the panel, plate normal within attach_max_angle of the panel normal.
    attach_dist: float = 0.03
    attach_max_angle: float = np.radians(30.0)
    # Observation noise on the panel / target pose estimates (deployable group)
    pos_noise_std: float = 0.0  # m
    rot_noise_std: float = 0.0  # rad
    # Randomization (all sampled from the reset seed)
    randomize: bool = True
    panel_pos_range: float = 0.05  # +/- m, in rack plane (x, y)
    panel_yaw_range: float = np.radians(10.0)  # +/- rad
    target_offset_range: float = 0.05  # +/- m, in wall plane (y, z)
    panel_mass_range: tuple = (1.0, 3.0)  # kg
    panel_friction_range: tuple = (0.5, 1.0)  # sliding friction coefficient
    # Success
    success_pos_tol: float = 0.020  # m
    success_ori_tol: float = np.radians(5.0)  # rad
    success_lin_vel: float = 0.01  # m/s
    success_ang_vel: float = 0.05  # rad/s
    success_steps: int = 5
    # Panel counts as "in contact with the wall" within this distance (absorbs sim contact
    # compliance / spring-back on release). Same tolerance engages the ledge fixture.
    wall_contact_tol: float = 0.002  # m
    # Events
    force_threshold: float = 50.0  # N, panel-wall normal force
    collision_inplane_margin: float = 0.05  # panel-wall contact this far from target = collision
    drop_unsupported_steps: int = 3
    # Reward weights
    reward_weights: dict = field(
        default_factory=lambda: {
            "pos_error": 1.0,  # per m
            "ori_error": 0.3,  # per rad
            "collision": 1.0,  # per step with a collision
            "contact_force": 1.0,  # per (F - threshold) / threshold
            "action": 0.01,  # per ||a||^2 (motion dims)
            "jerk": 0.01,  # per ||a - a_prev||^2 (motion dims)
            "drop": 50.0,
            "success": 100.0,  # easy challenge (Parts B/C as reported); hard: hard_success_weight
        }
    )
    # Hard challenge success weight (user, 2026-11-18): x10. With +100 the discounted success
    # (~100 * 0.99^150 = 22) was worth only ~70 steps of 0.3 m error, and IQL learned to keep holding
    # the panel instead of releasing it (held_at_end = 9-10 of 20 validation episodes).
    hard_success_weight: float = 1000.0


class DrywallPlacementEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None,
                 render_size: tuple[int, int] = (480, 640)):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.render_size = render_size
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.dt = m.opt.timestep * self.cfg.substeps
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.dt)))

        # --- ids
        self._qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self._dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self._act = np.array([m.actuator(a).id for a in ARM_ACTUATORS])
        self._ctrl_lo = m.actuator_ctrlrange[self._act, 0]
        self._ctrl_hi = m.actuator_ctrlrange[self._act, 1]
        self._ee_site = m.site("panda/ee").id
        self._base_body = m.body("panda/link0").id
        self._plate_body = m.body("panda/plate").id
        self._panel_body = m.body("panel").id
        self._wall_body = m.body("wall").id
        self._target_body = m.body("target").id
        self._target_site = m.site("target").id
        self._panel_geom = m.geom("panel").id
        self._panel_qadr = m.joint("panel").qposadr[0]
        self._panel_dadr = m.joint("panel").dofadr[0]
        self._eq_grasp = m.equality("grasp").id
        self._eq_fixture = m.equality("fixture").id
        self._g_wall = m.geom("wall").id
        self._g_ledge = m.geom("ledge").id
        self._g_rack = m.geom("rack").id
        self._g_floor = m.geom("floor").id
        self._robot_geoms = {
            g for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name.startswith("panda/") and m.geom_contype[g] | m.geom_conaffinity[g]
        }
        self._plate_geom = m.geom("panda/plate").id
        self._env_geoms = {self._g_wall, self._g_ledge, self._g_rack, self._g_floor}

        # --- nominal values (restored at every reset so randomization never accumulates)
        self._nom_target_pos = m.body_pos[self._target_body].copy()
        self._nom_target_quat = m.body_quat[self._target_body].copy()
        self._rack_body = m.body("rack").id
        self._nom_wall_pos = m.body_pos[self._wall_body].copy()
        self._nom_wall_quat = m.body_quat[self._wall_body].copy()
        self._nom_rack_pos = m.body_pos[self._rack_body].copy()
        self._nom_rack_quat = m.body_quat[self._rack_body].copy()
        self._nom_panel_mass = float(m.body_mass[self._panel_body])
        self._nom_panel_inertia = m.body_inertia[self._panel_body].copy()
        self._nom_panel_friction = m.geom_friction[self._panel_geom].copy()
        self._nom_panel_qpos = m.qpos0[self._panel_qadr:self._panel_qadr + 7].copy()

        self.ik = DLSIK(m, "panda/ee", ARM_JOINTS, np.array(HOME_QPOS))

        # --- spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        inf = np.inf
        box = lambda n: spaces.Box(-inf, inf, shape=(n,), dtype=np.float32)  # noqa: E731
        self._deployable_keys = {
            "joint_pos": 7, "joint_vel": 7, "ee_pos": 3, "ee_quat": 4, "attached": 1,
            "panel_pos_est": 3, "panel_quat_est": 4, "target_pos_est": 3, "target_quat_est": 4,
        }
        self._privileged_keys = {
            "panel_pos": 3, "panel_quat": 4, "panel_linvel": 3, "panel_angvel": 3,
            "target_pos": 3, "target_quat": 4,
            "panel_wall_force": 1, "panel_ledge_force": 1, "panel_rack_force": 1,
            "robot_env_force": 1,
        }
        policy_dim = sum(self._deployable_keys.values()) + 7
        self.observation_space = spaces.Dict({
            "deployable": spaces.Dict({k: box(n) for k, n in self._deployable_keys.items()}),
            "privileged": spaces.Dict({k: box(n) for k, n in self._privileged_keys.items()}),
            "policy": box(policy_dim),
        })

        self._renderer = None
        self._cams = None
        # Optional callable(env, substep_index) run after every physics substep (used by the
        # 60 Hz UMI-style recorder). Read-only observers only: must not modify model/data.
        self.substep_hook = None

    # ------------------------------------------------------------------ helpers
    def _base_pose(self):
        d = self.data
        return d.xpos[self._base_body].copy(), d.xquat[self._base_body].copy()

    def _to_base(self, pos, quat):
        """World pose -> robot base frame."""
        bp, bq = self._base_pose()
        ip, iq = pose_inv(bp, bq)
        return pose_mul(ip, iq, pos, quat)

    def _to_world(self, pos, quat):
        bp, bq = self._base_pose()
        return pose_mul(bp, bq, pos, quat)

    def _body_pose(self, bid):
        return self.data.xpos[bid].copy(), canonical(self.data.xquat[bid].copy())

    def _ee_pose_world(self):
        d = self.data
        return d.site_xpos[self._ee_site].copy(), mat_to_quat(d.site_xmat[self._ee_site])

    def _target_pose_world(self):
        d = self.data
        return d.site_xpos[self._target_site].copy(), mat_to_quat(d.site_xmat[self._target_site])

    def _relpose(self, b1, b2):
        """Pose of body2 in body1's frame (MuJoCo weld relpose convention)."""
        p1, q1 = self._body_pose(b1)
        p2, q2 = self._body_pose(b2)
        ip, iq = pose_inv(p1, q1)
        return pose_mul(ip, iq, p2, q2)

    def _activate_weld(self, eq_id, b1, b2):
        pos, quat = self._relpose(b1, b2)
        self.model.eq_data[eq_id, 0:3] = 0.0
        self.model.eq_data[eq_id, 3:6] = pos
        self.model.eq_data[eq_id, 6:10] = quat
        self.data.eq_active[eq_id] = 1

    @property
    def attached(self) -> bool:
        return bool(self.data.eq_active[self._eq_grasp])

    @property
    def fixture_active(self) -> bool:
        return bool(self.data.eq_active[self._eq_fixture])

    def _can_attach(self) -> bool:
        ee_p, ee_q = self._ee_pose_world()
        pp, pq = self._body_pose(self._panel_body)
        Rp = quat_to_mat(pq)
        local = Rp.T @ (ee_p - pp)
        if abs(local[0]) > PANEL_HALF[0] or abs(local[1]) > PANEL_HALF[1]:
            return False
        gap = abs(local[2]) - PANEL_HALF[2]
        if gap > self.cfg.attach_dist or gap < -0.005:
            return False
        # plate normal (ee z) must point into the panel face
        face_normal = Rp[:, 2] * np.sign(local[2])  # outward normal of the face near the plate
        ee_z = quat_to_mat(ee_q)[:, 2]
        return float(np.dot(ee_z, -face_normal)) > np.cos(self.cfg.attach_max_angle)

    def _panel_wall_dist(self) -> float:
        """Signed panel-wall distance (m, negative = penetration), capped at 5 cm."""
        return float(mujoco.mj_geomDistance(self.model, self.data, self._panel_geom, self._g_wall, 0.05, None))

    def _contact_metrics(self):
        """Per-substep contact summary from the current contact list."""
        m, d = self.model, self.data
        f6 = np.zeros(6)
        out = {"wall": 0.0, "ledge": 0.0, "rack": 0.0, "floor": False, "robot_env": 0.0,
               "collision": False, "panel_wall_contact": False}
        pp, _ = self._body_pose(self._panel_body)
        tp, tq = self._target_pose_world()
        inplane = (quat_to_mat(tq).T @ (pp - tp))[:2]
        far_from_target = np.linalg.norm(inplane) > self.cfg.collision_inplane_margin
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(f6[0])
            pair = {g1, g2}
            if self._panel_geom in pair:
                other = g2 if g1 == self._panel_geom else g1
                if other == self._g_wall:
                    out["wall"] += fn
                    out["panel_wall_contact"] = True
                    if far_from_target:
                        out["collision"] = True
                elif other == self._g_ledge:
                    out["ledge"] += fn
                elif other == self._g_rack:
                    out["rack"] += fn
                elif other == self._g_floor:
                    out["floor"] = True
                elif other in self._robot_geoms and other != self._plate_geom:
                    out["collision"] = True
            elif (g1 in self._robot_geoms and g2 in self._env_geoms) or (
                g2 in self._robot_geoms and g1 in self._env_geoms
            ):
                out["robot_env"] += fn
                out["collision"] = True
        return out

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        m, d, cfg = self.model, self.data, self.cfg
        rng = self.np_random

        # Restore nominal model parameters, then sample this episode's factors.
        m.body_pos[self._target_body] = self._nom_target_pos
        m.body_quat[self._target_body] = self._nom_target_quat
        m.body_pos[self._wall_body] = self._nom_wall_pos
        m.body_quat[self._wall_body] = self._nom_wall_quat
        m.body_pos[self._rack_body] = self._nom_rack_pos
        m.body_quat[self._rack_body] = self._nom_rack_quat
        m.body_mass[self._panel_body] = self._nom_panel_mass
        m.body_inertia[self._panel_body] = self._nom_panel_inertia
        m.geom_friction[self._panel_geom] = self._nom_panel_friction
        if cfg.challenge == "hard":
            return self._reset_hard(rng)
        if cfg.randomize:
            r = cfg.panel_pos_range
            rnd = {
                "panel_dx": float(rng.uniform(-r, r)),
                "panel_dy": float(rng.uniform(-r, r)),
                "panel_dyaw": float(rng.uniform(-cfg.panel_yaw_range, cfg.panel_yaw_range)),
                "target_dy": float(rng.uniform(-cfg.target_offset_range, cfg.target_offset_range)),
                "target_dz": float(rng.uniform(-cfg.target_offset_range, cfg.target_offset_range)),
                "panel_mass": float(rng.uniform(*cfg.panel_mass_range)),
                "panel_friction": float(rng.uniform(*cfg.panel_friction_range)),
            }
        else:
            rnd = {"panel_dx": 0.0, "panel_dy": 0.0, "panel_dyaw": 0.0, "target_dy": 0.0,
                   "target_dz": 0.0, "panel_mass": self._nom_panel_mass,
                   "panel_friction": float(self._nom_panel_friction[0])}
        self._rand = rnd

        m.body_pos[self._target_body] = self._nom_target_pos + [0.0, rnd["target_dy"], rnd["target_dz"]]
        m.body_mass[self._panel_body] = rnd["panel_mass"]
        m.body_inertia[self._panel_body] = self._nom_panel_inertia * rnd["panel_mass"] / self._nom_panel_mass
        m.geom_friction[self._panel_geom, 0] = rnd["panel_friction"]

        pq = self._nom_panel_qpos.copy()
        pq[0] += rnd["panel_dx"]
        pq[1] += rnd["panel_dy"]
        pq[3:7] = canonical(quat_mul(rotvec_to_quat([0, 0, rnd["panel_dyaw"]]), pq[3:7]))
        return self._finish_reset(pq)

    def _sample_hard(self, rng) -> dict:
        """Wall angle, rack placement and target offset for the hard challenge. Rejection-samples
        within the configured region; every draw comes from the episode's rng (deterministic)."""
        cfg = self.cfg
        u = lambda r: float(rng.uniform(*r))  # noqa: E731
        lo_d, hi_d = cfg.hard_demo_wall_angle
        lo_g, hi_g = cfg.hard_demo_rack_gap

        def in_demo(theta, gap):
            return lo_d <= theta <= hi_d and lo_g <= gap <= hi_g

        for _ in range(1000):
            if cfg.region == "demo":
                theta, gap = u(cfg.hard_demo_wall_angle), u(cfg.hard_demo_rack_gap)
            else:
                theta = u(cfg.hard_wall_angle)
                gap = u(cfg.hard_rack_gap) * (1.0 if rng.uniform() < 0.5 else -1.0)
            phi = theta + gap
            if abs(phi) > cfg.hard_rack_angle_limit:
                continue
            if cfg.region == "ood" and in_demo(theta, gap):
                continue
            break
        else:
            raise RuntimeError("could not sample a hard-challenge layout")
        return {"wall_angle": theta, "rack_angle": phi, "rack_radius": u(cfg.hard_rack_radius),
                "panel_rel_yaw": u(cfg.hard_panel_rel_yaw),
                "target_dy": u(cfg.hard_target_dy), "target_z": u(cfg.hard_target_z)}

    def _reset_hard(self, rng):
        """Hard challenge: rotate the wall (with its target and ledge) and place the rack anywhere
        around the robot. Wall/target poses are the easy ones rotated by the wall angle about the
        base z axis, so the wall-facing geometry relative to the arm is unchanged."""
        m, cfg = self.model, self.cfg
        lay = self._sample_hard(rng)
        r = cfg.panel_pos_range
        rnd = {**lay,
               "panel_dx": float(rng.uniform(-r, r)), "panel_dy": float(rng.uniform(-r, r)),
               "panel_mass": float(rng.uniform(*cfg.panel_mass_range)),
               "panel_friction": float(rng.uniform(*cfg.panel_friction_range))}
        self._rand = rnd
        qz = rotvec_to_quat([0, 0, lay["wall_angle"]])
        Rz = quat_to_mat(qz)
        m.body_pos[self._wall_body] = Rz @ self._nom_wall_pos
        m.body_quat[self._wall_body] = quat_mul(qz, self._nom_wall_quat)
        t_local = self._nom_target_pos + [0.0, lay["target_dy"], lay["target_z"] - self._nom_target_pos[2]]
        m.body_pos[self._target_body] = Rz @ t_local
        m.body_quat[self._target_body] = quat_mul(qz, self._nom_target_quat)
        # rack: centre on the ring, x axis radial (long side tangential)
        phi, rad = lay["rack_angle"], lay["rack_radius"]
        rack_c = np.array([rad * np.cos(phi), rad * np.sin(phi), self._nom_rack_pos[2]])
        m.body_pos[self._rack_body] = rack_c
        m.body_quat[self._rack_body] = rotvec_to_quat([0, 0, phi])
        m.body_mass[self._panel_body] = rnd["panel_mass"]
        m.body_inertia[self._panel_body] = self._nom_panel_inertia * rnd["panel_mass"] / self._nom_panel_mass
        m.geom_friction[self._panel_geom, 0] = rnd["panel_friction"]
        # panel on the rack: offset in the rack frame, yaw relative to the radial direction
        Rr = quat_to_mat(rotvec_to_quat([0, 0, phi]))
        pq = self._nom_panel_qpos.copy()
        pq[:3] = rack_c + Rr @ np.array([rnd["panel_dx"], rnd["panel_dy"], 0.0])
        pq[2] = self._nom_panel_qpos[2]
        pq[3:7] = rotvec_to_quat([0, 0, phi + lay["panel_rel_yaw"]])
        return self._finish_reset(pq)

    def _finish_reset(self, panel_qpos):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self._panel_qadr:self._panel_qadr + 7] = panel_qpos
        d.qpos[self._qadr] = HOME_QPOS
        d.ctrl[self._act] = HOME_QPOS
        d.eq_active[:] = 0
        mujoco.mj_forward(m, d)

        # Commanded EE pose (base frame) starts at the measured pose.
        self._cmd_pos, self._cmd_quat = self._to_base(*self._ee_pose_world())
        self._q_cmd = np.array(HOME_QPOS, float)
        self._prev_action = np.zeros(7)
        self._steps = 0
        self._stable_count = 0
        self._unsupported_count = 0
        self._last_contacts = self._contact_metrics()

        obs = self._get_obs()
        info = {"randomization": dict(self._rand)}
        return obs, info

    def step(self, action):
        m, d, cfg = self.model, self.data, self.cfg
        raw = np.asarray(action, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"non-finite action: {raw}")
        a = np.clip(raw, -1.0, 1.0)
        action_clipped = bool(np.any(a != raw))

        # 1) Cartesian delta in the robot base frame -> new commanded EE pose.
        new_pos = self._cmd_pos + a[:3] * cfg.max_pos_step
        clipped_pos = np.clip(new_pos, cfg.workspace_low, cfg.workspace_high)
        workspace_clipped = bool(np.any(clipped_pos != new_pos))
        self._cmd_pos = clipped_pos
        drot = rpy_to_mat(*(a[3:6] * cfg.max_rot_step))
        self._cmd_quat = mat_to_quat(drot @ quat_to_mat(self._cmd_quat))

        # 2) Attach / release (suction). Weld uses the CURRENT relative pose: no snapping.
        attach_failed = False
        if a[6] > 0 and not self.attached:
            if self._can_attach():
                self._activate_weld(self._eq_grasp, self._plate_body, self._panel_body)
                d.eq_active[self._eq_fixture] = 0  # fixture lets go when the robot re-grasps
            else:
                attach_failed = True
        elif a[6] <= 0 and self.attached:
            d.eq_active[self._eq_grasp] = 0

        # 3) IK -> joint position targets for the position actuators.
        wp, wq = self._to_world(self._cmd_pos, self._cmd_quat)
        if cfg.challenge == "hard":
            # Posture-aware IK: the null-space home posture turns joint1 towards the commanded EE's
            # azimuth, so the arm keeps the easy task's posture relative to wherever it is working.
            # With joint1 fixed at 0 the redundancy is resolved by twisting joints 3/5/7 until joint6
            # hits its limit near walls away from 0 deg (LABNOTEBOOK 2026-11-18). Easy is unchanged.
            self.ik.q_home[0] = np.clip(np.arctan2(self._cmd_pos[1], self._cmd_pos[0]),
                                        self.ik.lo[0] + 0.05, self.ik.hi[0] - 0.05)
        self._q_cmd, ik_residual = self.ik.solve(self._q_cmd, wp, wq)
        d.ctrl[self._act] = np.clip(self._q_cmd, self._ctrl_lo, self._ctrl_hi)

        # 4) Physics, tracking worst-case contact events over the substeps.
        max_wall_force = 0.0
        collision = False
        floor = False
        for i in range(cfg.substeps):
            mujoco.mj_step(m, d)
            c = self._contact_metrics()
            max_wall_force = max(max_wall_force, c["wall"])
            collision |= c["collision"]
            floor |= c["floor"]
            # External fixture: a released panel resting on the ledge against the wall is held
            # at its current pose.
            if not self.attached and not self.fixture_active and c["ledge"] > 0 and (
                c["panel_wall_contact"] or self._panel_wall_dist() <= cfg.wall_contact_tol
            ):
                self._activate_weld(self._eq_fixture, self._wall_body, self._panel_body)
            if self.substep_hook is not None:
                self.substep_hook(self, i)
        self._last_contacts = c
        self._steps += 1

        # 5) Task metrics.
        pp, pq = self._body_pose(self._panel_body)
        tp, tq = self._target_pose_world()
        pos_err = float(np.linalg.norm(pp - tp))
        ori_err = quat_angle(pq, tq)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self._panel_body, vel, 0)
        ang_speed, lin_speed = float(np.linalg.norm(vel[:3])), float(np.linalg.norm(vel[3:]))
        wall_dist = self._panel_wall_dist()
        on_wall = wall_dist <= cfg.wall_contact_tol

        detached = not self.attached
        supported = self.fixture_active or c["rack"] > 0 or c["ledge"] > 0
        self._unsupported_count = self._unsupported_count + 1 if (detached and not supported) else 0
        dropped = detached and (floor or self._unsupported_count >= cfg.drop_unsupported_steps)

        placed = (
            pos_err <= cfg.success_pos_tol and ori_err <= cfg.success_ori_tol
            and on_wall and detached
            and lin_speed < cfg.success_lin_vel and ang_speed < cfg.success_ang_vel
        )
        self._stable_count = self._stable_count + 1 if placed else 0
        success = self._stable_count >= cfg.success_steps

        # 6) Reward, every term logged separately.
        w = cfg.reward_weights
        da = a[:6] - self._prev_action[:6]
        terms = {
            "pos_error": -w["pos_error"] * pos_err,
            "ori_error": -w["ori_error"] * ori_err,
            "collision": -w["collision"] * float(collision),
            "contact_force": -w["contact_force"] * max(0.0, max_wall_force - cfg.force_threshold) / cfg.force_threshold,
            "action": -w["action"] * float(np.dot(a[:6], a[:6])),
            "jerk": -w["jerk"] * float(np.dot(da, da)),
            "drop": -w["drop"] * float(dropped),
            "success": w["success"] * float(success),
        }
        reward = float(sum(terms.values()))
        self._prev_action = a

        terminated = bool(success or dropped)
        truncated = bool(not terminated and self._steps >= cfg.max_steps)
        info = {
            "reward_terms": terms,
            "randomization": dict(self._rand),
            "action_clipped": action_clipped,
            "workspace_clipped": workspace_clipped,
            "attach_failed": attach_failed,
            "attached": self.attached,
            "fixture_active": self.fixture_active,
            "collision": collision,
            "max_panel_wall_force": max_wall_force,
            "panel_wall_dist": wall_dist,
            "pos_error": pos_err,
            "ori_error": ori_err,
            "ik_residual": ik_residual,
            "success": success,
            "is_success": success,
            "dropped": dropped,
            "steps": self._steps,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ observations
    def _get_obs(self):
        d, cfg, rng = self.data, self.cfg, self.np_random
        f32 = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)  # noqa: E731

        ee_p, ee_q = self._to_base(*self._ee_pose_world())
        panel_p, panel_q = self._to_base(*self._body_pose(self._panel_body))
        targ_p, targ_q = self._to_base(*self._target_pose_world())

        def noisy(p, q):
            if cfg.pos_noise_std > 0:
                p = p + rng.normal(0.0, cfg.pos_noise_std, 3)
            if cfg.rot_noise_std > 0:
                q = canonical(quat_mul(rotvec_to_quat(rng.normal(0.0, cfg.rot_noise_std, 3)), q))
            return p, q

        panel_pe, panel_qe = noisy(panel_p, panel_q)
        targ_pe, targ_qe = noisy(targ_p, targ_q)

        deployable = {
            "joint_pos": f32(d.qpos[self._qadr]),
            "joint_vel": f32(d.qvel[self._dadr]),
            "ee_pos": f32(ee_p),
            "ee_quat": f32(ee_q),
            "attached": f32([float(self.attached)]),
            "panel_pos_est": f32(panel_pe),
            "panel_quat_est": f32(panel_qe),
            "target_pos_est": f32(targ_pe),
            "target_quat_est": f32(targ_qe),
        }

        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self._panel_body, vel, 0)
        bq_inv = quat_to_mat(quat_conj(self._base_pose()[1]))
        c = self._last_contacts
        privileged = {
            "panel_pos": f32(panel_p),
            "panel_quat": f32(panel_q),
            "panel_linvel": f32(bq_inv @ vel[3:]),
            "panel_angvel": f32(bq_inv @ vel[:3]),
            "target_pos": f32(targ_p),
            "target_quat": f32(targ_q),
            "panel_wall_force": f32([c["wall"]]),
            "panel_ledge_force": f32([c["ledge"]]),
            "panel_rack_force": f32([c["rack"]]),
            "robot_env_force": f32([c["robot_env"]]),
        }

        # Panel pose expressed in the (estimated) target frame.
        ip, iq = pose_inv(targ_pe, targ_qe)
        rel_p, rel_q = pose_mul(ip, iq, panel_pe, panel_qe)
        policy = np.concatenate([deployable[k] for k in self._deployable_keys] + [f32(rel_p), f32(rel_q)])
        return {"deployable": deployable, "privileged": privileged, "policy": policy}

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        h, w = self.render_size
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            overview = mujoco.MjvCamera()
            # behind-left of the robot, looking at the wall
            overview.lookat[:] = [0.45, -0.15, 0.5]
            overview.distance, overview.azimuth, overview.elevation = 2.2, 35.0, -25.0
            # from the side (+y), looking across the robot at the wall
            side = mujoco.MjvCamera()
            side.lookat[:] = [0.45, -0.1, 0.55]
            side.distance, side.azimuth, side.elevation = 1.7, -90.0, -12.0
            self._cams = [overview, side]
        if self.cfg.challenge == "hard":
            # high overview of the whole ring (wall at the top of the image), and the easy side view
            # rotated with the wall so the placement is always seen from the same relative angle
            th = self._rand["wall_angle"]
            ov, side = self._cams
            ov.lookat[:] = [0.0, 0.0, 0.35]
            ov.distance, ov.azimuth, ov.elevation = 2.9, np.degrees(th), -62.0
            Rz = quat_to_mat(rotvec_to_quat([0, 0, th]))
            side.lookat[:] = Rz @ np.array([0.45, -0.1, 0.55])
            side.distance, side.azimuth, side.elevation = 1.7, -90.0 + np.degrees(th), -12.0
        frames = []
        for cam in self._cams:
            self._renderer.update_scene(self.data, camera=cam)
            frames.append(self._renderer.render())
        return np.concatenate(frames, axis=1)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
