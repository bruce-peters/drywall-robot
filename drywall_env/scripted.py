"""Scripted waypoint controller that uses ONLY the 7-D action interface and deployable obs.

Phases: approach (above panel) -> descend -> attach -> lift -> transport -> rotate ->
pre-align (10 cm off the wall at target orientation) -> slow approach until contact ->
release -> retreat -> done.

Everything is computed in the robot base frame from obs["deployable"]:
EE pose (forward kinematics), attach state, and panel / target pose *estimates*.
Known constants (as on a real robot): panel thickness and the tool (plate) geometry.
"""

from __future__ import annotations

import numpy as np

from .env import PANEL_HALF, EnvConfig
from .transforms import (
    mat_to_rpy,
    pose_inv,
    pose_mul,
    quat_error,
    quat_mul,
    quat_to_mat,
    rotvec_to_quat,
)

# Grasp: plate (EE) frame relative to the panel frame. EE x = panel x, EE z = -panel z,
# plate face on the panel's +z face (panel centre is half a thickness beyond the plate face).
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
GRASP_POS = np.array([0.0, 0.0, PANEL_HALF[2]])


class ScriptedController:
    PHASES = ["approach", "descend", "attach", "lift", "transport", "rotate", "advance",
              "prealign", "contact", "release", "retreat", "done"]

    def __init__(self, env_config: EnvConfig | None = None, action_noise_std: float = 0.0,
                 seed: int | None = None):
        self.cfg = env_config or EnvConfig()
        self.action_noise_std = action_noise_std
        self.rng = np.random.default_rng(seed)
        # Tuning
        self.hover = 0.10  # m above the panel before descending
        self.descend_gap = 0.003  # m: plate face stops this far above the panel face
        self.lift_height = 0.45  # m (base frame z) to lift the panel to
        # Rotate tool-down -> tool-forward at (rotate_x, 0, z) on the robot's centre line, clear
        # of the wall, then advance on the centre line to the standoff depth before moving
        # sideways. With the plate facing forward, the Panda's shoulder (joint2) passes through
        # 0 on the way out; doing that crossing at y = 0 keeps DLS IK from flipping
        # joints 1/3 (see LABNOTEBOOK, stage 3).
        self.rotate_x = 0.47
        self.standoff = 0.10  # m off the wall for pre-alignment
        self.contact_speed = 0.005  # m/step during the final approach (slows to 0.8 mm/step)
        self.contact_tol = 0.0005  # m: stop when estimated panel is this close to the target plane
        self.push_depth = 0.0005  # m: aim this far into the wall so contact is actually made
        self.ledge_clearance = 0.005  # m: approach this far above the target, settle on release
        self.retreat_dist = 0.10
        self.reset()

    def reset(self):
        self.phase = "approach"
        self._t_phase = 0
        self._grasp_ee_in_panel = None  # (pos, quat) of EE in panel frame, recorded at attach
        self._last_ee = None
        self._stall = 0
        self._retreat_goal = None

    # --------------------------------------------------------------- helpers
    def _goto(self, obs, pos, quat, attach, max_speed=None, gain=0.6):
        """P-controller towards a base-frame EE pose; returns a 7-D action in [-1, 1]."""
        cfg = self.cfg
        ee_p, ee_q = obs["ee_pos"].astype(float), obs["ee_quat"].astype(float)
        dp = gain * (np.asarray(pos) - ee_p)
        vmax = cfg.max_pos_step if max_speed is None else min(max_speed, cfg.max_pos_step)
        n = np.linalg.norm(dp)
        if n > vmax:
            dp *= vmax / n
        rv = gain * quat_error(quat, ee_q)
        ang = np.linalg.norm(rv)
        if ang > cfg.max_rot_step:
            rv *= cfg.max_rot_step / ang
        rpy = mat_to_rpy(quat_to_mat(rotvec_to_quat(rv)))
        a = np.concatenate([dp / cfg.max_pos_step, rpy / cfg.max_rot_step, [1.0 if attach else -1.0]])
        return np.clip(a, -1.0, 1.0)

    @staticmethod
    def _err(obs, pos, quat):
        ee_p, ee_q = obs["ee_pos"].astype(float), obs["ee_quat"].astype(float)
        return np.linalg.norm(np.asarray(pos) - ee_p), np.linalg.norm(quat_error(quat, ee_q))

    def _set(self, phase):
        self.phase = phase
        self._t_phase = 0

    def _ee_goal_for_panel_pose(self, panel_p, panel_q, grasp=None):
        gp, gq = grasp if grasp is not None else (GRASP_POS, GRASP_QUAT)
        return pose_mul(panel_p, panel_q, gp, gq)

    # --------------------------------------------------------------- policy
    def act(self, observation) -> np.ndarray:
        obs = observation["deployable"] if "deployable" in observation else observation
        a = self._act(obs)
        if self.action_noise_std > 0:
            a[:6] += self.rng.normal(0.0, self.action_noise_std, 6)  # attach bit is left clean
        self._t_phase += 1
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _act(self, obs) -> np.ndarray:
        panel_p, panel_q = obs["panel_pos_est"].astype(float), obs["panel_quat_est"].astype(float)
        targ_p, targ_q = obs["target_pos_est"].astype(float), obs["target_quat_est"].astype(float)
        attached = obs["attached"][0] > 0.5
        ee_p = obs["ee_pos"].astype(float)
        ph = self.phase

        if ph == "approach":
            gp, gq = self._ee_goal_for_panel_pose(panel_p, panel_q)
            hover = gp + quat_to_mat(panel_q)[:, 2] * self.hover
            pe, re = self._err(obs, hover, gq)
            if pe < 0.005 and re < np.radians(1.0):
                self._set("descend")
            return self._goto(obs, hover, gq, attach=False)

        if ph == "descend":
            gp, gq = self._ee_goal_for_panel_pose(panel_p, panel_q)
            goal = gp + quat_to_mat(panel_q)[:, 2] * self.descend_gap
            pe, _ = self._err(obs, goal, gq)
            if pe < 0.002:
                self._set("attach")
            return self._goto(obs, goal, gq, attach=False, max_speed=0.01)

        if ph == "attach":
            if attached:
                # Record the actual grasp: EE pose in the (estimated) panel frame.
                ip, iq = pose_inv(panel_p, panel_q)
                self._grasp_ee_in_panel = pose_mul(ip, iq, ee_p, obs["ee_quat"].astype(float))
                self._lift_goal = (ee_p + [0, 0, self.lift_height - ee_p[2]], obs["ee_quat"].astype(float))
                self._set("lift")
                return self._goto(obs, ee_p, obs["ee_quat"], attach=True)
            if self._t_phase > 3:  # attach refused: get a bit closer and retry
                self.descend_gap -= 0.002
                self._set("descend")
            return self._goto(obs, ee_p, obs["ee_quat"], attach=True)

        # From here on we carry the panel. EE goal for a desired panel pose uses the grasp.
        # The panel is brought in slightly above the target (clear of the ledge lip) and
        # settles onto the ledge after release.
        R_t = quat_to_mat(targ_q)
        wall_normal = R_t[:, 2]  # target z points out of the wall, at the robot
        up = R_t[:, 1]  # target y points up along the wall
        place_p = targ_p + up * self.ledge_clearance
        wall_ee_p, wall_ee_q = self._ee_goal_for_panel_pose(place_p, targ_q, self._grasp_ee_in_panel)

        if ph == "lift":
            gp, gq = self._lift_goal
            if self._err(obs, gp, gq)[0] < 0.01:
                self._set("transport")
            return self._goto(obs, gp, gq, attach=True)

        prealign_p = wall_ee_p + wall_normal * self.standoff

        if ph == "transport":
            gq = self._lift_goal[1]
            gp = np.array([self.rotate_x, 0.0, wall_ee_p[2]])
            if self._err(obs, gp, gq)[0] < 0.01:
                self._set("rotate")
            return self._goto(obs, gp, gq, attach=True)

        if ph == "rotate":
            gp = np.array([self.rotate_x, 0.0, wall_ee_p[2]])
            pe, re = self._err(obs, gp, wall_ee_q)
            if pe < 0.01 and re < np.radians(1.0):
                self._set("advance")
            return self._goto(obs, gp, wall_ee_q, attach=True)

        if ph == "advance":
            gp = np.array([prealign_p[0], 0.0, prealign_p[2]])
            if self._err(obs, gp, wall_ee_q)[0] < 0.01:
                self._set("prealign")
            return self._goto(obs, gp, wall_ee_q, attach=True)

        if ph == "prealign":
            gp = prealign_p
            pe, re = self._err(obs, gp, wall_ee_q)
            if pe < 0.003 and re < np.radians(0.5):
                self._set("contact")
                self._last_ee = ee_p.copy()
                self._stall = 0
            return self._goto(obs, gp, wall_ee_q, attach=True)

        if ph == "contact":
            # Remaining distance of the panel to the target plane along the wall normal.
            dist = float(np.dot(panel_p - targ_p, wall_normal))
            progress = float(np.dot(self._last_ee - ee_p, wall_normal))
            self._last_ee = ee_p.copy()
            self._stall = self._stall + 1 if (self._t_phase > 2 and progress < 0.0005) else 0
            if dist < self.contact_tol or self._stall >= 2:
                self._set("release")
                return self._goto(obs, ee_p, obs["ee_quat"], attach=True, gain=0.0)
            # Move along the normal (slowing down near the wall) while correcting alignment.
            gp = wall_ee_p - wall_normal * self.push_depth
            speed = float(np.clip(0.25 * dist, 0.0008, self.contact_speed))
            return self._goto(obs, gp, wall_ee_q, attach=True, max_speed=speed, gain=1.0)

        if ph == "release":
            if not attached and self._t_phase >= 2:
                self._retreat_goal = (ee_p + wall_normal * self.retreat_dist, obs["ee_quat"].astype(float))
                self._set("retreat")
            return self._goto(obs, ee_p, obs["ee_quat"], attach=False, gain=0.0)

        if ph == "retreat":
            gp, gq = self._retreat_goal
            if self._err(obs, gp, gq)[0] < 0.005:
                self._set("done")
            return self._goto(obs, gp, gq, attach=False, max_speed=0.01)

        # done: hold still, suction off
        return np.array([0, 0, 0, 0, 0, 0, -1.0])
