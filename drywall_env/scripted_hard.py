"""Scripted controller for the HARD challenge (wall and rack anywhere around the robot), with two
behaviour modes. Uses only the 7-D action and deployable observations, like the easy controller.

Geometry: everything after the pick is computed in the WALL FRAME, the robot base frame rotated by
the wall angle theta (found from the target-pose estimate). In that frame the wall is exactly where
it is in the easy challenge, so the easy rotate -> advance -> pre-align -> contact -> release logic
carries over, including crossing the shoulder (joint2 = 0) on the wall's centre line.

New phases: SWING_OUT (home -> above the rack) and SWING_IN (rack -> in front of the wall) move the
EE along an arc around the base at a fixed radius, never through the robot. The panel is carried
flat, long side tangential, so the final 90 deg pitch onto the wall is the same as in the easy task.

Modes (setpoints are sampled per episode from the controller's own seed):
  demo     moderate setpoint variance and a little smooth noise -- a competent but varied human
           stand-in for the UMI demonstrations (mostly successes)
  explore  large setpoint variance, smooth (AR(1)) action noise up to std 0.3, random arc detours and
           random suction timing, including setpoints that cause realistic failures (stopping short
           of the wall, clipping the ledge, attach retries, early / late release) -- the behaviour
           policy for the sim offline-RL data
"""

from __future__ import annotations

import numpy as np

from .env import EnvConfig
from .scripted import ScriptedController
from .transforms import mat_to_quat, pose_inv, pose_mul, quat_to_mat

SETPOINT_RANGES = {
    "demo": dict(hover=(0.08, 0.12), descend_gap=(0.002, 0.004), lift_height=(0.42, 0.50),
                 swing_radius=(0.40, 0.45), rotate_x=(0.45, 0.50), standoff=(0.08, 0.12),
                 contact_speed=(0.004, 0.006), ledge_clearance=(0.004, 0.007), push_depth=(0.0003, 0.001),
                 gain=(0.5, 0.7), bump=(0.0, 0.03), noise=(0.0, 0.03), release_wait=(0, 1),
                 attach_gap=(0.0, 0.0), retreat=(0.08, 0.12)),
    "explore": dict(hover=(0.05, 0.18), descend_gap=(0.0, 0.008), lift_height=(0.36, 0.58),
                    swing_radius=(0.36, 0.52), rotate_x=(0.42, 0.55), standoff=(0.04, 0.18),
                    contact_speed=(0.002, 0.012), ledge_clearance=(-0.002, 0.012), push_depth=(-0.006, 0.003),
                    gain=(0.3, 0.9), bump=(0.0, 0.08), noise=(0.0, 0.2), release_wait=(0, 6),
                    attach_gap=(0.0, 0.04), retreat=(0.05, 0.15)),
}
SWING_STEP = 0.12  # rad: how far ahead on the arc the pursuit waypoint sits
NOISE_RHO = 0.9  # AR(1) correlation of the action noise (smooth drifts, not jitter)
PRECISION_PHASES = {"descend", "attach", "prealign", "contact", "release"}
PRECISION_NOISE_SCALE = 0.3  # noise is weaker where a human slows down to be precise
# A varied but purposeful operator moves on after a while instead of wobbling in place forever
# (without this, explore episodes sat in prealign until the 500-step time-out).
PHASE_TIMEOUT = {"approach": (60, "descend"), "rotate": (60, "advance"), "advance": (60, "prealign"),
                 "prealign": (60, "contact")}
# Rotating tool-down -> wall-facing needs height: below ~0.68 m, close to the base, joint6 hits its
# 3.75 rad limit (LABNOTEBOOK 2026-11-18). Rotate at least this high, then move down to pre-align.
MIN_ROTATE_Z = 0.68
# Swing radius: carried flat with the long side tangential, the panel corner reaches
# sqrt((r + 0.2)^2 + 0.3^2) from the base; demo keeps it inside the 0.75 m wall (r <= 0.45).


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Panel lying flat, face up, with panel x = the target's x axis, expressed in the wall frame
# (target x = -y_w, target z = -x_w; flat panel: x = -y_w, y = +x_w, z = +z_w).
R_FLAT_W = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class HardScriptedController(ScriptedController):
    PHASES = ["swing_out", "approach", "descend", "attach", "lift", "swing_in", "rotate", "advance",
              "prealign", "contact", "release", "retreat", "done"]

    def __init__(self, env_config: EnvConfig | None = None, mode: str = "demo", seed: int | None = None,
                 action_noise_std: float = 0.0):
        if mode not in SETPOINT_RANGES:
            raise ValueError(f"mode must be one of {list(SETPOINT_RANGES)}")
        self.mode = mode
        super().__init__(env_config, action_noise_std=action_noise_std, seed=seed)

    # --------------------------------------------------------------- per-episode setpoints
    def reset(self):
        super().reset()
        rng, R = self.rng, SETPOINT_RANGES[self.mode]
        sp = {k: float(rng.uniform(*v)) for k, v in R.items()}
        sp["release_wait"] = int(round(sp["release_wait"]))
        sp["bump_sign_r"] = float(rng.choice([-1.0, 1.0]))
        sp["bump_sign_z"] = float(rng.choice([-1.0, 1.0]))
        self.sp = sp
        self.hover, self.descend_gap = sp["hover"], sp["descend_gap"]
        self.lift_height, self.rotate_x, self.standoff = sp["lift_height"], sp["rotate_x"], sp["standoff"]
        self.contact_speed, self.ledge_clearance = sp["contact_speed"], sp["ledge_clearance"]
        self.push_depth, self.retreat_dist = sp["push_depth"], sp["retreat"]
        self.phase = "swing_out"
        self._noise = np.zeros(6)
        self._swing0 = None
        self._hold = 0

    def act(self, observation) -> np.ndarray:
        a = super().act(observation)  # base adds the white action_noise_std (if any) and clips
        sigma = self.sp["noise"] * (PRECISION_NOISE_SCALE if self.phase in PRECISION_PHASES else 1.0)
        if sigma > 0:
            self._noise = NOISE_RHO * self._noise + np.sqrt(1 - NOISE_RHO ** 2) * self.rng.normal(0, sigma, 6)
            a = a.copy()
            a[:6] = np.clip(a[:6] + self._noise, -1.0, 1.0)
        return a.astype(np.float32)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _wall_angle(targ_q) -> float:
        n = quat_to_mat(targ_q)[:, 2]  # target z points out of the wall, towards the robot
        return float(np.arctan2(-n[1], -n[0]))

    def _arc_goal(self, ee_p, goal_angle, radius, height):
        """Pursuit waypoint on an arc around the base towards goal_angle (no wrap through +-180 deg:
        both angles are inside the joint1 range, so the direct difference is the reachable way)."""
        a_cur = float(np.arctan2(ee_p[1], ee_p[0]))
        d = goal_angle - a_cur
        if self._swing0 is None:
            self._swing0 = (abs(d), radius, ee_p[2])
        d0 = max(self._swing0[0], 1e-6)
        s = float(np.clip(1.0 - abs(d) / d0, 0.0, 1.0))  # progress 0 -> 1
        a_wp = a_cur + float(np.clip(d, -SWING_STEP, SWING_STEP))
        bump = self.sp["bump"] * np.sin(np.pi * s)
        r = radius + self.sp["bump_sign_r"] * bump
        z = height + self.sp["bump_sign_z"] * 0.5 * bump
        return np.array([r * np.cos(a_wp), r * np.sin(a_wp), z]), a_wp, abs(d)

    def _set(self, phase):
        super()._set(phase)
        self._swing0 = None

    # --------------------------------------------------------------- policy
    def _act(self, obs) -> np.ndarray:
        panel_p, panel_q = obs["panel_pos_est"].astype(float), obs["panel_quat_est"].astype(float)
        targ_p, targ_q = obs["target_pos_est"].astype(float), obs["target_quat_est"].astype(float)
        attached = obs["attached"][0] > 0.5
        ee_p, ee_q = obs["ee_pos"].astype(float), obs["ee_quat"].astype(float)
        theta = self._wall_angle(targ_q)
        Rw = _rz(theta)
        g = self.sp["gain"]
        if self.phase in PHASE_TIMEOUT and self._t_phase > PHASE_TIMEOUT[self.phase][0]:
            nxt = PHASE_TIMEOUT[self.phase][1]
            self._set(nxt)
            if nxt == "contact":
                self._last_ee, self._stall = ee_p.copy(), 0
        ph = self.phase

        if ph == "swing_out":
            goal_angle = float(np.arctan2(panel_p[1], panel_p[0]))
            gp_grasp, gq_grasp = self._ee_goal_for_panel_pose(panel_p, panel_q)
            # orientation turns with the arc so that it equals the grasp orientation over the rack
            R_rel = _rz(-goal_angle) @ quat_to_mat(gq_grasp)
            wp, a_wp, rem = self._arc_goal(ee_p, goal_angle, self.sp["swing_radius"], self.lift_height)
            if rem < 0.05:
                self._set("approach")
            return self._goto(obs, wp, mat_to_quat(_rz(a_wp) @ R_rel), attach=False, gain=g)

        if ph in ("approach", "descend"):
            gp, gq = self._ee_goal_for_panel_pose(panel_p, panel_q)
            up = quat_to_mat(panel_q)[:, 2]
            if ph == "approach":
                hover = gp + up * self.hover
                pe, re = self._err(obs, hover, gq)
                if pe < 0.005 and re < np.radians(1.0):
                    self._set("descend")
                return self._goto(obs, hover, gq, attach=False, gain=g)
            goal = gp + up * self.descend_gap
            gap = float(np.dot(ee_p - gp, up))
            pe, _ = self._err(obs, goal, gq)
            # explore: sometimes try suction early (beyond 3 cm it is refused -> retry path)
            early = self.sp["attach_gap"] > 0 and gap < self.sp["attach_gap"]
            if pe < 0.002 or early:
                self._set("attach")
            return self._goto(obs, goal, gq, attach=False, max_speed=0.01, gain=g)

        if ph == "attach":
            if attached:
                ip, iq = pose_inv(panel_p, panel_q)
                self._grasp_ee_in_panel = pose_mul(ip, iq, ee_p, ee_q)
                self._lift_goal = (ee_p + [0, 0, self.lift_height - ee_p[2]], ee_q)
                self._set("lift")
                return self._goto(obs, ee_p, ee_q, attach=True, gain=0.0)
            if self._t_phase > 3:
                self.descend_gap = max(self.descend_gap - 0.002, -0.002)
                self.sp["attach_gap"] = 0.0
                self._set("descend")
            return self._goto(obs, ee_p, ee_q, attach=True, gain=0.0)

        # ---- carrying the panel: wall-frame geometry (the easy task rotated by theta)
        R_t = quat_to_mat(targ_q)
        wall_normal, up_wall = R_t[:, 2], R_t[:, 1]
        place_p = targ_p + up_wall * self.ledge_clearance
        wall_ee_p, wall_ee_q = self._ee_goal_for_panel_pose(place_p, targ_q, self._grasp_ee_in_panel)
        prealign_p = wall_ee_p + wall_normal * self.standoff
        to_w = lambda p: Rw.T @ p  # noqa: E731
        from_w = lambda p: Rw @ p  # noqa: E731

        if ph == "lift":
            gp, gq = self._lift_goal
            if self._err(obs, gp, gq)[0] < 0.01:
                self._set("swing_in")
            return self._goto(obs, gp, gq, attach=True, gain=g)

        if ph == "swing_in":
            # carry flat, long side tangential: panel orientation Rz(a) R_FLAT_W, EE = panel * grasp
            gpos, gquat = self._grasp_ee_in_panel
            R_grasp = quat_to_mat(gquat)
            z = max(wall_ee_p[2], MIN_ROTATE_Z)
            wp, a_wp, rem = self._arc_goal(ee_p, theta, self.sp["swing_radius"], z)
            if rem < 0.05:  # close in angle: finish at the easy-task rotate point
                wp = from_w(np.array([self.rotate_x, 0.0, z]))
                a_wp = theta
            R_panel = _rz(a_wp) @ R_FLAT_W
            gq = mat_to_quat(R_panel @ R_grasp)
            if rem < 0.05 and self._err(obs, wp, gq)[0] < 0.01:
                self._set("rotate")
            return self._goto(obs, wp, gq, attach=True, gain=g)

        if ph == "rotate":
            gp = from_w(np.array([self.rotate_x, 0.0, max(wall_ee_p[2], MIN_ROTATE_Z)]))
            pe, re = self._err(obs, gp, wall_ee_q)
            if pe < 0.01 and re < np.radians(1.0):
                self._set("advance")
            return self._goto(obs, gp, wall_ee_q, attach=True, gain=g)

        if ph == "advance":
            pw = to_w(prealign_p)
            gp = from_w(np.array([pw[0], 0.0, pw[2]]))
            if self._err(obs, gp, wall_ee_q)[0] < 0.01:
                self._set("prealign")
            return self._goto(obs, gp, wall_ee_q, attach=True, gain=g)

        if ph == "prealign":
            pe, re = self._err(obs, prealign_p, wall_ee_q)
            if pe < 0.003 and re < np.radians(0.5):
                self._set("contact")
                self._last_ee = ee_p.copy()
                self._stall = 0
            return self._goto(obs, prealign_p, wall_ee_q, attach=True, gain=g)

        if ph == "contact":
            dist = float(np.dot(panel_p - targ_p, wall_normal))
            progress = float(np.dot(self._last_ee - ee_p, wall_normal))
            self._last_ee = ee_p.copy()
            self._stall = self._stall + 1 if (self._t_phase > 2 and progress < 0.0005) else 0
            # A stall means "blocked by the wall" only when the panel is already close; further out
            # the arm is just creeping (it lags at 0.8 mm/step), so keep pushing a little longer.
            stall_needed = 2 if dist < 0.0015 else 6
            if dist < max(self.contact_tol, -self.push_depth) or self._stall >= stall_needed:
                self._hold = self.sp["release_wait"]
                self._set("release")
                return self._goto(obs, ee_p, ee_q, attach=True, gain=0.0)
            gp = wall_ee_p - wall_normal * self.push_depth
            speed = float(np.clip(0.25 * dist, 0.0008, self.contact_speed))
            return self._goto(obs, gp, wall_ee_q, attach=True, max_speed=speed, gain=1.0)

        if ph == "release":
            if self._hold > 0:  # random suction timing: keep holding a few steps
                self._hold -= 1
                self._t_phase = 0
                return self._goto(obs, ee_p, ee_q, attach=True, gain=0.0)
            if not attached and self._t_phase >= 2:
                self._retreat_goal = (ee_p + wall_normal * self.retreat_dist, ee_q)
                self._set("retreat")
            return self._goto(obs, ee_p, ee_q, attach=False, gain=0.0)

        if ph == "retreat":
            gp, gq = self._retreat_goal
            if self._err(obs, gp, gq)[0] < 0.005:
                self._set("done")
            return self._goto(obs, gp, gq, attach=False, max_speed=0.01)

        return np.array([0, 0, 0, 0, 0, 0, -1.0])


def make_scripted(cfg: EnvConfig, mode: str | None = None, seed: int | None = None,
                  action_noise_std: float = 0.0) -> ScriptedController:
    """Easy challenge -> the original controller (mode ignored); hard -> HardScriptedController."""
    if cfg.challenge == "hard":
        return HardScriptedController(cfg, mode=mode or "demo", seed=seed, action_noise_std=action_noise_std)
    return ScriptedController(cfg, action_noise_std=action_noise_std, seed=seed)
