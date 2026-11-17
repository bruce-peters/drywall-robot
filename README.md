# Drywall panel placement — MuJoCo environment (Part A)

A Franka Panda with a suction plate picks a 0.6 × 0.4 × 0.01 m drywall panel off a rack,
carries it to a wall, aligns it with a target region, places it flush against the wall and
releases it onto a ledge fixture. Gymnasium-style API, Cartesian end-effector delta actions,
deployable vs. sim-only observations, separately logged reward terms, seeded randomization, and
a scripted waypoint controller that uses only the policy interface.

Demo video: [`media/seed0.mp4`](media/seed0.mp4), seed 0, with the overview on the left and the side view on the right.

## Setup

Python 3.10+ (developed on 3.12.4, Windows 11; mujoco 4.0.1).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

The Panda model comes from MuJoCo Menagerie via `robot_descriptions`. The first run clones
Menagerie into `~/.cache/robot_descriptions` (needs network once). The pinned
`robot_descriptions==3.3.0` resolved Menagerie commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`;
we use `franka_emika_panda/panda_nohand.xml`.

## Commands

```bash
python scripts/run_scripted.py --seed 0 --video out.mp4   # one episode + two-view mp4
python scripts/eval_scripted.py --n 50                    # success / error / collision stats
python -m pytest -q                                       # tests (~10 s)
```

`eval_scripted.py` also takes `--action-noise STD` (scripted-controller action noise) and
`--start-seed`.

## Interactive viewer

```bash
python scripts/view_interactive.py --seed 0
```

Opens the MuJoCo viewer plus a small Tk control panel with these controls:
- **Play / Pause**: runs the scripted controller at 10 Hz times the speed slider (0.25–4x).
- **Step**: advances exactly one `env.step`.
- **Reset**: resets with the seed in the box, and resets the controller too.
- **Readout**: step, phase, panel pos/ori error, attached, fixture_active, collision, last reward,
  and the final status (SUCCESS / DROPPED / TRUNCATED).
- **Manual**: the scripted controller is bypassed. You drive the 7-D action with sliders that
  spring back to 0 and a suction toggle. Unticking it restarts the controller at "approach".

When the episode ends, stepping stops until you press Reset. Closing either window closes both.

## Results (scripted controller, seeds 0–49, noise-free observations)

| metric | value |
|---|---|
| success rate | **100 %** (50/50) |
| drop rate | 0 % |
| mean final position error | 3.07 mm (the 3 mm is the designed drop onto the ledge) |
| mean final orientation error | 0.037° |
| collision rate (episodes with ≥1 collision step) | 0 % |
| mean episode length | 171 steps (17.1 s) |
| peak panel–wall force | mean 15 N, worst 55 N (seed 18, just above the 50 N threshold) |

With `--action-noise 0.2` (10 seeds): 40 % success, 50 % collision rate, which shows the knob
produces imperfect demos.

Seeds: every random quantity is drawn from `env.reset(seed=N)` (Gymnasium `np_random`).
The scripted controller is deterministic unless `action_noise_std > 0` (its own `seed`).
Same seed → bit-identical rollout on the same machine/build (tested). Results may differ
across MuJoCo versions or platforms.

## Layout

```
drywall_env/
  assets/scene.xml   work cell: floor, wall, target (site + marker + ledge), rack, free panel,
                     disabled welds "grasp" (plate↔panel) and "fixture" (wall↔panel)
  model.py           MjSpec: scene + Menagerie Panda (no hand) + suction plate, gravcomp on
  ik.py              damped-least-squares IK (mj_jacSite) + null-space posture pull
  env.py             DrywallPlacementEnv, EnvConfig
  scripted.py        ScriptedController (waypoint state machine)
  transforms.py      quaternion / RPY helpers (wxyz)
scripts/run_scripted.py, scripts/eval_scripted.py
tests/test_env.py
LABNOTEBOOK.md       full development log (every decision, failure and fix)
```

### Scene (robot base frame = world; x towards the wall, z up)

| item | value |
|---|---|
| Panda base | origin |
| rack | top surface z = 0.20 m, centre (0.35, −0.50); panel lies with its long side along y |
| wall | front face x = 0.75 m |
| target (panel centre when flush) | (0.745, 0, 0.70) ± 5 cm in the wall plane (y, z) |
| ledge | top 3 mm below the target's bottom edge, 4 cm deep, moves with the target |
| suction plate | 24 × 18 × 1 cm, 0.5 kg, on the flange; EE frame at the centre of its outer face |

Distances were tuned for the Panda's reach (see LABNOTEBOOK, stages 1 and 3). The main constraints:
- **Target height:** with the plate facing the wall, a non-flipped wrist only reaches at heights of about 0.6 m and above, so the target centre is at 0.70 m.
- **Wall distance:** the wall sits at 0.75 m so the 10 cm pre-align pose is past the shoulder's joint2 = 0 crossing.
- **Rack orientation:** the panel lies on the rack so that the pick pose is the Panda's home "tool down" orientation.

## Environment API

```python
from drywall_env import DrywallPlacementEnv, EnvConfig, ScriptedController
env = DrywallPlacementEnv(EnvConfig(pos_noise_std=0.0), render_mode="rgb_array")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

**Timing:** physics dt 0.002 s, 50 substeps per step → 10 Hz control, max 300 steps (30 s).

**Action** `Box(-1, 1, (7,), float32)`: `[dx, dy, dz, droll, dpitch, dyaw, attach]`.
- **Translation:** each step moves at most 2 cm.
- **Rotation:** at most 3° per component, applied as `R ← Rz(dyaw)·Ry(dpitch)·Rx(droll)·R`.
- **Frame:** everything is in the robot base frame.
- **Attach:** `attach > 0` means attach / hold, `≤ 0` means release.

How a step is processed:
1. The action is clipped to [−1, 1] (`info["action_clipped"]`).
2. The delta is added to an internal *commanded* EE pose. On `reset` this pose is set to the measured pose.
3. The commanded position is clipped to a workspace box, x ∈ [0.10, 0.80], y ∈ [−0.75, 0.60], z ∈ [0.15, 1.00] (`info["workspace_clipped"]`).
4. DLS IK turns the commanded pose into joint targets for the position actuators. It runs 10 iterations, warm-started from the previous solution.

**Observation** (`Dict`; quaternions wxyz, everything in the robot base frame):

| group | key | dim | notes |
|---|---|---|---|
| **deployable** (a real robot has these) | `joint_pos`, `joint_vel` | 7, 7 | encoders |
| | `ee_pos`, `ee_quat` | 3, 4 | forward kinematics of the plate frame |
| | `attached` | 1 | suction state (on hardware: vacuum sensor) |
| | `panel_pos_est`, `panel_quat_est` | 3, 4 | true + Gaussian noise (`pos_noise_std`, `rot_noise_std`, default 0) |
| | `target_pos_est`, `target_quat_est` | 3, 4 | same noise model |
| **privileged** (sim only) | `panel_pos`, `panel_quat` | 3, 4 | exact |
| | `panel_linvel`, `panel_angvel` | 3, 3 | exact |
| | `target_pos`, `target_quat` | 3, 4 | exact |
| | `panel_wall_force`, `panel_ledge_force`, `panel_rack_force`, `robot_env_force` | 1 each | summed normal contact forces (N) |
| **policy** | flat float32 vector | 43 | the 36 deployable values (in the order above), then the panel pose in the estimated target frame (pos 3 + quat 4) |

**Reward** (each term logged in `info["reward_terms"]`; the reward is their sum):

| term | definition (default weight) |
|---|---|
| `pos_error` | −1.0 · ‖panel − target‖ (m) |
| `ori_error` | −0.3 · geodesic angle (rad) |
| `collision` | −1 per step with a collision. A collision is any robot geom touching the wall, rack, ledge or floor; a robot link other than the plate touching the panel; or the panel touching the wall more than 5 cm (in the wall plane) from the target |
| `contact_force` | −(F − 50)/50 when the peak panel–wall normal force over the step exceeds 50 N |
| `action`, `jerk` | −0.01‖a₁:₆‖², −0.01‖a₁:₆ − a_prev‖² |
| `drop` | −50 once: the panel is detached and either touches the floor or touches neither rack nor ledge for 3 consecutive steps |
| `success` | +100 once |

**Success:** the panel is within 20 mm and 5° of the target, touching the wall (panel–wall distance ≤ 2 mm), released, and stable (|v| < 1 cm/s, |ω| < 0.05 rad/s) for 5 consecutive steps.
**Termination:** `terminated` = success or drop. `truncated` = time limit only. They are never merged.

**Randomization** (all from the reset seed; ranges in `EnvConfig`). The sampled values are returned in `info["randomization"]` from both `reset` and `step`:
- panel start offset ±5 cm in x and y on the rack (`panel_dx`, `panel_dy`)
- panel yaw ±10° (`panel_dyaw`)
- target offset ±5 cm in the wall plane (`target_dy`, `target_dz`); the ledge moves with it
- panel mass 1–3 kg (`panel_mass`), with inertia scaled to match
- sliding friction 0.5–1.0 (`panel_friction`)

Each reset restores the nominal model before sampling, so randomization never accumulates across episodes.

Other `info` keys: `attached`, `attach_failed`, `fixture_active`, `collision`,
`max_panel_wall_force`, `panel_wall_dist`, `pos_error`, `ori_error`, `ik_residual`, `success`
/ `is_success`, `dropped`, `steps`.

## Design decisions

**Why suction + weld.**
- **Why suction:** drywall is handled with vacuum lifters in practice. A flat plate also contacts the large face, so it can carry the panel and then press it flush against the wall. A parallel gripper can do neither on a 1 cm panel.
- **How attachment is modelled:** a weld equality constraint between plate and panel. It switches on at runtime only when the plate face is within 3 cm of the panel face, centred over it, and within 30° of parallel.
- **No snap:** the weld's relative pose is set to the *current* relative pose, so the panel doesn't jump when it switches on (measured: 0.00 mm).
- **Release:** the weld is switched off.
- **Plate–panel contacts:** excluded, standing in for compliant suction cups. Without this, the contact and the weld would fight each other.
- **Trade-off:** this is a kinematic "perfect suction" model. There is no vacuum dynamics, leak or peel-off.

**Why Cartesian deltas in the base frame.**
- **Portability:** the policy's output is independent of the arm's kinematics, so it transfers across robots and IK/controller implementations.
- **Bounded steps:** a delta per step caps how far the arm moves in one step (2 cm / 3°), which is a natural safety envelope.
- **Base frame:** chosen over the tool frame because the task geometry (wall, rack, workspace limits) is fixed in the base frame. A base-frame delta means the same thing wherever the tool is pointing.
- **Integrated command:** deltas are added to a commanded pose rather than to the measured pose. Adding to the measured pose would let the payload sag accumulate step after step.

**How actions map to a real robot controller.**
- **Host side:** the same steps run on the host at 10 Hz: clip the action, add the delta to the commanded pose, clip to the workspace box.
- **Robot side:** the commanded pose is streamed as a Cartesian pose setpoint to the robot's own controller, for example a Franka Cartesian impedance or pose controller via libfranka / `franka_ros2`, or a UR's `servoL` over RTDE. On the real arm, the vendor controller's interpolation and IK replace our DLS IK and joint position servos.
- **Attach bit:** drives the vacuum valve output. `attached` is read back from a vacuum-pressure switch.
- **Pose estimates:** `panel_*_est` and `target_*_est` would come from perception (fiducials on the rack, a laser- or camera-registered wall layout). The noise parameters stand in for that error.

**Other choices**
- **Gravity compensation:** on for all Panda links, as the real Panda's controller does. The payload is not compensated.
- **IK:** DLS with λ = 0.05, plus a null-space pull toward the home posture.
- **Fixture:** the ledge carries the panel's weight. When a released panel rests on the ledge within 2 mm of the wall, a panel→wall weld engages at the panel's *current* pose, standing in for the external fixture. Placement error therefore stays visible; the fixture doesn't correct it. A panel released in mid-air still falls.
- **Contact tolerance:** "in contact with the wall" is defined as distance ≤ 2 mm (`mj_geomDistance`), not as an active contact force. Contact compliance makes a released panel spring back about 1 mm. Once the fixture weld holds the pose, an active-contact test would flicker.

## Scripted controller

`ScriptedController(env_config, action_noise_std=0.0, seed=None)` uses only `obs["deployable"]`
and the 7-D action. It needs two known constants, the panel thickness and the plate geometry, both of which a real cell would know.

1. **Approach:** move to 10 cm above the panel's estimated pose. The grasp follows the panel's yaw.
2. **Descend:** go down at ≤ 1 cm/step to 3 mm above the panel face.
3. **Attach:** turn suction on. If the attach is refused, move 2 mm lower and retry. Once attached, record the actual grasp (the EE pose in the panel frame).
4. **Lift:** raise to z = 0.45 m.
5. **Transport:** move to (0.47, 0, z_target) with the tool still facing down.
6. **Rotate:** turn to the wall orientation, clear of the wall. The recorded grasp is used here, which also corrects the panel's ±10° yaw on the rack.
7. **Advance:** move out on the centre line to the pre-align depth.
8. **Pre-align:** go to 10 cm off the wall at the target orientation, 5 mm above the target to clear the ledge lip.
9. **Contact:** approach slowly, from 5 mm/step down to 0.8 mm/step, aiming 0.5 mm into the wall, until the estimated gap is under 0.5 mm or the EE stops making progress.
10. **Release:** turn suction off. The panel settles about 8 mm onto the ledge.
11. **Retreat:** back off 10 cm.

`action_noise_std` adds Gaussian noise to the 6 motion components only; the attach bit stays clean.

## Limitations (honest list)

- **Rigid panel.** Drywall flexes and sags noticeably at 0.6 m spans; real panels are 1.2 × 2.4 m.
  No flex or breakage model.
- **Weld-based attachment.** Ideal suction: no vacuum build-up time, no leak, no slip or peel
  under moment load, no cup compliance. Attach is a distance/alignment check, not a seal model.
- **Approximate contact.** MuJoCo soft contacts; forces depend on solver parameters.
  Wall forces come from position-servo stiffness, not a force or impedance controller, so the scripted contact phase reaches up to about 55 N.
  A released panel springs back about 1 mm, which is why contact is defined as ≤ 2 mm.
- **Fixture is a weld.** It holds the panel at whatever pose it was released in (within 2 mm of the wall). There are no fastening forces and no clamping toward flush.
- **No camera-based perception yet.** Panel and target poses are ground truth plus optional
  Gaussian noise. There are no occlusions, outliers or latency.
- **IK limitation.** With the plate facing the wall, the Panda's shoulder (joint2) passes through 0 on the way out.
  If that crossing happens with a sideways offset, DLS IK can drift into a flipped joint1/joint3 branch.
  The scripted controller avoids this by crossing on the centre line. A learned policy can still run into it. A better fix would be a posture-aware or multi-seed IK, or the real robot's Cartesian controller.
- **Episode ends on success** 5 steps after release, so the retreat is cut short in the video.
- **Single panel, single target height range** (0.65–0.75 m). The arm cannot reach a plate-forward
  pose low on the wall (≲0.6 m) without a flipped wrist.
- **Determinism** is per machine and build; MuJoCo does not guarantee bit-identical results across
  platforms.
