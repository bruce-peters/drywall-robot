# Lab Notebook — Drywall Panel Placement Sim (Part A)

Chronological log of everything done on this project: commands, findings, design
decisions (with who decided and why), failures and fixes. Newest entries at the bottom.

Format: `## YYYY-MM-DD — <topic>` then short bullet entries. Decisions are tagged
**[DECISION]** with status `proposed` / `approved by user` / `changed`.

---

## 2026-11-14 — Session 1: kickoff and environment survey

- Task: build Part A of the robotics deployment take-home — a MuJoCo env for picking a
  drywall panel off a rack and placing it flush on a wall (spec pasted in chat; ~3 h budget).
- User rules for this project: log everything here; every design decision goes to chat first.
- Workspace `drywall-robot/` was empty. Not a git repo.
- Environment survey:
  - Python 3.12.4 (only interpreter installed), Windows 11, NVIDIA RTX 4070 (offscreen
    rendering via GLFW should work).
  - `mujoco`, `robot_descriptions` not installed yet.
  - Latest on PyPI today: mujoco 4.0.1, robot_descriptions 3.3.0, gymnasium 1.4.0,
    imageio 2.38.1, imageio-ffmpeg 0.8.1, numpy 2.5.4, pytest 9.2.0.
- Next: present open design decisions in chat before writing any code.

### Design decisions (presented in chat 2026-11-14)

User answered the 4 open forks (all recommended options); the remaining defaults were
presented in the same message and not objected to.

- **[DECISION] Fixture = ledge + auto-hold** (approved by user). Ledge carries the weight;
  when the panel is released while touching both ledge and wall, a panel→wall weld is
  activated at the panel's *current* pose (stand-in for the external fixture). Why: a
  vertical panel on a ledge is balanced on an edge and tips back; this keeps the physics
  honest about placement error without flaky tipping. Mid-air releases still fall.
- **[DECISION] Panda from `robot_descriptions`** (approved). Pinned package version pins
  the Menagerie commit; first run clones Menagerie into ~/.cache.
- **[DECISION] Gravity compensation on arm links** (approved). Mirrors the real Panda's
  controller; payload not compensated.
- **[DECISION] git init + project-local `.venv`** (approved). Commit only when user says.
- **[DECISION] Defaults presented, not objected to** (approved implicitly):
  - Layout: Panda base at origin; wall face ≈ x=0.65 m; target 0.6 w × 0.4 h, center
    ≈0.45 m high; ledge top ≈3 mm below target bottom edge; rack ≈(0.25, −0.5), panel
    flat, long side along x. Numbers to be tuned for reach and documented.
  - Control: integrate deltas into an internal commanded EE pose (reset to measured on
    reset); rotation deltas in base frame `R ← R_delta·R`; clip action to [−1,1], then
    commanded position to a workspace box; separate clip flags in info.
  - IK: ~10 DLS iterations per control step on scratch MjData + small null-space pull to
    home posture.
  - Attach only if attach>0, plate face within 3 cm of panel top face and plate centre over
    panel; weld relpose = current relative pose; plate–panel contacts excluded (compliant
    suction cups); `info["attach_failed"]` otherwise.
  - Collision: robot geom vs wall/rack/ledge/floor, robot non-plate link vs panel, panel vs
    wall when >5 cm in-plane from target.
  - Drop: detached and (touches floor or no rack/ledge contact for 3 consecutive steps).
  - Success: spec criteria; stable = |v|<1 cm/s, |ω|<0.05 rad/s.
  - Reward weights: −1.0·pos_err(m), −0.3·ori_err(rad), −1/step collision,
    −(F−50)/50 above 50 N, −0.01‖a‖², −0.01‖Δa‖², −50 drop, +100 success.
  - Obs: dict {deployable, privileged, policy(flat float32)}; wxyz quats;
    `pos_noise_std`, `rot_noise_std` default 0; subclass `gymnasium.Env`.
  - Randomization: ledge moves with target; reset rebuilds from nominal model;
    sampled values in `info["randomization"]`.
  - Scripted controller: P-controller to waypoints on deployable obs only; grasp transform
    recorded at attach and used to compute EE pose at wall (corrects yaw); approach at
    ~5 mm/step until estimated panel within ~2 mm of target plane or no progress.
  - Video: two cameras side by side, one frame per control step, 10 fps (real time).
- Process note: brainstorming skill wants a separate spec doc + commit; using this notebook
  as the design record instead, and not committing (user commits on request only).

## 2026-11-14 — Stage 1: arm loads, IK tracks a moving target

- `git init`; `.venv` with mujoco 4.0.1, robot_descriptions 3.3.0, gymnasium 1.4.0,
  numpy 2.5.4, imageio 2.38.1, imageio-ffmpeg 0.8.1, pytest 9.2.0.
- `robot_descriptions.panda_mj_description` cloned Menagerie to
  `~/.cache/robot_descriptions/mujoco_menagerie` at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`.
  Using `panda_nohand.xml` (7 position-style `general` actuators, kp 4500/3500/2000,
  joints 5–7 force-limited to ±12 Nm).
- Wrote `drywall_env/assets/scene.xml` (floor, wall, target body with site + marker + ledge,
  rack, free panel, two disabled welds `grasp` / `fixture`, plate–panel contact exclude) and
  `drywall_env/model.py` (MjSpec: load scene + Panda, gravcomp=1 on all Panda bodies, add
  plate body 24×18×1 cm, 0.5 kg on the flange with `ee` site at the plate's outer face,
  attach Panda at `robot_base` frame with prefix `panda/`). Compiles: nq 14, nv 13, nu 7.
- Weld `relpose` convention verified from the auto-computed `eq_data` at qpos0:
  `eq_data[3:6]` = position of body2 in body1's frame, `eq_data[6:10]` = body2's
  orientation in body1's frame. (mujoco 4 weld eq_data has 12 entries; only 3:10 used.)
- `drywall_env/ik.py`: DLS IK (λ=0.05, 10 iterations, ≤0.2 rad/iteration, null-space pull to
  Menagerie home posture, gain 0.1). `drywall_env/transforms.py`: quaternion/RPY helpers.
- Check script (scratchpad `check_ik.py`), 10 Hz control / 50 substeps:
  - First attempt used unreachable test targets (x=0.85 m low) → 400 mm error. Test bug.
  - Circle r=10 cm around (0.5, 0, 0.4): mean 3.6 mm, rot 0.03°.
  - Task waypoints with panel long side along x on the rack: joint7 hit its −2.9 limit at
    the pick and IK then wandered to a bad wall configuration (joint1 ≈ 2.4 rad, joint6 at
    limit, 15–70 mm residual).
  - Static IK sweep (random seeds): plate pointing horizontally at the wall (+x) is only
    reachable with a non-flipped wrist at heights ≳0.6 m; at z=0.5 all solutions need
    joint5≈±2.9 or joint6 at its 3.75 limit.
- **[DECISION] Layout tuned for reach** (within the approved "tune numbers for reach"):
  - Panel lies on the rack with its long side along world y (panel x = −world y), rack at
    (0.35, −0.5), top at z=0.20. Pick orientation = Panda home "tool down"; wall
    orientation = pure 90° pitch from it. Grasp: ee x = panel x, ee z = −panel z.
  - Target centre raised to z=0.70 m (panel spans 0.5–0.9 m), wall face x=0.70 m.
  - Rotation from tool-down to tool-forward must happen ≥25 cm from the wall (panel edges
    sweep a 0.2 m radius about the plate) → scripted controller rotates at x≈0.45.
- Result with new layout (home → above rack → pick → lift → (0.45,−0.1,0.6) → rotate →
  (0.59,0,0.7) → wall (0.69,0,0.7) → ±5 cm offsets): ~11 mm lag while moving at the
  2 cm/step limit (one-step servo lag), <1 mm once the target stops, joint6 ≈3.3–3.45 at
  the wall (≥0.3 rad margin). **Stage 1 passed.**

## 2026-11-14 — Stage 2: panel + attach/release

- Wrote `drywall_env/env.py` (`DrywallPlacementEnv`, `EnvConfig`) with the full step pipeline:
  clip action → integrate commanded EE pose (base frame) → workspace clip → attach/release →
  DLS IK → ctrl → 50 substeps tracking contacts (max wall force, collision, floor) → metrics,
  reward terms, termination. Also wrote `drywall_env/scripted.py` (needed by the package import).
- Check (scratchpad `check_attach.py`, randomize off):
  - attach requested far from the panel → refused, `attach_failed=True`. ✔
  - scripted approach/descend → attach at t=45; panel moved **0.00 mm** during the attach step
    (relpose = current relative pose, no snap). ✔
  - lift 20 cm: grasp offset changed by <0.5 mm; panel at rest while held. ✔
  - release over the rack: panel fell back onto the rack → *supported*, not dropped (correct).
  - release off the rack: panel hits the floor after 3 steps → `dropped=True`, terminated,
    `drop` term −50. ✔ **Stage 2 passed.**

## 2026-11-14 — Stage 3: scripted placement on one seed + video

- `scripts/run_scripted.py` written. First run (seed 0): reached 4.9 mm / 0.67° but no success
  and a contact-force penalty. Debugged with per-step traces (scratchpad `debug_place.py`):
  1. Panel came in with only 3 mm clearance over the ledge lip; servo lag (~1 cm at speed) made
     it clip the ledge (130–169 N spikes). → **Controller** now approaches 5 mm above the target
     (8 mm ledge clearance) and lets the panel settle onto the ledge on release.
  2. Second run still hit the ledge: arm configuration had drifted to joint1≈+2.4, joint3≈−2.4
     and the wrist whipped near the wall. Root cause (scratchpad `ik_cross.py`, `ik_planar.py`,
     `ik_lim.py`): with the plate facing forward, joint2 changes sign between the rotate pose
     (x≈0.45, joint2≈−0.45) and the wall (joint2≈+0.35); the crossing happened right at the
     pre-align point (x≈0.59), and with a lateral offset (y≠0) DLS escapes into the flipped
     joint1/joint3 branch. Not fixed by null-space gain/damping (tested 0.1–1.0 / 0.05–0.1);
     Tikhonov posture regularisation avoids the flip but leaves 20–100 mm residual; narrowing
     joint1/3 limits traps the IK (160–270 mm residual). A planar path exists (joint6 3.3–3.5,
     inside its limit) and the crossing is trivial at y=0.
     → **[DECISION] Wall face moved 0.70 → 0.75 m** (target x 0.745), so the 10 cm pre-align
     point is past the crossing, and **the controller crosses on the robot's centre line**
     (new sub-phase `advance`: rotate at (0.47, 0, z), advance at y=0 to the standoff depth,
     then shift sideways to pre-align). Verified kinematically: 0 mm residual, ≥0.42 rad joint
     margin for target z 0.65–0.75, y ±5 cm. Documented as an IK limitation for learned
     policies (they can still hit this branch flip if they cross joint2=0 off-centre).
  3. Contact pushed 2 mm into the wall → 74–94 N, and on release the compressed contact
     sprang the panel 1.3 mm off the wall; it then stood on the ledge without touching the wall,
     so neither the fixture nor success triggered.
     → **Controller**: final approach slows to 0.8 mm/step (speed = 0.25·distance), aims 0.5 mm
     into the wall, releases when the estimated gap <0.5 mm or the EE stalls.
     → **[DECISION, to report to user] "In contact with the wall" = panel–wall distance ≤ 2 mm**
     (`mj_geomDistance`, `EnvConfig.wall_contact_tol`), used both for success and for engaging
     the fixture (released + resting on the ledge + within 2 mm of the wall). The fixture still
     holds the panel at its current pose. Reason: active-contact checks are too brittle once a
     weld holds the pose / with sim contact compliance.
- Result seed 0: success at step 172–175, final 3.1 mm / 0.03° (the 3 mm is the designed drop
  onto the ledge), no collision.
- First 20 seeds (`scripts/eval_scripted.py --n 20`): **20/20 success**, mean 3.07 mm / 0.035°,
  0 % collisions, mean length 170 steps, panel–wall force mean 17.6 N, worst 54.9 N (seed 18,
  just over the 50 N penalty threshold).
- Video: first cameras were wrong (overview looked at the back of the wall: MuJoCo free-camera
  azimuth sign). Fixed (overview azimuth 35°, side −90°) and added a headlight. Saved
  `media/seed0.mp4` (186 frames @ 10 fps, real time, two views side by side). Stills checked:
  pick, rotation clear of the wall, panel flush over the green target on the ledge.
  Note: success terminates the episode ~5 steps after release, so the retreat is cut short.
  **Stage 3 passed.**

## 2026-11-14 — Stage 4: randomization, reward terms, tests, README

- Randomization and reward terms were already in `env.py` from stage 2; verified by tests.
- `tests/test_env.py` (11 tests, ~8 s): identical rollouts for the same seed (random actions and
  scripted), different seeds differ, sampled factors within ranges *and* applied to the model,
  action clipping reported and bounded to 2 cm, workspace clipping, far attach refused,
  observation groups + `observation_space.contains`, observation noise, reward terms sum to
  reward, scripted controller ≥18/20 successes. **11 passed.**
- `gymnasium.utils.env_checker.check_env`: passes (only warnings: unbounded obs boxes).
- `scripts/eval_scripted.py --n 50`: **100 % success**, 0 % drops, mean 3.07 mm / 0.037°,
  0 % collision rate, mean length 171.1 steps, panel–wall force mean 15.2 N / worst 54.9 N.
- `--action-noise 0.2`, 10 seeds: 40 % success, 50 % collision rate, mean 48 mm — the noise knob
  produces imperfect demonstrations as intended.
- Extras beyond the spec (small, reported to user): `eval_scripted.py` flags `--action-noise`
  and `--start-seed`; extra info keys (`fixture_active`, `panel_wall_dist`, `ik_residual`,
  `max_panel_wall_force`, `is_success`).
- `requirements.txt` pinned (mujoco 4.0.1, robot_descriptions 3.3.0, gymnasium 1.4.0, numpy
  2.5.4, imageio 2.38.1, imageio-ffmpeg 0.8.1, pytest 9.2.0); `.gitignore` (keeps
  `media/seed0.mp4`). README written: setup, commands, seeds, results, layout, API tables
  (deployable vs privileged), reward/success/termination, design decisions, mapping to a real
  controller, scripted controller, limitations.
- Nothing committed (user commits on request).

## 2026-11-16 — Interactive viewer (`scripts/view_interactive.py`)

- **[DECISION]** (given by user) 3D window = MuJoCo's built-in passive viewer
  (`mujoco.viewer.launch_passive(env.model, env.data)`, `viewer.sync()` after every env.step).
- **[DECISION]** (given by user) Control panel = small Tkinter window in its own top-level window.
  No web UI, no new dependencies (tkinter ships with Python, Tk 8.6 here).
- **[DECISION]** (given by user) Only one new file, `scripts/view_interactive.py`.
  `drywall_env/` is unchanged.
- **[DECISION]** Threading: everything runs on one thread. The Tk main thread does all env
  access, and the stepping loop is `root.after`, which also polls `viewer.is_running()`. The passive
  viewer renders on its own internal thread, so `env.step` / `env.reset` are wrapped in
  `viewer.lock()` so it never draws a half-updated model/data (`reset` also changes model
  params). Chosen because it needs no locks of our own and the env stays single-threaded.
- Play at speed s schedules the next tick `100/s ms − step compute time` later. The first
  version ignored compute time: 4x gave 34 steps/s, not 40 (env.step ≈ 4 ms). Fixed: 39.2 steps/s.
- Reset pauses playback. Unticking Manual calls `ctrl.reset()` (phase → approach).
- Reference: `python scripts/run_scripted.py --seed 0` → success at step 175, 3.12 mm / 0.029°.
- Test: scratchpad `drive_viewer.py` drives the real `App` (real Tk + real passive viewer) by
  calling its methods and pumping `root.update()`:
  3 Steps → step count 3. Reset → step 0, phase approach. Manual dz=+1 for 5 steps → EE +89 mm
  (100 mm commanded). Back to scripted → phase approach. Play at 4x on seed 0 → **SUCCESS at
  step 175, 3.12 mm / 0.029°, same as run_scripted**. Step after the end does nothing. Reset
  + Play resumes. Closing the viewer window closes the app (env renderer released). Closing the
  Tk window closes the viewer. **ALL OK.**
- `timeout 8 python scripts/view_interactive.py --seed 0` → ran until killed, no traceback.
  `pytest -q` → 11 passed.
- Not verified: I cannot see the screen, so nobody has checked how the windows look or used real
  mouse interaction (slider spring-back on mouse release, button clicks).
- Found, not fixed (`drywall_env/` is off-limits for this task): `ScriptedController.reset()` does
  not restore `descend_gap`, which the attach-retry path lowers by 2 mm each time. A reused
  controller (`eval_scripted.py`, `test_env.py`, Reset in the viewer) keeps the smaller gap in
  later episodes. Seed 0 attaches on the first try, so it is not affected.

## 2026-11-16 — UMI-style dataset recorder (`scripts/record_umi.py`)

Goal (user): record 500 scripted episodes as a simulated UMI (Chi et al. 2024, arXiv:2402.10329)
dataset in one `.npz`: timestamped EE poses, gripper/attach state, camera images, panel pose in
the target frame, episode start/end, success label. Mid-task the user added: UMI style = 60 Hz,
absolute poses, world frame.

From the UMI paper (fetched PDF): wrist GoPro, 155° fisheye, raw fisheye fed to the policy
(no undistortion), side mirrors for implicit stereo, policy images 224×224, 10 Hz obs/action
for quasi-static tasks, IMU-aided ORB-SLAM3 poses, continuous gripper width via fiducials.

### Assumptions (requested by user, tracked here)
- **No SLAM drift:** EE poses are the exact sim poses. Real UMI poses come from monocular
  visual-inertial SLAM and carry drift, relocalisation jumps and scale error.
- **Perfect gripper tracking:** `attached` is the exact suction state. Real UMI estimates
  gripper width from fiducial markers, with noise and dropouts.
- **Human motion approximated by a scripted controller:** demonstrations come from
  `ScriptedController` plus per-episode action noise, not from humans. There is no human
  multimodality, hesitation or re-grasping, no hand-held dynamics, and the "human" moves at
  robot speeds.

### Decisions
- **[DECISION] Camera = UMI-like wrist camera + external camera** (user: "UMI-like camera";
  layout asked for wrist + external for the CV stretch). Wrist = 155° equidistant fisheye,
  224×224 (UMI's resolution, user: "whatever UMI uses").
- **[DECISION] Physics dt 0.002 s → 1/600 s, substeps 50 → 60** (approved by user). Control is
  still 10 Hz, and poses are sampled every 10 substeps = exactly 60 Hz. Re-verified
  `eval_scripted.py --n 50`: 100 % success, 3.06 mm / 0.041°, 0 % collisions, 170.7 steps, wall
  force mean 13.1 N / worst 46.0 N (was 3.07 mm, 15.2 / 54.9 N). `pytest`: 12 passed.
  `media/seed0.mp4` was not re-rendered.
- **[DECISION] Images at 10 Hz, poses at 60 Hz** (approved by user). Disk has 11 GB free. A
  224² frame compresses about 4× (measured: 52.7 MB → 13.2 MB for 175 two-view frames), so
  images at 60 Hz would need about 38 GB. UMI also downsamples the camera to 10 Hz for the policy.
- **[DECISION] Wrist camera on a mast past the panel's long edge** (approved by user): the panel
  sits on the plate and blocks any view along the plate normal. Position in the plate frame:
  (0, +0.25, −0.04), looking along the approach direction, image-up = world-up at the wall. No mass
  or collision, and the mast is not modelled. First tried the −y edge: at the home pose it was 5 cm
  inside the wall (rendered through it) and 2 mm inside the ledge. Probed all four edges over 4
  episodes (clean and noisy): −y bad, while +y / +x / −x clear by ≥ 2.6 cm. Picked +y (smallest
  change; at the wall it sees the ledge and the target's bottom edge).
- **[DECISION] Fisheye rendering:** MuJoCo cameras are pinhole only. Render a 768² pinhole at
  fovy 155°, then remap to an equidistant fisheye (r = f·θ, inscribed image circle = 155°, black
  outside) at 2× supersampling with a 2×2 box filter. The pinhole has ≥ the fisheye's pixel density
  everywhere (85 vs 83 px/rad at the centre). **Simplification: no side mirrors.**
- **[DECISION] File contains only UMI streams** (user: "this is a database and shouldn't include
  our action or reward, everything should seem as if it came directly from UMI"). No action,
  reward, terminated or truncated. Sim-only per-episode metadata (seed, noise std, randomization,
  dropped/truncated, n_control_steps) goes to the sidecar `<out>_sim_meta.npz`, so the per-factor
  breakdown the user wanted is still possible without polluting the UMI file.
- **[DECISION] Failures via per-episode noise** (user): action-noise std ~ U[0, 0.2] per episode,
  drawn from `default_rng(start_seed)`, logged in the sidecar. Each episode gets a fresh
  `ScriptedController(seed=env seed)`, which also avoids the `descend_gap` carry-over bug noted in
  the viewer entry.
- **[DECISION] Layout** (user): one group per episode, `episode_XXX/<name>` keys in a single npz.
  Timestamps are on a session clock (episodes back to back, the next starting 1/60 s after the
  previous ends), with `start_time` / `end_time` scalars. Written by streaming each array into
  the zip (`NpzStreamWriter`), so RAM use is per episode. `np.load` reads it normally.

### Code
- `drywall_env/env.py`: `substeps` 60; new optional `env.substep_hook(env, i)` called after every
  substep (read-only observers). `scene.xml`: timestep 1/600.
- `drywall_env/model.py`: `wrist` camera on the plate (`WRIST_CAM_POS`, `WRIST_CAM_FOVY`).
- `drywall_env/umi_recorder.py`: `UMIRecorder`, `fisheye_maps`, `NpzStreamWriter`, schema in the
  docstring. `scripts/record_umi.py`: CLI, with a size projection after 5 episodes that aborts if it
  won't fit on disk. `tests/test_umi_recorder.py`: 60 Hz spacing, images = poses[::6], shapes,
  fisheye corners black, one attach + one release, success, npz round trip. `.gitignore`: `data/`.

### Checks
- Smoke runs (3 + 2 episodes): pose dt = 1/60 exactly, image dt = 0.1, image ts == pose ts[::6],
  attach transitions where expected, episode boundaries back to back, sidecar correct. About 5 s per
  episode, about 13 MB per episode compressed. Wrist frames viewed: floor/rack/robot at the pick,
  ledge + target edge + wall during placement, image-up = world-up.
- `pytest -q`: 12 passed.

## 2026-11-16 — UMI zarr converter + camera calibration (proposals 1 and 2)

Confirmed UMI's real layout from `scripts_slam_pipeline/07_generate_replay_buffer.py` (fetched):
zarr ZipStore; `data/robot{i}_eef_pos` (T,3) f32, `robot{i}_eef_rot_axis_angle` (T,3) f32,
`robot{i}_gripper_width` (T,1) f32, `robot{i}_demo_start_pose` / `demo_end_pose` (T,6),
`camera{i}_rgb` (N,224,224,3) u8 with JpegXl level 99 and chunks (1,H,W,3). One row per camera
frame. Episode ends are kept by UMI's ReplayBuffer in `meta/episode_ends`.

- **[DECISION] Zarr rows at 10 Hz** (approved by user): row k = 60 Hz pose sample 6k, which lines up
  with the images. UMI training config: `down_sample_steps = 1`.
- **[DECISION] `robot0_gripper_width` = attached as 0.0/1.0** (approved by user), documented as a
  suction stand-in.
- **[DECISION] JpegXl level 99 like UMI** (approved by user). New pinned deps: zarr 2.18.7 (v2 API,
  as UMI uses), numcodecs 0.15.2, imagecodecs 2026.11.11. imagecodecs names the class `Jpegxl`
  (not `JpegXl`), but its codec id is `imagecodecs_jpegxl`, the same id UMI's own codec registers.
  On a real wrist frame: 150.5 kB → 21.3 kB, mean abs error 0.25/255.
- **[DECISION] Relative trajectory = helper function only** (approved by user):
  `umi_zarr.relative_trajectory()`, T_rel = inv(T_base) @ T. UMI computes it at training time,
  so nothing derived goes in the file.
- Extras in the zarr (UMI's loader only reads its shape_meta keys): `external_rgb`, `timestamp`,
  `panel_pos_in_target`, `panel_rot_axis_angle_in_target`, `meta/episode_success`, and the camera
  calibration as JSON in the root attrs.
- ZipStore can't overwrite. Images go into pre-sized arrays chunked per frame, so each chunk is
  written once; low-dim arrays are written once at the end; attrs in a single `put`. The first
  version set two attrs separately, which gave a "Duplicate name: .zattrs" warning (fixed;
  the tests now run with `-W error::UserWarning`).
- **Calibration** (`UMIRecorder.calibration()`, written to `meta/` by `record_umi.py`, with
  `--append-calibration` for existing files): OpenCV conventions. The wrist camera is
  `opencv_fisheye` with D = 0 (equidistant), K from `fisheye_maps` (f = 82.80 px, c = 111.5 at 224
  px), and `T_ee_wrist`. The external camera is a pinhole with f = 270.39 px and `T_world_external`.
- **Bug found by the calibration test:** I first took the external pose from `scene.camera[0]`,
  which is the *left stereo eye*, offset ipd/2 = 3.4 cm. Ray-cast vs segmentation agreement was
  stuck at about 91 % whatever the f / principal-point perturbation (a parallax signature). Mono
  rendering uses the mean of both eyes, which equals `lookat − distance·forward`. After the fix:
  external 98.3 %, wrist 99.0 %, with the wrist peaking at the stored f and c in a ±1 % / ±0.5 px
  sweep. The remaining misses are object-edge pixels.
- Tests: `test_camera_calibration_matches_render`, `test_relative_trajectory` (identity at base,
  invariant to a global rigid transform), `test_convert_to_umi_zarr` (keys, shapes,
  episode_ends, values, demo start pose, JpegXl error < 2/255, success, calibration attrs).
  `pytest -q`: 15 passed.

## 2026-11-16 — Closing Part A (checked against the task doc)

- Read the task doc (General Instinct take-home). Part A is "Simulation Environment", 25 % of the grade.
  Mapped every Part A requirement to the repo; all were already covered. Three loose ends were closed
  (user: "finish part a"; Part B is being done in a separate chat).
- **Bug fix:** `ScriptedController.reset()` now restores `descend_gap` from the new
  `descend_gap_nominal`. Attach retries shrink it by 2 mm each time, so a reused controller carried a
  smaller gap into later episodes. Added `test_scripted_reset_restores_descend_gap`.
- `python -m pytest -q` → 16 passed (includes the other chat's UMI tests).
- `python scripts/eval_scripted.py --n 50`: 100 % success, 0 % drops, 3.06 mm / 0.041°, 0 % collisions,
  170.7 steps, wall force mean 13.1 N / worst 46.0 N. Identical to the dt-change entry, so no seed in 0–49
  hit the attach-retry path.
- **Video:** re-rendered `media/seed0.mp4` at the current dt (1/600 s × 60 substeps). Seed 0: success at
  step 175, 3.05 mm / 0.027°, 186 frames @ 10 fps. Checked the last frame: the panel is flush on the wall,
  over the target, on the ledge.
- **[DECISION]** (recommended in chat, user said "finish part a") No `obs_mode` flag. Instead the README
  states that policies read only `policy` / `deployable`, privileged is for critics / teachers /
  evaluation, and wrapping the whole dict (for example with `FlattenObservation`) would leak it.
- **README additions:**
  - Why one arm instead of the spec's bimanual robot.
  - Payload: the Panda is rated 3 kg, plate + the heaviest panel is 3.5 kg, and the sim does not enforce
    payload limits.
  - Cameras are not in the env observation.
  - Removed the "video recorded at old dt" note.
- Not committed (user commits on request).
- Renamed `media/seed0.mp4` → `media/scripted_seed0.mp4` (`git mv`, user request), so the clip is
  clearly labelled as the scripted controller and not a learned policy. Updated the `.gitignore`
  exception and both README references; the README caption now says "scripted controller (no
  learned policy)". Older entries above keep the old name.

## 2026-11-16 — 500-episode recording finished

- `python scripts/record_umi.py --n 500 --out data/umi_sim.npz`: **500 episodes, 74.2 % success,
  7.11 GB, 2889 s** (5.8 s/episode). Size projection after 5 episodes was 7.3 GB. Noisy failures are
  mostly 300-step truncations, not drops. Log: `data/record_umi.log`.
- The run started before the calibration code existed, so the calibration was added afterwards with
  `record_umi.py --out data/umi_sim.npz --append-calibration`. The file has all 14 `meta/` keys and
  500 episode groups.
- **Converter speed-up:** a sequential JpegXl encode at UMI's settings (level 99, default effort)
  costs about 11.7 ms per 224² image, which is about 40 min for this dataset. Benchmarked effort
  1/3: about 4 ms with error 0.39/255, but that deviates from UMI. Kept UMI's settings and parallelised
  instead: encode on a thread pool, then write each frame's chunk directly to
  `<array>/<row>.0.0.0` in the ZipStore. Smoke data: 18 s → 2 s, and the images are byte-identical
  to the sequential version (checked).

## 2026-11-16 — Dataset viewer (`scripts/view_dataset.py`)

User request: a separate script to view a zarr dataset, many simulations at once, with optional
camera views and optional state overlays.

- **[DECISION]** (given by user) Main view = **both** 2D trajectory plots (top x-y and side x-z,
  green/red by success, moving marker per episode) **and** a 3D MuJoCo replay.
- **[DECISION]** (given by user) Toolkit = **Tkinter**, no new dependencies. Pillow (already
  installed through imageio) is used for the image tiles.
- **[DECISION]** (given by user) Overlays = **attached state** (thick path runs, SUCTION badge on tiles)
  and **panel-to-target error** (marker colour scale, mm/° text on tiles).
- 3D replay design: the focus episode is re-posed with the env's DLS IK from the recorded EE pose
  (3 warm-started solves per frame), and the scene layout comes from `env.reset(seed)` using the
  sidecar seed. All selected episodes are drawn in `viewer.user_scn` as EE path lines (thick
  when attached) and translucent ghost panels, using target = nominal + sidecar
  (target_dy, target_dz), then panel = target ∘ panel_in_target. Same single-thread Tk
  pattern as `view_interactive.py` (`viewer.lock()` + `sync()`).
- Test (scratchpad `drive_dataset_viewer.py`, real Tk + real passive viewer on a 3-episode zarr):
  sidecar auto-found; Play advances about 10 steps/s; camera switch / overlay toggles; 3D replay
  opens with 646 user geoms; **IK EE error 0.0 mm** vs recorded; the replayed panel-target distance
  equals the recorded one (276.1 mm); closing the 3D window clears the replay; ALL OK.
- Bugs found while testing: (1) the wall line was drawn at x ≈ −0.05 because geom positions were
  read before `mj_forward`; fixed. (2) Stacked wide plots left the paths only about 100 px wide;
  switched to two tall panels side by side. (3) The rack outline went past the panel edge; the
  bounds now include the rack. Checked numerically from the canvas coordinates: everything is
  inside its panel.
- Process note: my first visual check used `ImageGrab` of the window region. That captured other
  windows that happened to be on top of the viewer (the user's browser). I deleted those images
  right away and switched to checking canvas item coordinates instead. Don't screen-grab.
- Not verified: how it looks and feels on screen (I can't see the display), and speed with 50
  camera tiles on the full dataset (each frame decodes 50 JpegXl images; there's a small cache).
- Full conversion: `convert_to_umi_zarr.py` → `data/umi_sim.zarr.zip`, **500 episodes, 109,077 rows
  (10 Hz), 4.14 GB, 442 s** (parallel JpegXl). Checked: all UMI keys + extras with the expected
  shapes; eef_pos equals the npz 60 Hz poses[::6] (episodes 0, 250, 499); wrist image error 0.26/255;
  success labels match; calibration in attrs; success rate 74.2 %.
- Viewer on the full zarr, 50 episodes with wrist tiles and both overlays: 167 ms/frame uncached
  (50 JpegXl decodes), 50 ms cached. So the first playback of 50 tiles runs at about 6 fps, not
  10; replays and scrubbing back are full speed. Without tiles it's plot updates only.
- `register_codecs(verbose=False)`: a second registration printed about 80 "already registered" lines.
- `pytest -q`: 16 passed.

## 2026-11-17 — UMI side mirrors + background re-record

User: "you are still missing some things from the UMI. the UMI has mirrors", with the UMI hardware
build guide (Google Doc) attached, and asked to re-record the images in the background.

- Build guide (read via the Google Docs connector): lists "Plastic Stick-on Mirror" in the BOM and
  "Install GoPro and mirrors to the mounting plate". **No mirror geometry**; that is in the Onshape
  CAD, which isn't accessible here.
- UMI code `umi/common/cv_util.py` (fetched): `get_mirror_canonical_polygon()` gives the left mirror
  image region as 6 points on the 2704×2028 GoPro frame, (540,1700) (680,1450) (590,1070)
  (290,1130) (290,1770) (550,1770); the right region is x-mirrored. Their pipeline's mirror swap
  copies the horizontally flipped frame into the mirror regions (the paper's "digital reflection").
- **[DECISION] Exact planar optics** (approved by user). New `drywall_env/umi_mirrors.py`: the
  regions are UMI's polygons in normalised coords (UMI resizes 4:3 → 224² the same way). Each mirror
  centre lies on the ray through its region centroid (62.9° off-axis), **6 cm** from the lens, with
  its normal chosen so that ray reflects forward, tilted **15° inward**. Result: mirror points at
  (±4.4, 3.0, 2.7) cm in the wrist camera frame, virtual camera centres at (±5.9, 3.0, −3.1) cm,
  **stereo baseline 11.9 cm** (the UMI gripper is 17.5 cm wide). These are assumed numbers (no CAD).
  - Implementation: two fixed virtual cameras `wrist_mirror_left/right` on the plate (model.py,
    pose = lens reflected across the plane, pinhole fovy 60.7° covering the region). The recorder
    reflects each mirror-region fisheye pixel's ray, projects it into the virtual pinhole (320²) and
    samples it. The content is physically mirror-flipped, like the GoPro. No mirror frame, tint or dirt.
  - Calibration `meta/` gains `mirror_sides`, `mirror_polygons`, `mirror_normals`, `mirror_points`,
    `mirror_virtual_origins`. Schema → `umi-sim-2`.
- **[DECISION] Mirror swap** (user: "swap but make a note"): the npz stays **raw**, and the zarr
  converter applies UMI's mirror swap to `camera0_rgb` by default (`--no-mirror-swap` to disable;
  root attr `mirror_swap`). **NOTE: inside the two mirror regions, zarr `camera0_rgb` ≠ npz `wrist_rgb`.**
  Documented in README, the converter docstring and here.
- Test `test_side_mirrors_are_planar_reflections`: for 200 random pixels per mirror, intersect the
  ray with the stored mirror plane, ray-cast the reflected ray in MuJoCo, and compare with the
  composited segmentation image. The first run got 94.5 %. Full diagnosis over all 2058 pixels: all
  94 misses were pixels *outside the fisheye image circle*, because UMI's polygons come from a 4:3
  frame and **4.6 %** of each falls outside our inscribed 155° circle (rendered black). Inside
  the circle every sampled pixel matches. No occluder-behind-mirror cases.
  The converter test was updated to expect the swapped image. `pytest -q`: **17 passed**.
- **[DECISION] Re-record to new files, verify, then replace** (approved by user):
  `record_umi.py --n 500 --out data/umi_sim_mirror.npz` running in the background (log
  `data/record_umi_mirror.log`). After it finishes: check that poses, attach state and success
  labels are identical to `data/umi_sim.npz` (the sim is deterministic per seed), then replace the
  old npz and re-convert the zarr.
- Re-record finished: **500 episodes, 74.2 % success, 7.10 GB, 3248 s** (6.5 s/episode; the two
  mirror renders add about 0.7 s/episode).
- Verification against the old file, all 500 episodes: timestamp, ee_pos, ee_quat, attached,
  panel_pos/quat_in_target, image_timestamp, start/end_time and success are **bit-identical**;
  sidecar files identical. `wrist_rgb` changed in 7.8 % of pixels (the mirror regions). Unexpected:
  `external_rgb` differs too, but only by ±1/255 in about 0.003 % of pixels, scattered: GPU
  rasterization noise from the extra rendering context, not a content change.
- Replaced: `data/umi_sim.npz` (+ sidecar, `record_umi.log`) is now the mirror version (schema
  umi-sim-2), and the old version was deleted. Zarr rebuilt from it with mirror swap (default).
- Zarr rebuilt: 500 episodes, 109077 rows, 4.21 GB, 384 s; verified mirror_swap=True, camera0_rgb matches mirror_swap(npz wrist_rgb), poses match.

## 2026-11-17 — Plan for Part B (conversion, BC) and Part C (offline RL)

Plan (agreed in chat): one shared state + transition format for B and C, so the BC trainer, IQL
trainer and evaluator are written once. Order: transitions.py → umi_to_transitions.py (B.B) →
train_bc.py (B.C) → eval_policy.py + policy viewer → generate_offline_data.py (C.A) →
train_iql.py (C.B) → comparison (C.C).

- **[DECISION]** (approved by user) **29-D low-dim state**, no images: the env's 43-D `policy` vector
  minus joint_pos / joint_vel, because UMI has no robot joints. Layout: ee_pos 3, ee_quat 4, attached 1,
  panel_pos 3, panel_quat 4, target_pos 3, target_quat 4, panel-in-target pos 3 + quat 4. All in the base
  frame, quats wxyz with w ≥ 0. The same function builds it from a live env obs and from UMI data.
- **[DECISION]** (approved) **Target pose from the sim sidecar** (nominal + target_dy/dz), documented as a
  stand-in for a perception-detected target (for example fiducials). Panel pose = target ∘ panel_in_target.
- **[DECISION]** (approved) **Conversion details:**
  - General `T_base_world` transform. It is the identity here; the test uses a non-identity transform.
  - Timestamp resampling to the control rate: linear interpolation for position, slerp for rotation.
  - Actions = base-frame pose deltas t → t+1, scaled by 2 cm / 3° and clipped to [-1, 1]; clip rate reported.
  - Attach bit = +1 if attached at t+1.
  - Workspace-box violations flagged.
  - Validation: replay the derived actions in the env with the recorded seeds and compare success to 74 %.
- **[DECISION]** (approved) **Rewards from UMI data use only the computable terms:** pos_error,
  ori_error, action, jerk, success. collision, contact_force and drop are 0 and documented (UMI has no
  force or contact sensing). Every term is stored separately, so relabelling is a sum. For the combined
  UMI + sim policy, the sim data is relabelled with the same subset.
- **[DECISION]** (approved) **Transition file:** D4RL-style `.npz`: observations, actions, rewards,
  next_observations, terminals, timeouts, episode_ends, per-term rewards, plus a small meta describing
  the state layout. terminal = True on the last step of successful episodes; failed episodes end as
  timeouts.
- **[DECISION]** (approved) **PyTorch with CUDA** (RTX 4070) as a new pinned dependency.
- **[DECISION]** (approved) **IQL implemented from scratch** in one file, so the user can follow it,
  with docstrings citing Kostrikov et al. 2021 (arXiv:2110.06169) and the equations in each loss. CORL
  is used only as a reference and is credited.
- **[DECISION]** (approved) **Seed allocation:**
  - 0–499: UMI recording, the BC training data.
  - 0–49: also eval_scripted and the tests (scripted controller only).
  - 1000–1999: Part C offline data.
  - 5000–5049: validation, only if used.
  - 10000–10099: held-out test set, used only for the final comparison and the policy viewer's default
    batch.
  - Data scripts refuse seeds in the test range.
- **[DECISION]** (user request) **Interactive policy viewer** for the B and C evaluations: run a single
  episode or many consecutive test episodes with each final model.
- **[DECISION]** (user request) **Part C training charts** in `media/`, made with matplotlib (new
  dependency) and descriptive (titles, axes, captions).
- Open questions asked in chat before implementation: see the next entry.

### Answers to the open questions (2026-11-17)
- **[DECISION]** (user) The other chat finished only B.A (`data/umi_sim.zarr.zip`). Everything from B.B on
  is built from this chat.
- **[DECISION]** (user) Policy viewer = new `scripts/view_policy.py` (design (a)): a 3D MuJoCo window + a
  Tk panel. Single test: choose a model and seed, then Play / Step / Reset with a live readout. Many tests:
  N consecutive test seeds, watched or fast, with a results table; clicking a row replays it in 3D.
- **[DECISION]** (user) **Periodic 50-episode validation during training, on seeds 5000–5049.** Why:
  to plot success rate against training step, so the charts show whether the losses translate into
  behaviour. Why this is still offline RL: the validation rollouts are never added to the dataset, never
  used in a gradient update, and never used to choose a checkpoint (the final model is always the last
  step). They are measurement only, as with a held-out evaluation in supervised learning. The doc's rule
  ("must learn from previously collected data without gathering additional environment interactions
  during its offline optimization phase") concerns what the learner trains on, which stays the fixed
  dataset. The final comparison uses the separate test seeds 10000–10099.
- **[DECISION]** (user) Training charts for both B (BC) and C (IQL) in `media/`, matplotlib.
- **[DECISION]** (user) Budget: tonight until tomorrow afternoon (2026-11-18).
- **[DECISION]** (user) Commit at the end of each phase from the console as the user, with **no Claude
  co-author line** in the message.

### State representation change: quaternions → 6D rotations (2026-11-17)
- **Finding:** over one scripted episode (seed 0, 175 steps) the EE quaternion in the observation flips
  sign **10 times**. In the tool-down pick orientation w ≈ 0, so `canonical()` (w ≥ 0) alternates between
  q and −q, for example [0, −0.707, 0.707, 0] and its negative. Same rotation, but an input jump of norm 2
  for a network, right where the grasp needs precision. Panel and target quats are fine (|w| ≥ 0.5).
- **[DECISION]** (approved by user, option (a)) **Every rotation in the learning state is the 6D
  representation**: the first two columns of the rotation matrix (Zhou et al., "On the Continuity of
  Rotation Representations in Neural Networks", CVPR 2019). It is continuous with no sign ambiguity, and it
  is what UMI uses for policy inputs (`rotation_6d`). State: **37-D** = ee_pos 3 + ee_rot6 6 +
  attached 1 + panel_pos 3 + panel_rot6 6 + target_pos 3 + target_rot6 6 + panel-in-target pos 3 +
  rot6 6. The rest of the plan is unchanged.
- The rejected option (b), a sign relative to a fixed reference quaternion, only moves the discontinuity
  180° away from the reference.
- The env's own 43-D `policy` vector (Part A) has the same flaw. We don't train on it; noted in the
  README limitations.

## 2026-11-17 — Part B.B: UMI → robot transitions; Part B.C: behaviour cloning

### Setup
- `pip install torch --index-url https://download.pytorch.org/whl/cu128` → torch 2.11.0+cu128, CUDA
  available (RTX 4070). `pip install matplotlib` → 3.11.2.

### Code
- `drywall_env/transitions.py`: the 37-D state (`build_state`, `state_from_obs`, deployable obs only),
  batched rotation helpers (6D rotation, Rodrigues, slerp resampling), `actions_from_poses`,
  `umi_reward_terms`, the D4RL-style file format (`save_transitions` / `load_transitions` with reward
  relabelling), and the seed allocation (`TEST_SEEDS`, `VAL_SEEDS`, `OFFLINE_SEEDS`,
  `check_not_test_seeds`).
- `drywall_env/umi_to_robot.py`: the conversion, in order: world → base (`T_base_world`), resample to the
  10 Hz control rate, absolute → relative (pose deltas), gripper → attach bit, workspace check. Also
  `replay_actions` for validation.
- `scripts/umi_to_transitions.py` → `data/umi_transitions.npz`.
- `tests/test_transitions.py` (9 tests):
  - rot6 round trip; resampling exact + 45° slerp midpoint.
  - Derived actions integrate back to the next pose.
  - **UMI rows → states equal the env's own states**, also with a rotated and translated "SLAM map" frame,
    so the world → base transform is really exercised.
  - UMI reward terms equal the env's terms; test seeds refused; file round trip + relabel.

### Results
- Conversion check (scratchpad `check_convert.py`): states rebuilt from the zarr equal a re-run of the same
  episode in the env (seed + noise from the sidecar) to **1.2e-7** (float32). Open-loop replay of the derived
  actions on 20 episodes: lag 0 → 14/20 successes (recorded 14/20), outcome agreement 18/20, EE tracking
  error 7.1 mm. Lag 1 → 8/20 agreement, 13.0 mm. **[DECISION] lag = 0** (no latency compensation): the env
  servo tracks within the step, so shifting the targets only adds error.
- `python scripts/umi_to_transitions.py --replay 500` (265 s): **500 episodes, 108,577 transitions,
  20.9 MB**.
  - 0.89 % of actions clipped to the per-step limit; 0 EE rows outside the workspace.
  - Full open-loop replay: 73.4 % success vs 74.2 % recorded, outcome agreement 87.2 %; 90.8 % of the
    recorded successes also succeed on replay; mean EE tracking error 7.2 mm.
  - Disagreements are expected: open-loop replay of *measured* poses cannot reproduce contact events
    exactly, for example a noisy demo that grazed the ledge.
- **BC** (`scripts/train_bc.py`): filtered BC on the successful demos (371 episodes: 334 train / 37
  held-out for the validation loss, 63,644 train transitions). MLP 3×256 + LayerNorm, AdamW 3e-4 with cosine
  decay, batch 512, 30k steps, about 7 min on the GPU with the rollouts.
  - **Single-step (H = 1)**: validation rollouts 0 %, 0 %, 8 % at 5k / 10k / 15k steps; the EE ends about
    0.7–0.8 m from the target.
  - **Action chunking (H = 8, execute 4 then re-plan)**: 2 %, 58 %, 88 %, 86 %, 80 %, **84 %** at 5k–30k.
  - **[DECISION]** (the time-boxed fallback in the approved plan) **The Part B policy is `bc_umi_h8`**,
    chosen on validation seeds 5000–5049, never on test seeds. Both variants are still reported on the test
    seeds.
  - Why chunking helps so much here: single-step BC suffers compounding error, and phase ambiguity in
    particular: the same state appears in several phases (hovering vs. starting to descend). Predicting 8
    future steps gives the network a commitment to a motion segment, as reported in ACT (Zhao et al. 2023)
    and Diffusion Policy (Chi et al. 2023).
- Smoke tests: IQL 20k steps on UMI + 8 sim episodes runs end to end (Q and V still rising at 20k:
  value propagation over about 200-step episodes needs more steps). With β = 3 the advantage weights stay
  around 1.0–1.4 (A std about 0.15 at reward scale 0.1), so the actor would be close to BC. **[DECISION]**
  Train both β = 3 (paper, locomotion) and β = 10 (paper, AntMaze) and report both on the test set,
  with no hidden selection.

## 2026-11-17 — Part B results on the test seeds, failure analysis, policy viewer

- `python scripts/eval_policy.py --policy scripted checkpoints/bc_umi_h8.pt` (test seeds 10000–10099,
  100 episodes each): scripted 100 % (95 % Wilson CI 96–100), 3.1 mm / 0.04°, 0 % collisions, 171 steps.
  **bc_umi_h8 80 % (CI 71–87)**, mean 132.7 mm / 14.6° (dominated by the failures), 9 % collisions,
  196 steps. Videos with labels burned in: `media/bc_umi_h8_success.mp4` (seed 10000) and
  `media/bc_umi_h8_failure.mp4`.
- **BC failure analysis** (all 20 failures are 300-step time-outs; traced with a per-step attach log):
  - 16 × **never grasped** (seed 10010): the policy descends to about 9 mm above the panel face and stops
    (|a| ≈ 0.05 for 250 steps), never commanding suction. Demos switch suction on at about 3 mm. The env
    would accept ≤ 30 mm, but the policy never asks. A BC fixed point: near the goal the demo motions are
    tiny, so the regressed motion reaches ≈ 0 before the state where the attach bit flips.
  - 4 × **released early** (seed 10038): the panel is carried to the target and released about 7 mm short of
    the wall. It lands on the ledge, but the fixture needs ≤ 2 mm, so it isn't placed (final error 5–9 mm,
    0°).
  - Common cause: the last few millimetres before a discrete suction event. This is what IQL should improve,
    because the reward separates "touching" from "7 mm short".
- **Held-out loss vs. behaviour:** for bc_umi_h8 the held-out attach BCE rises after about 3k steps
  (overfitting the suction timing) while validation success rises to 84–88 %, so validation rollouts are
  needed.
- **IQL observation:** the β = 3 and β = 10 runs have identical critic curves. That is correct for IQL: V and
  Q are trained only on dataset (s, a), never on π, so β affects only the actor. A shared critic would halve
  the cost (not done, to keep runs independent).
- **Policy viewer** `scripts/view_policy.py` tested with scratchpad `drive_policy_viewer.py` (real Tk + real
  passive viewer):
  - 3 Steps → step 3; Reset clears.
  - Fast batch scripted × 3 → all success, 3.1 mm. Fast batch bc_umi_h8 × 5 → 5/5 success, 3.0–3.8 mm.
  - Summary line per model; click row 0 → watched replay at 8x ends SUCCESS (the replay adds a row).
  - Stop mid-batch → stops, batch empty. Close → ALL OK.
  - Not verified: look and feel on screen.
- `tests/test_policies.py` (5 tests): chunk padding never crosses episodes, actor output range +
  checkpoint round trip, receding horizon re-plans every n_exec steps, expectile loss values + its
  minimiser lies above the mean, Wilson interval. With test_transitions: 14 passed.
- `requirements.txt`: + matplotlib 3.11.2, torch 2.11.0 (cu128 extra index). `.gitignore`: `!media/*.mp4`
  (small labelled clips are part of the submission); `data/` stays ignored (to discuss with the user).
- Single-step BC on the test seeds: **23 % (CI 16–32)**, 619 mm / 68.6°, 2 % collisions, 272 steps. Mostly
  never grasps.

## 2026-11-17 — Part B committed; Part C control run

- **Commit `76868be`** "Part B: …" as the user (author and committer Bruce Peters, no co-author trailer;
  checked with `git log -1`). It includes the Part A wrap-up and the other chat's B.A work, which had never
  been committed. Part C files (generator, IQL, test_iql.py, IQL runs) were left out of it.
- The IQL expectile test moved to `tests/test_iql.py` so the Part B commit doesn't import train_iql.py.
- **[DECISION]** (added during the overnight run, within the approved plan's "honest comparison") **Control
  run `bc_umi_sim_h8`:** filtered BC with the same network and settings on the *successful* episodes of
  UMI + sim data (943 episodes: 849 train / 94 held out, 162,488 transitions). It separates "IQL helps because
  of RL (advantage weighting, use of failures)" from "IQL helps because it saw more data".
- `transitions.load_many(paths, reward_terms)`: concatenates files, re-bases episode_ends, refuses test
  seeds and mixed reward definitions unless relabelled. Used by both train_bc.py (`--data` takes several
  files, rewards relabelled to the UMI subset; BC ignores them anyway) and train_iql.py (replaces its local
  loader). New test `test_load_many_concatenates_and_checks_reward_definitions`.
- IQL validation at 20k / 200k steps: sim β3 38 %, sim β10 30 %, umi+sim β3 28 %, umi+sim β10 26 %.
- **Control `bc_umi_sim_h8`** (filtered BC, UMI + sim successes): validation 40 / 64 / 74 / 74 / 76 / 74 % at
  5k–30k. **Test: 69 % (CI 59–77)**, 217 mm / 24.4°, 2 % collisions, 212 steps. *Lower* than BC on the UMI
  demos alone (80 %): the extra sim successes come from noisier behaviour (action noise up to 0.3, BC
  rollouts), so more data of lower quality hurts plain BC.
- **Automatic failure classification** (`evaluation.classify_failure`), from each episode's end state:
  dropped / never_grasped / held_at_end / released_off_wall / released_misaligned / other. Each episode now
  also records `ever_attached`, `attached_at_end` and `wall_dist_mm`; `summarize` counts the modes;
  eval_policy prints them; the comparison chart gets a stacked failure-mode bar. Checked against the
  hand-traced BC episodes: 10000 → success, 10010 → never_grasped, 10038 → released_off_wall (7.2 mm). The
  final comparison re-runs every policy so all results have these fields.
- Validation noise: with 50 episodes a single validation point has a standard error of about ±7 % (95 %
  interval about ±14 %), which explains IQL jumps such as sim β10 78 % → 54 % between 40k and 60k. The final
  numbers use 100 test episodes with Wilson intervals.

## 2026-11-17/18 — Part C: offline data, IQL, comparison

### C.A data
- `python scripts/generate_offline_data.py --bc checkpoints/bc_umi_h8.pt` (22 worker processes, 115 s):
  1,000 episodes on seeds 1000–1999, 234,717 transitions, 44.4 MB.
  - Behaviour per episode from the seed: scripted (477 episodes, 55.3 % success, 2.5 % drops, collisions in
    20 % of episodes) or noisy BC (523 episodes, 58.9 %, 5.9 % drops, collisions in 63 %).
  - Action noise std ~ U[0, 0.3] on both. Full env reward, terms stored separately. Overall 57.2 % success.

### C.B IQL (scripts/train_iql.py, from scratch; equations and citations in the docstring)
- Four runs, 200k steps each, in parallel on the RTX 4070: about 57 steps/s each, about 62 min each including
  10 × 50 validation rollouts.
  - Same actor as Part B (chunk 8 / execute 4), τ 0.7, γ 0.99, reward × 0.1.
  - β ∈ {3, 10} × data ∈ {sim (full reward), UMI + sim (UMI-subset reward, both relabelled)}.
- As expected, β does not change the critics (identical Q/V curves for β 3 and 10).
- Validation success (50 episodes, seeds 5000–5049) swings a lot, e.g. umi+sim β10 at 100k–200k:
  82 / 86 / 94 / 76 / 70 / 68 %. Final step only (no checkpoint picking).
- **Value diagnostics:** mean discounted return-to-go of the data (×0.1): UMI + sim −1.03 (successes +1.35,
  failures −3.69); sim −2.29. Learned mean V: UMI + sim rose to ≈ +1.0 at 45k and drifted down to ≈ +0.15 at
  200k; sim ≈ −1.3. V above the data mean is expected (upper expectile ≈ a better-than-behaviour policy), but
  the steady drift means **the critic had not converged at 200k** (the paper uses 1M).

### C.C Final evaluation (test seeds 10000–10099, 100 episodes each, all policies re-run with failure modes)
- scripted 100 % (96–100). bc_umi 23 % (16–32; never grasped 76).
- **bc_umi_h8 80 % (71–87)**: never grasped 16, released off wall 4; 9 % collisions.
- bc_umi_sim_h8 69 % (59–77): never grasped 27, off wall 4; 2 % collisions.
- iql_sim_b3 58 % (48–67): grasp 14, held 12, off wall 14, misaligned 2; 22 % collisions.
- iql_sim_b10 61 % (51–70): grasp 5, held 10, off wall 21, misaligned 3; 14 %.
- **iql_umi_sim_b3 71 % (61–79)**: grasp 9, held 6, off wall 14; 8 %.
- iql_umi_sim_b10 68 % (58–76): grasp 3, held 6, off wall 22, misaligned 1; 12 %.
- No drops on the test seeds. Mean lengths 171 / 196 (BC H8) / 210–231 (IQL).

### Findings (README "Part C"): IQL ≈ BC (not better)
- RL removes the grasp fixed point (never grasped 16–27 → 3–9) but adds release failures (held at end, or
  released > 2 mm off the wall).
- Combined data > sim only for IQL. For plain BC, adding noisy sim successes hurts (80 → 69 %).
- IQL collides more.
- Hypotheses, not verified: unconverged critic, sparse release reward, first-action chunk weighting, almost
  no reward difference between "touching" and "5 mm off" while holding.

### Plots and fixes
- `media/training_*.png` for all 7 runs and `media/comparison.png` (success with CI, per-episode error
  scatter, collisions, length, stacked failure modes).
- Fixes: step axis shown as "50k"; a Wilson error bar at 100 % came out as −1e-14 and crashed matplotlib
  → clipped at 0.

## 2026-11-18 — Part C committed; preparing a harder challenge

- **Commit `0f50f29`** "Part C: …" as the user (no trailers). Before it, the IQL failure video was re-rendered
  with a new `eval_policy.py --failure-kind` option, so it shows IQL's dominant failure (released_off_wall,
  test seed 10009, 5 mm off target but > 2 mm from the wall) instead of the first failure (a never_grasped
  episode).
- **User proposal:** make the task harder, with many different rack positions around the robot and placement
  positions on the wall, re-collect the UMI data, and give the scripted "demonstrator" large variance in its
  setpoints. My analysis and open questions are in chat; nothing on the new task is built yet.
- **[DECISION]** (user) Bundle the current experiment as `easy_challenge`: `git mv` of everything in `runs/`,
  `results/`, `checkpoints/` and `media/` into `<dir>/easy_challenge/`, except the Part A demo
  `media/scripted_seed0.mp4` (user: don't move it).
  - README paths updated; the Part B/C commands now pass `--results-dir` / `--out-dir` explicitly so
    re-running them writes back into easy_challenge.
  - `training_utils.default_log_path`: logs follow the checkpoint's subfolder
    (`checkpoints/hard_challenge/x.pt` → `runs/hard_challenge/x/log.jsonl`).
  - `view_policy.py` finds `checkpoints/**/*.pt` and labels models `easy/<name>` / `hard/<name>`.
  - `pytest -q`: 32 passed. Not committed yet (it goes with the hard-challenge phase).

### Hard challenge: decisions (user, 2026-11-18)
- **[DECISION]** (user) Racks on a **full ring around the robot, including behind it**. Physical limits are
  measured with a reachability sweep: Panda joint1 is ±166°, so directly behind may be excluded, and racks
  can't overlap the wall.
- **[DECISION]** (user) **UMI demos cover only a limited part** of the new distribution. Sim offline data
  and the tests cover the full distribution, so we can measure whether sim data extends the policy beyond the
  demos (the task doc's central question). Two test sets: in-demo-distribution and out-of-demo-distribution.
- **[DECISION]** (user) The scripted controller gets **two modes**: `demo` (moderate per-episode setpoint
  variance; a competent but varied human stand-in, mostly successes) and `explore` (large setpoint variance
  plus noise; many failures), for the UMI demos and the sim offline data respectively.
- **[DECISION]** (user) **The state may contain nothing the UMI pipeline couldn't get.** The added feature is
  the panel pose relative to the EE, computed only from the EE pose (forward kinematics / UMI SLAM pose) and
  the panel pose *estimate*, both already in the deployable state, re-expressed in the hand frame. State
  37-D → 46-D. Easy-challenge checkpoints keep the 37-D state (the version is chosen from the network's input
  size).
- **[DECISION]** (user) Wider placement positions on the wall, sized by the reachability sweep.

## 2026-11-18 — Hard challenge: environment, scripted controller, data pipeline

### Reachability sweep (scratchpad `reach_sweep.py`, static DLS IK, 4 joint7 seeds, joint margin ≥ 0.05 rad)
- **Pick (tool-down at z 0.214 + hover 0.314):** racks at **0.6–0.7 m** are reachable all around the robot up
  to ±165°, for 10–11 of 12 panel yaws. 0.4–0.5 m fails behind the robot. The unreachable yaws are always a
  band **+15°…+75° relative to the radial direction** (wrist limit).
- **Place (EE facing the wall at x = 0.74 and the 10 cm pre-align at 0.64):** reachable only in a band of
  **z ≈ 0.5–0.75 m**, about ±0.25–0.3 m sideways at z 0.6–0.65, narrowing to ±0.15 m at 0.75; nothing above
  0.8 m.
- User asked whether the panel and target poses come from perception: no. They are sim truth (+ optional
  noise, default 0), a stand-in for perception (documented). **[DECISION]** (user) **no perception noise**
  for the hard challenge.
- **[DECISION]** (user) "Allow the placement zone and pickup zone to appear anywhere around the robot" →
  the wall itself rotates around the robot.

### Environment (`EnvConfig(challenge="hard", region=...)`; `challenge="easy"` is bit-identical to before:
seed 0 still succeeds at step 175, 3.05 mm, return −27.86; env tests pass)
- **Wall:** angle θ ~ U(−150°, 150°). The wall, target and ledge are the easy poses rotated by θ about the
  base z.
- **Target on the wall:** dy ~ U(±0.20) m, centre height ~ U(0.55, 0.75) m.
- **Rack:** angle φ = θ ± U(60°, 150°) with |φ| ≤ 160°, radius U(0.60, 0.70) m, rack x axis radial.
- **Panel:** ±5 cm on the rack; yaw relative to radial ~ U(−135°, −15°) (avoids the wrist band).
- **Regions:**
  - `demo`: θ ∈ ±45°, rack 60–120° to the wall's right (contains the easy layout).
  - `ood`: the full distribution minus `demo`.
  - `full`: everything.
- **Other:** max_steps 500 (long swings); workspace box ±0.85 m; overhead + wall-relative render cameras;
  overhead external camera in the UMI recorder.
- **Posture-aware IK (hard only):** the null-space home posture turns joint1 towards the commanded EE azimuth.
  Without it, the arm reached far-from-0° walls in a twisted posture and joint6 hit its 3.75 rad limit during
  contact (seeds 0/13/21: IK residual 30–110 mm, mid-air release, drop).

### Scripted controller (`drywall_env/scripted_hard.py`, `make_scripted(cfg, mode, seed)`)
- **Wall frame:** everything after the pick is computed in the wall frame (θ from the target-pose estimate),
  so the easy rotate → advance → pre-align → contact → release logic carries over.
- **New swing_out / swing_in phases:** arcs around the base (pursuit waypoint 0.12 rad ahead). The panel is
  carried flat, long side tangential.
- **Modes** (setpoints sampled per episode from the controller seed):
  - `demo`: moderate variance, small AR(1) noise ≤ 0.03.
  - `explore`: large variance, AR(1) noise ≤ 0.2 (×0.3 in precision phases), random arc detours, early
    suction attempts, release delays, negative push depths / ledge clearances (realistic failures).
- **Iterations:**
  1. First try: demo 75 % (12 episodes). Arm hit the wall during swing_out (swing at the rack's radius); the
     carried panel hit the wall during swing_in (corner reach √((r + 0.2)² + 0.3²) > 0.75 for r > 0.47);
     joint6 limit when rotating at low target heights.
     → swing radius demo 0.40–0.45 m (explore 0.36–0.52); rotate at z ≥ 0.68 then move down.
  2. 75 % / 24 episodes with drops → posture-aware IK → 82 % / 40.
  3. Panel released 2.1–3.1 mm off the wall (stall rule fired while the arm was still creeping)
     → stall needs 6 steps unless within 1.5 mm → **demo 90–93 % (full), 98 % (demo region)**.
  4. explore 28 %: stuck wobbling in prealign until time-out → noise ×0.3 in precision phases, max 0.2,
     60-step phase time-outs → **explore 55 %** (similar to the easy offline data's 57 %).

### Data pipeline (state v2, 46-D)
- **[DECISION]** (user) State v2 = v1 + panel pose in the EE frame (from the EE pose and the panel
  *estimate* only). `TorchPolicy` picks the state version from the network's input size, so the easy
  checkpoints are unaffected.
- **record_umi.py:** `--challenge hard --region demo --mode demo`; the sidecar stores the full target pose
  per episode (the wall rotates now).
- **umi_to_transitions.py:** `--challenge/--region/--state-dim`; uses the sidecar target pose.
- Smoke: 2 episodes → zarr → transitions → replay 2/2 success, EE tracking 9.3 mm.
- **Seeds:** UMI 20000–20499 (demo region), sim 21000–22999 (full), validation 25000–25049 (full),
  **test in-demo 30000–30099 (region demo), test out-of-demo 31000–31099 (region ood)**. All test ranges
  are refused by the data scripts.
- **[DECISION]** **Hard sim data = scripted `explore` mode only** (2,000 episodes), not a BC mix as in easy.
  It is exactly the user's spec ("large variance in its setpoints") and it doesn't wait for BC, so it runs in
  parallel with the UMI recording.
- **Training / eval / viewer:** `--challenge hard` everywhere. Eval has `--test in|ood`. The viewer has a
  region selector and lists only that challenge's checkpoints. Hard viewer drive: scripted 4/4 on ood + demo
  seeds.
- **Running now:** `record_umi.py --challenge hard --region demo --mode demo --n 500 --start-seed 20000`
  → data/hard/umi_hard.npz, and `generate_offline_data.py --challenge hard` → data/hard/offline_sim_hard.npz.
- **Hard sim data:** `generate_offline_data.py --challenge hard --workers 20` (240 s): 2,000 explore episodes,
  **649,725 transitions, 63.9 % success, 10.4 % drops, collisions in 40.5 % of episodes**, 173 MB (over
  GitHub's 100 MB file limit: can't be committed as is).
- **Started IQL on hard sim data** (no UMI needed): `train_iql.py --challenge hard` with the same settings as
  easy (200k steps, β 3 and 10, chunk 8 / execute 4, τ 0.7, reward × 0.1), so differences come from the
  task, not the tuning. Validation: seeds 25000–25049, full region.
- **Part D draft:** `docs/deployment_proposal.md` (sim-to-real, closed-loop control, failure handling, field
  deployment). Main recommendation from our results: the learned policy does gross motion and carrying, and a
  force-controlled, verifiable routine does the last centimetres and the release (where every learned policy
  failed).
- UMI recording is the critical path: 38 / 500 after 393 s (slowed while the sim data generated in parallel).
- **[DECISION]** (user) **Data files are not committed;** the README documents how to regenerate them
  (section "Datasets", exact commands, seeds and run times for easy and hard). `data/` stays git-ignored.
  Checkpoints, logs and results are committed.
- **Hard scripted demo video** (user request): `run_scripted.py` gained `--challenge/--region/--mode/--label`
  (easy path unchanged: seed 0 still 3.05 mm, return −27.86). The label helper moved to `drywall_env/video.py`
  (shared with eval_policy.py).
  `python scripts/run_scripted.py --challenge hard --seed 1 --video media/hard_challenge/scripted_hard_seed1.mp4`
  → seed 1 (wall 4°, rack behind the robot at 149°): **success at step 258, 3.06 mm / 0.02°, no collision**,
  269 frames, caption "Scripted controller (no learning) | hard challenge, demo mode | seed 1 | SUCCESS".
  Frames checked: swing out to the rack behind, grasp, flat carry around the base, panel flush on the target.
- **First hard IQL (sim only) results, before restart:** validation success **0 % at 20k / 40k / 60k** for both
  β (collisions 24–56 %, panel ≈ 0.9–1.1 m from the target). The hard task is much harder for offline RL on
  noisy explore data. Charts were rendered from these partial logs (overwritten by the restarted runs).
- **Snapshots** (user: "can I view how the RL model is doing at different validation steps?"):
  - train_bc.py / train_iql.py now save the actor at every validation rollout to
    `checkpoints/<challenge>/snapshots/<run>/step_XXXXXX.pt` (git-ignored, about 0.6 MB each).
  - **Viewing only:** the reported model is still the final step, so there is no checkpoint picking.
  - `view_policy.py --snapshots` lists them as `hard/<run>@<step>`.
  - The two hard sim IQL runs were stopped at about 60–70k steps and **restarted from scratch with snapshots**
    (no cost on the critical path: the UMI-dependent runs wait for the recording anyway).
- **Hard UMI recording:** `record_umi.py --challenge hard --region demo --mode demo --n 500 --start-seed 20000
  --max-noise 0` → **500 demos, 99.8 % success, 6.67 GB, 3030 s**.
- **Zarr:** 500 episodes, 109,760 rows, 4.01 GB, 497 s.
- `umi_to_transitions.py --challenge hard --region demo --replay 500` (544 s): **109,260 transitions, 28.4 MB**.
  3.3 % of actions clipped (faster swings than easy); 0 outside the workspace.
  **Open-loop replay: 99.6 % vs 99.8 % recorded, outcome agreement 99.4 %, EE tracking 8.6 mm.**
- **Started the remaining hard runs** (all with snapshots, `--challenge hard`, chunk 8 / execute 4):
  - bc_umi_h8 (BC on UMI demos);
  - iql_umi_sim_b3 / _b10 (UMI + sim, rewards relabelled to the UMI subset);
  - bc_umi_sim_h8 (control: BC on UMI + sim successes).
  The two iql_sim runs (restarted) continue in parallel.
- **[DECISION]** No single-step BC on the hard challenge: the easy experiment already settled chunking
  (23 % vs 80 % test), and the hard runs are slower (longer episodes in validation).
- `view_policy.py`: wider model box (72 chars), readout and table column (user request), so snapshot paths
  fit.

### Hard challenge: reward change (success × 10)
- **User observation:** the hard IQL snapshots seem to keep holding the panel close to the goal instead of
  releasing it.
- **Measured** (20 validation episodes, step-140k snapshots of iql_sim_b3 / b10): success 20 % / 20 %;
  **held_at_end 9 / 10** was the dominant failure.
  - Only about 3 of the 19 held episodes are actually near the goal (19–36 mm).
  - Most hold the panel pressed against the wall in the wrong place (wall gap ≈ 0, 100–550 mm in-plane),
    or still in the air.
- **Why the reward invites this:** with success +100, the discounted success from about 150 steps out is
  100 · 0.99^150 ≈ 22, worth only about 70 steps of 0.3 m position error. Releasing in the data is risky
  (drop −50, off-wall release = 0) while holding is safe, so the critic prefers holding.
- **[DECISION]** (user edited env.py, then approved the plan):
  - **Hard success weight 1000** via `EnvConfig.hard_success_weight`. **Easy stays at 100** (applied in
    `__post_init__` only for `challenge="hard"`), so the easy results stay reproducible. User's original
    edit (global 1000) adjusted to this.
  - The stored rewards are baked in at data generation, so **the hard data was relabelled by regenerating**:
    - `generate_offline_data.py --challenge hard` (452 s; same trajectories, 63.85 % success,
      success term max 1000, 1277 successes);
    - `umi_to_transitions.py --challenge hard` without replay (6 s; 499 successes at 1000).
  - **Stopped** iql_umi_sim_b3 / b10 (minutes in, old labels) and deleted their partial outputs.
  - **Kept** the two original iql_sim_b3 / b10 runs (success 100) running; they are renamed to `*_s100`
    when they finish = the reward ablation.
  - **Started** iql_sim_b3_s1000, iql_sim_b10_s1000, iql_umi_sim_b3_s1000, iql_umi_sim_b10_s1000. BC runs
    (bc_umi_h8, bc_umi_sim_h8) are unaffected (BC ignores rewards).
  - Expectation: this targets "near the goal but won't release"; it doesn't directly fix "pressed against
    the wall in the wrong place".

### Hard challenge: first results
- iql_sim_b3 / b10 (success 100), now renamed `*_s100`: final validation 4 % (b10, collisions 70 %). b3: see
  runs/hard_challenge/iql_sim_b3_s100.txt.
- bc_umi_h8 (demos from the demo region only): validation (full distribution) 0–2 %, collisions 70–78 %.
  bc_umi_sim_h8 (control, full coverage): 22–28 %. Coverage effect visible already in validation.
- **Test results** (100 episodes each; in = 30000–30099 region demo, ood = 31000–31099 region ood):
  - scripted (demo mode): **in 99 % (95–100), ood 92 % (85–96)**.
  - bc_umi_sim_h8: **in 55 % (45–64), ood 20 % (13–29)**; failures dominated by **held_at_end (36 / 46)**.
- **Finding:** BC (no reward at all) shows the same "held at end" failure, so for BC it can't be reward
  hacking: the policy fails to make the release decision (imitation / timing). The IQL holding failure is
  therefore probably partly this too; the success 100 vs 1000 ablation measures how much the reward
  explains.
- **bc_umi_h8 test:** in 6 % (3–12; dropped 50, never grasped 17, held 14; collisions 84 %), ood 0 %
  (0–4; never grasped 91).
  - The ood 0 % is expected (no demo coverage). In-region 6 % was suspicious, so I traced seeds
    30000–30002: **premature mid-air release** (30002: suction off 222 mm from the target → drop),
    **suction flicker** at the target (30001: off / on / off within 3 steps at 7–22 mm), stalls with
    collisions (30000).
  - With 449 demos over widely varying wall / rack layouts, BC learns the motion but the suction on/off
    decision generalizes poorly (cf. the easy task's rising held-out attach BCE).
- iql_sim_b3_s100 in 21 % (held 53), iql_sim_b10_s100 in 2 % (held 71).
- **Idea for the user (not implemented, it's a new feature):** a rule-based **release gate** around any
  learned policy: suction off only when the panel estimate is within a few mm of the target and against the
  wall; suction on only near the panel. That is the Part D hybrid recommendation, testable as an extra eval
  row without retraining.

### Hard challenge: final results (100 test episodes per cell; in = 30000–30099 region demo, out = 31000–31099 region ood)
| policy | data | in | out | collisions in/out |
|---|---|---|---|---|
| scripted demo | — | 99 % (95–100) | 92 % (85–96) | 0 / 6 % |
| bc_umi_h8 | UMI demo region | 6 % (3–12) | 0 % (0–4) | 84 / 65 % |
| bc_umi_sim_h8 | UMI + sim successes | 55 % (45–64) | 20 % (13–29) | 19 / 34 % |
| iql_sim_b3_s100 | sim | 21 % (14–30) | 11 % (6–19) | 56 / 52 % |
| iql_sim_b10_s100 | sim | 2 % (1–7) | 10 % (6–17) | 70 / 63 % |
| iql_sim_b3_s1000 | sim | 8 % (4–15) | 14 % (9–22) | 59 / 57 % |
| iql_sim_b10_s1000 | sim | 16 % (10–24) | 19 % (13–28) | 51 / 44 % |
| iql_umi_sim_b3_s1000 | UMI + sim | 19 % (13–28) | 16 % (10–24) | 49 / 53 % |
| iql_umi_sim_b10_s1000 | UMI + sim | 39 % (30–49) | 22 % (15–31) | 36 / 36 % |

- **Findings (README "Hard challenge"):**
  1. Demos alone 0 % outside their region (never grasped 91); sim data gives coverage.
  2. 449 demos aren't enough even in-region (6 %): suction decisions generalize poorly (50 drops, flicker).
  3. IQL ≤ BC control on the same data (39 vs 55 % in, 22 vs 20 % out); UMI + sim > sim only for IQL.
  4. Held at end dominates every learned policy, BC included (so not only reward hacking). Reward ×10 helped
     β10 (2 → 16 % in, CIs disjoint) but not β3 (21 → 8 %).
  5. IQL collides more.
  6. Scripted 99 / 92 % → the Part D hybrid recommendation.
- **Charts:** comparison_in.png / comparison_ood.png (comparison chart fixed: rotated labels, legend below,
  test-set name in the title, time-out note; easy comparison re-rendered with the same layout),
  training_*.png for all 8 hard runs.
- Chart footer fixed: it said "seeds 5000-5049" for hard runs; now it reads the seeds from the log.
- The value curves show the same overshoot and drift as easy (e.g. umi+sim β10: peak ≈ 24 at 50k, ≈ 19 at
  200k), so the critics have not converged.
- README: hard-challenge section, updated headline, viewer docs (`--challenge hard --snapshots`).
- **Found while committing: `MUJOCO_LOG.TXT`, 13 MuJoCo "simulation is unstable" warnings** (NaN / Inf /
  huge QVEL or QACC at DOFs 0–2), 2026-11-18 00:18–03:09.
  - On this warning MuJoCo resets the simulation state and continues, so the affected episode is silently
    corrupted from that step on.
  - **4 are deterministic in the hard sim data:** the same sim times (14.15, 18.60, 35.00, 36.87 s) appear in
    the original generation and in the regeneration, so about 4 / 2000 episodes (0.2 %).
  - The rest occur late in episodes (t ≈ 45–49 s) during learned-policy validation / evaluation, most likely
    a policy pressing the panel hard into the wall near the time-out: a handful out of thousands of episodes.
  - Impact on the reported numbers: negligible (they count as failures either way), but it is a correctness
    issue.
  - **Proposed fix (not done; it changes the data, user's call):** the env checks `data.warning`
    (mjWARN_BADQVEL / BADQACC) after each substep and ends the episode with `info["unstable"] = True`;
    regenerate the data and exclude or flag such episodes.
  - MUJOCO_LOG.TXT is git-ignored; this entry is the record.
- Commit `e4ed52a` "Hard challenge: …" as the user (no trailers). MUJOCO_LOG.TXT was unstaged and
  git-ignored; the notebook entry above went into the next commit.
- **Technical report draft** `docs/technical_report.md` (about 4 pages): architecture and decisions, training
  methodology, easy + hard results, failure analysis, deployment summary. One overstatement fixed while
  drafting ("IQL collides more than BC" → "than the BC control on the same data"; BC on demos alone collides
  most).
- **Submission video** `scripts/make_submission_video.py` → `media/submission_video.mp4`: **3.0 min**, 1,790
  frames. Title cards + 10 labelled clips at 2×: easy env/scripted, hard scripted, BC success + failure (grasp
  fixed point), IQL success + failure (off-wall release), hard BC-demos failure (mid-air drop), hard BC
  control out-of-demo success, best hard IQL success + failure (held at end), takeaways. Covers the doc's
  requirements: environment, data-collection rollout, learned policy, failure cases with interpretation.
  Cards checked visually.

## 2026-11-18 — Diagnosing low IQL success (user: "propose fixes to get IQL success higher")

Diagnostics only. Nothing retrained; scripts were in the scratchpad.
- **Release is rare in the data.** Among suction-on states, the release action is
  - 1 : 151 in easy UMI data and 1 : 173 in easy sim data;
  - 1 : 144 in hard UMI data and 1 : 224 in hard sim data.
  The suction BCE is unweighted with a fixed 0.5 threshold.
- **Correction to my own earlier claim: hard-IQL "held at end" mostly does NOT happen at the target.**
  The panel is carried but not placed.
  - **Hard IQL held-at-end episodes:** median final position error is 104–205 mm; only 0–11 % are within 20 mm.
  - **Hard BC control (UMI + sim):** median 31 mm, 75 % within 50 mm. This one is close to a release-decision
    failure.
  - **Easy IQL UMI+sim β3:** all 6 held episodes are within 20 mm and 1 mm from the wall. This is a genuine
    release-decision failure.
  - So the hard IQL problem is approach precision; the easy IQL problem is release timing. The report and
    video card that say "carried to the wall but never released" need softening.
- **Hard runs are undertrained.** Validation success is still rising at 200k steps (0 → 12–20 %); the paper uses
  1M steps.
- **Advantage weights saturate on hard with success = 1000.** The advantage std is about 2.0 (s1000) vs 1.0
  (s100) and 0.2–0.4 (easy).
  - With β = 10, 11–17 % of samples hit the exp-weight clip of 100, and the mean weight is 15–20.
  - The actor is effectively trained on a noisy subset.
  - Reward scale 0.1 was chosen for success = 100 and was never re-tuned for 1000.
- **The critics are β-independent.** Q and V logs are identical across the β pairs (same seed, and the critic
  never sees the actor), so a β sweep only needs to retrain the actor.
- Proposals went to chat; nothing is implemented pending the user's decision.

### [DECISION] IQL fixes 1–4 (user: "ok do 1-4. be autonomous until result")
The user approved fixes 1–4. The sub-choices below are mine, made autonomously while the user was away.
1. **Advantage weighting.** Use `--reward-scale 0.01` with hard success = 1000, so Q has the same range as the
   s100 runs. Sweep β ∈ {1, 3, 10}. Target: under 2 % of samples at the exp-weight clip.
2. **Longer training.** 1M steps instead of 200k (paper setting). Validation every 100k instead of 20k, so
   validation does not dominate wall time.
3. **Balanced sampling** (`--balance`): every batch draws an equal share from each `--data` file. With
   UMI + sim that is 50 / 50 demo / sim, instead of about 20 / 80 by transition count.
4. **Release weighting.**
   - **Training** (`--release-weight 10`): BCE elements whose state has suction on and whose target is "off"
     (the release and the steps after it inside the chunk) get weight 10.
   - They are about 1 : 44 of suction-on chunk elements, so 10 only partly rebalances. Fully balancing (about
     44) risks early releases, which are already easy IQL's largest failure.
   - **Inference:** an optional `release_threshold` in `TorchPolicy`. While suction is on, keep it on only if
     p(attach) > threshold. Grasping still uses 0.5.
   - The threshold is tuned on VALIDATION seeds only (new `scripts/tune_release_threshold.py`), then tested
     once on the test seeds. The default of 0.5 reproduces every earlier result.
- **Runs.** Separate processes, one per β, rather than one shared critic. Each process trains identical critics:
  wasteful on GPU, but it parallelises the CPU-bound validation rollouts and keeps the script simple.
- **Ablation.** One extra run, `iql_umi_sim_b10_s1000_1M`, uses the old settings (scale 0.1, no balance, no
  release weight) but 1M steps. It separates "trained longer" from fixes 1, 3 and 4.
- **Scope.** Hard UMI + sim data only. On the easy challenge, fix 4 is applied as threshold tuning of the
  existing policies, with no retraining.

### IQL fixes: implementation and launch
- **Code changes:**
  - `ActorNet.loss_per_sample(bce_weight=)`.
  - `TorchPolicy(release_threshold=)` and `load_policy(release_threshold=)`; the default 0.5 is identical to
    before.
  - `load_many` returns `source` (the file index per transition).
  - `train_iql.py --balance --release-weight`.
  - `eval_policy.py --release-threshold` (results suffix `_rt<v>`).
  - New `scripts/tune_release_threshold.py` (validation seeds only).
- **Tests:** two new (the threshold acts only on attached states; the BCE weight scales the loss). **40 passed.**
- **Smoke test** (600 steps): release-weighted elements are 1.4 % of all chunk elements; source sizes UMI
  109,260 / sim 649,725 transitions.
- **Launched 4 parallel runs** (`runs/hard_challenge/console/*.txt`, about 87 steps/s each, so about 4 h
  including validation every 100k):
  - **Shared settings:** UMI + sim hard data, `--reward-terms umi --chunk 8 --n-exec 4 --steps 1000000
    --rollout-every 100000`.
  - `iql_umi_sim_fix_b{1,3,10}`: `--reward-scale 0.01 --balance --release-weight 10`.
  - `iql_umi_sim_b10_s1000_1M`: the old settings (the 1M-step-only ablation). Its config diff vs the 200k
    original is only steps, rollout_every and the new defaults.
- **First check at 6.5k steps:**
  - Advantage std is 0.13 at scale 0.01 vs 1.8 at 0.1.
  - The exp-weight clip fraction is 0 / 0.06 / 0.4 % for β 1 / 3 / 10, vs 7 % for the old settings.
  - Mean weight at β = 1 is 1.0, so it is nearly plain BC on balanced data; β = 10 is the meaningful setting.
- **Easy challenge:** started threshold tuning for the 4 existing easy IQL policies on validation seeds
  5000–5049, with thresholds {0.2, 0.35, 0.5, 0.65, 0.8, 0.9}.
- **Easy-challenge release-threshold tuning** (validation seeds 5000–5049, 50 episodes per threshold; log in
  `runs/easy_challenge/tune_release_threshold.txt`):

  | policy | 0.2 | 0.35 | 0.5 | 0.65 | 0.8 | 0.9 | chosen |
  |---|---|---|---|---|---|---|---|
  | iql_umi_sim_b3 | 46 | 72 | 80 | **86** | 84 | 78 | 0.65 |
  | iql_umi_sim_b10 | 58 | 64 | 68 | **70** | 64 | 60 | 0.65 |
  | iql_sim_b3 | 46 | 58 | **74** | 66 | 68 | 64 | 0.5 (unchanged) |
  | iql_sim_b10 | 54 | 68 | 70 | **74** | 74 | 72 | 0.65 |

  - **Low thresholds hurt clearly:** the policy holds and "held at end" rises to 8–18.
  - **Above 0.5 the gain is small** (+2 to +6 points, within the noise of 50 episodes). Releasing sooner trades
    held-at-end for released-off-wall.
  - Conclusion: on the easy task the suction head is roughly calibrated; release *timing / position*, not
    reluctance, is the limit there.
  - The chosen thresholds are evaluated once on the test seeds (`results/easy_challenge/*_rt0.65.json`).
- **Easy test result for the tuned thresholds** (test seeds 10000–10099, evaluated once; log in
  `runs/easy_challenge/eval_rt065.txt`). The paired comparison uses the same seeds; p is an exact McNemar
  test.

  | policy | 0.5 (before) | tuned (0.65) | episodes fixed / broken | p |
  |---|---|---|---|---|
  | iql_umi_sim_b3 | 71 % | **81 % (72–87)** | 10 / 0 | 0.002 |
  | iql_umi_sim_b10 | 68 % | 68 % | 4 / 4 | 1.0 |
  | iql_sim_b10 | 61 % | 60 % | 7 / 8 | 1.0 |

  - **iql_umi_sim_b3 is now on par with BC on demos (80 %).** The fix is real on this policy:
    released-off-wall went 14 → 9 and held-at-end 6 → 1.
  - **The other two are unchanged.** Their failures are released-off-wall (27–28), which a threshold cannot fix.

### IQL fix runs killed at about 600k steps; restarted with resume support
- **What happened.** All four runs died at 09:59 when the Claude Code session ended. They were children of the
  session's shell (`nohup` in Git Bash does not detach them on Windows). They had reached about 600k / 1M steps.
  - **Partial validation** (full region, 50 episodes, %), step in thousands:
    - fix_b1: 100k 30, 200k 18, 300k 26, 400k 26, 500k 26
    - fix_b3: 18, 32, 6, 10, 32, 600k 34
    - fix_b10: 10, 12, 16, 16, 24
    - b10_s1000_1M: 8, 16, 2, 18, 16, 600k 18
  - For reference, the original 200k b10_s1000 run validated at 18–24 % from 120k on.
- **Not resumable.** Only actor snapshots were saved mid-run (no critics or optimizers), so the runs restarted
  from step 0 with the same seed and settings.
- **Fix 1: resume support.** `train_iql.py --resume`:
  - At every validation step it saves the full training state (actor, critics, target, value, 3 optimizers,
    LR schedule, CPU + CUDA RNG) to `checkpoints/<sub>/snapshots/<stem>/train_state.pt`, written then renamed.
  - `--resume` loads it, truncates the log to that step and appends.
  - Smoke-tested 200 → 300 steps. Bug found and fixed: `map_location="cuda"` moved the RNG ByteTensor to the
    GPU, so it is now `.cpu()`.
  - 40 tests pass.
- **Fix 2: detached launch.** The runs are now launched with WMI `Win32_Process.Create` (`cmd /c … >>
  console\<name>.txt`), outside the session's process tree, with `--resume`. If they are killed again,
  re-running the same command continues from the last 100k step.
- **Post-processing driver** (detached the same way): a scratchpad script `post_iql_fixes.py`, progress in
  `runs/hard_challenge/console/post_iql_fixes.txt`. Once all 4 runs log `done`, it:
  1. tunes the release threshold of the 3 fix runs on hard validation seeds 25000–25049 (full region);
  2. evaluates all 4 runs once on test in (30000–30099) and test out (31000–31099) at 0.5;
  3. also evaluates the fix runs at their tuned threshold, when it isn't 0.5.

  It never looks at test seeds before choosing the threshold.

### IQL fixes: results (hard runs finished 20:13; tuning and evaluation by the driver by 20:42)
- **Training.** The restarted runs reproduced the killed runs' validation numbers exactly up to 600k
  (deterministic).
  - **Final validation** (full region, 50 episodes): fix_b1 26 %, fix_b3 30 %, fix_b10 24 %, ablation 26 %
    (the 200k original ended at 20 %).
  - **Final weight stats:** mean weight 0.99 / 1.01 / 1.44 and clip fraction 0 / 0.02 / 0.25 % for β 1 / 3 /
    10. The ablation (scale 0.1) sat at mean weight 16.9 with 13 % clipped.
- **Hard release-threshold tuning** (validation): best 0.35 for b1, 0.65 for b3, 0.65 for b10. The gains are
  2–4 points over 0.5, which is noise.
- **Test results** (100 episodes each; in = 30000–30099, out = 31000–31099):

  | run | in | out | collisions in / out |
  |---|---|---|---|
  | BC control (reference) | 55 % | 20 % | 19 / 34 % |
  | old best IQL b10_s1000 (200k) | 39 % | 22 % | 36 / 36 % |
  | ablation: same, 1M steps | 33 % (25–43) | 17 % (11–26) | 41 / 50 % |
  | fix_b1 | 50 % (40–60) | **32 % (24–42)** | 36 / 38 % |
  | **fix_b3** | **65 % (55–74)** | 26 % (18–35) | 34 / 37 % |
  | fix_b10 | 57 % (47–66) | 19 % (13–28) | 39 / 52 % |
  | fix_b1 at threshold 0.35 | 51 % | 33 % | |
  | fix_b3 at threshold 0.65 | 61 % | 26 % | |
  | fix_b10 at threshold 0.65 | 55 % | 18 % | |

- **Paired exact McNemar tests on the same seeds:**

  | comparison | test set | fixed / broken | p |
  |---|---|---|---|
  | fix_b3 vs old IQL | in | +36 / −10 | **< 0.001** |
  | fix_b3 vs BC control | in | +31 / −21 | 0.21 |
  | fix_b3 vs fix_b1 | in | +37 / −22 | 0.07 |
  | 1M ablation vs old IQL | in | +21 / −27 | 0.47 |
  | fix_b1 vs BC control | out | +22 / −10 | 0.05 |
  | everything else | out | | ≥ 0.10 |

- **Model choice.** fix_b3 is the headline because it had the best final *validation* success (30 %). The test
  agrees, but test results were not used to choose.
- **Interpretation:**
  - The fixes work in-region (39 → 65 %), and training longer alone does not explain it.
  - Out of region nothing moved significantly.
  - The four changes were not ablated individually.
  - At reward × 0.01, β3's weights are almost uniform (mean 1.01), so the β3 actor is close to BC on all the
    balanced, release-weighted data. Only β3 vs β1 (p = 0.07) hints that the advantage weighting adds
    something.
  - **Overcorrection, in hindsight:** the target was "under 2 % clipped", but scale 0.01 made the weights
    nearly flat. A larger β (about 30) at scale 0.01 would be the natural next run.
  - **Hard held-at-end** (fix_b3: 11 in / 34 out) is still an approach-precision failure: median 13–15 cm from
    the target, under 3 % within 20 mm. A threshold can't fix that. The wall-frame state (fix 6, not done)
    targets it.
- **Artifacts:**
  - charts `media/hard_challenge/comparison_fixes_{in,ood}.png` and `training_iql_umi_sim_fix_b*.png`;
  - clips `media/hard_challenge/iql_umi_sim_fix_b3_in_{success,failure}.mp4` (the evaluation re-run
    reproduced 65 % exactly; the failure clip shows a held-at-end episode).
- **Docs.** The submission video was rebuilt (3.0 min): cards 7 and 8 now show the fixed IQL and the corrected
  "carried but never placed, a median ~13 cm off" interpretation, and the takeaway was updated. README
  (headline, easy table, hard table, an "IQL fixes" subsection, findings 3–4) and the technical report were
  updated to match.

## 2026-11-18 (evening) — Wall frame, β = 30, mixture head

### [DECISION] Fixes 1, 4 and 5 from the "how can we increase the success rate" list
The user said "lets try 1 and 4 and maybe do 5". The sub-choices below are mine.
1. **Wall frame** (`--frame wall`, `ActorNet.frame`).
   - **What changes:** the network sees states and predicts actions in the base frame rotated about z by
     −ψ, where ψ is the wall target's yaw from the state's `target_rot6`. The target's z-axis is the wall's
     inward normal, so ψ = atan2(−n_y, −n_x), which is 0 on the easy layout.
   - **What is rotated:** base-frame positions and rotation columns (EE, panel, target). The relative
     poses (panel in target / EE) and the suction state are unchanged.
   - **Actions:** the position delta becomes R v and the rotation delta becomes R D Rᵀ (exact, via rpy;
     the env applies D on the left).
   - `TorchPolicy` converts the state in and the action out, so env, evaluation and viewer code is
     unchanged. Datasets are converted at load time (`dataset_to_wall_frame`); no regeneration.
   - **Checked on the hard data:** state and action round-trips are exact (≤ 1.2e-7), the canonical yaw is
     ≡ 0, and the invariant parts are untouched.
   - **Output range:** 2–4 % of rotated x / y / roll / pitch components exceed ±1 (max √2), so in the wall
     frame the motion outputs are tanh × √2 (z stays 1). The env still clips the back-rotated action to ±1.
   - **Validity:** uses only the target pose estimate, which is deployable. Not privileged.
4. **β = 30** at reward × 0.01, as agreed. The final β3 advantage histogram predicts 2.9 % clipped and a mean
   weight of 5.0, vs 11–17 % clipped originally.
   - **Side finding:** the effective sample size of the weights is only 5–7 % for β 10–30. A few
     high-advantage samples dominate, so the weighting is sharper than "mean weight 1.4" suggested.
5. **Mixture head** (`--head gmm`, K = 5): a Gaussian mixture over the whole 8-step motion chunk.
   - Diagonal std clamped to [0.01, 1]; means are tanh × the frame's scale.
   - The NLL is divided by 6H so its scale sits near BCE's.
   - At run time it executes the mean of the most likely mode.
   - I chose this over diffusion as the smaller change that still represents several modes (clockwise vs
     anticlockwise swings). In IQL the AWR weights multiply this NLL instead of the MSE.
- **Runs:** all hard, UMI + sim, chunk 8 / execute 4, seed 0.
  - IQL runs: the fixed recipe (reward × 0.01, `--balance`, `--release-weight 10`, 1M steps), with
    `--resume`.
  - BC runs: the BC-control recipe (UMI + sim successes, 30k steps).

  | run | frame | head | β | isolates |
  |---|---|---|---|---|
  | iql_umi_sim_fix_b30 | base | mse | 30 | fix 4 (vs fix_b3 / fix_b10) |
  | iql_umi_sim_fix_b3_wall | wall | mse | 3 | fix 1 (vs fix_b3) |
  | iql_umi_sim_fix_b30_wall | wall | mse | 30 | 1 + 4 |
  | iql_umi_sim_fix_b30_wall_gmm | wall | gmm | 30 | fix 5 (vs b30_wall) |
  | bc_umi_sim_h8_wall | wall | mse | — | fix 1 for BC (vs BC control) |
  | bc_umi_sim_h8_wall_gmm | wall | gmm | — | fix 5 for BC |

- **Code:**
  - `transitions.py`: `wall_yaw`, `rotate_states_z`, `rotate_actions_z`, `rpy_to_mat_batch`,
    `dataset_to_wall_frame`.
  - `policies.py`: `ActorNet(frame, head, n_modes)`, with `motion_scale` a non-persistent buffer so old
    checkpoints still load. `TorchPolicy` converts frames.
  - `train_bc.py` / `train_iql.py`: `--frame --head --n-modes`.
  - **Tests (42 pass):** wall-frame policy equivariance (rotate the scene 37° → actions rotate the same way,
    for both heads; the target is straight ahead in the wall frame), and GMM head shapes and likelihood
    ordering.
  - Smoke-tested both scripts with wall + gmm.
- **Launch, attempt 1 (21:41): all 6 aborted within seconds** with `forrtl: error (200): program aborting due to
  window-CLOSE event`.
  - The WMI launch opens a visible console window per process. The windows were closed (most likely by hand,
    since the user was at the machine), and the Intel Fortran runtime that numpy uses aborts on that event.
    The earlier 4-run batch survived only because nobody touched its windows.
  - **Fix:** relaunched at 21:53 with `Win32_ProcessStartup.ShowWindow = 0` (hidden; nothing pops up). The
    console files were cleared and the runs started from scratch.
  - **First check:** at β30 the clip fraction is 2.4 % (as predicted) and the mean weight 4.2–4.6, at about
    55 steps/s each while the BC runs are still going.
- **Post-processing driver** `post_round2.py` (scratchpad, hidden, progress in
  `runs/hard_challenge/console/post_round2.txt`): it evaluates each run on hard test in and out as soon as
  that run finishes.
  - No release-threshold tuning this round: it made no difference on the hard task.
  - Test seeds are evaluated once per model, at release threshold 0.5.

### 2026-11-18 23:25 — Round 2: BC wall-frame / GMM results (IQL runs still training)

BC runs finished 22:13; the driver evaluated them on hard test in / ood (release threshold 0.5):

| run | in | ood |
|---|---|---|
| bc_umi_sim_h8 (base, mse; control) | 55 % | 20 % |
| bc_umi_sim_h8_wall (wall frame, mse) | 63 % | 47 % |
| bc_umi_sim_h8_wall_gmm (wall frame, GMM K=5) | **93 %** | **74 %** |

Paired exact McNemar on the same seeds:
- wall vs base BC: in +29/−21 p=0.32; ood +38/−11 p=1.4e-4. The wall frame mainly fixes generalisation outside the demo region, as intended.
- wall+GMM vs wall: in +32/−2 p=7e-8; ood +37/−10 p=1e-4. The multimodal head is the biggest single gain.
- wall+GMM BC vs best IQL so far (fix_b3): in +34/−6 p=8e-6; ood +53/−5 p=4e-11.

IQL round-2 status at 23:20, about 55 steps/s, so finishing about 03:00–04:00 (validation success, 50 seeds):
- fix_b30 (base): 296k, val 10→14 %. β30 looks harmful.
- fix_b3_wall: 300k, val 58 % at 100k, 32 % at 200k.
- fix_b30_wall: 383k, val 2–8 %.
- fix_b30_wall_gmm: 380k, val 6–10 %.

The driver evaluates each on test as it finishes.

### 2026-11-18 23:45 — Fix: policy viewer crashed with --snapshots

`python scripts/view_policy.py --challenge hard --snapshots` crashed at start-up with `KeyError: 'config'`.
`find_models` listed every `*.pt` under `snapshots/`, which now includes the IQL resume files
`snapshots/<run>/train_state.pt`. Those aren't actors. That file sorted last and the default model
was `models[-1]`, so the viewer tried to load it at once.

Fix in `scripts/view_policy.py`:
- Snapshots are now only `snapshots/*/step_*.pt`.
- Final models are listed before snapshots.
- The default model is the last *final* checkpoint.

Checked: 155 hard entries listed, no `train_state.pt`, all 155 load with `load_policy`.

### 2026-11-19 00:05 — Technical report as LaTeX

Wrote `docs/technical_report.tex`, a one-to-one LaTeX version of `docs/technical_report.md` (same
sections, tables, findings; no content changes). It uses article 10pt, A4 with 2 cm margins, and the packages
booktabs, enumitem, titlesec, hyperref and amsmath, so it compiles on Overleaf as is.

Local test compile: abandoned. MiKTeX here is missing packages (`infwarerr.sty`, needed by hyperref), and
its package installer hung. The user will render on Overleaf. The .md is still the source the README links;
the round-2 results (BC wall + GMM 93 % / 74 %) are not in either version yet.

### 2026-11-19 00:23 — Technical report .tex rewritten in Bruce's voice

Rewrote `docs/technical_report.tex` with the bruce-writing-style skill, using the samples in the vault's
`Context/Voice Samples/` (academic + application registers).
- **What changed:** most bullet stacks are now prose paragraphs, in first person where I did the work, with
  parentheses for asides and no em dashes.
- **What stayed:** sections, both tables, the conversion list, the IQL loss list, and every number and claim.
  No new facts.
- **Voice check (academic register):** LOOP SCORE 9. The remaining flags are mostly technical:
  - parentheses for units and ranges;
  - "learned policy" counted as a vague word;
  - short list items.

The .md is still the old bullet version. Round-2 results are not in either yet.

### 2026-11-19 01:10 — Checked a "failed but looks placed" episode from the viewer

The user saw an episode in the policy viewer that looked placed but was marked failed. Their screenshot
shows the arm by the wall with the blue **suction plate** (not the panel). The panel is the light sheet on
the wall, and the green **target outline** shows along one edge.

I replayed the best policy's (`bc_umi_sim_h8_wall_gmm`) in-region `released_misaligned` failures (seeds
30001 / 30003 / 30019):
- Each was released on the ledge, and the fixture engaged at release.
- At the end:
  - wall distance 0.3–0.5 mm;
  - height within 3 mm of the target;
  - orientation 0.01–0.55°;
  - but **21–29 mm sideways along the wall** (target-frame x +21.0 / +24.7 / +29.2 mm, all the same
    sign).

They fail only the 20 mm position tolerance, so the evaluator is correct by its own definition.

Near misses, counted as released on the wall but 20–30 mm off:

| policy | in | out |
|---|---|---|
| wall + GMM BC | 6 of 7 fails | 4 of 26 |
| IQL fix_b3 | 4 of 35 | 6 of 74 |

The consistent +x sign suggests a small systematic lateral bias in that policy. Not investigated yet.

## 2026-11-19 03:05 — Round 2 complete: all results, paired tests, charts

The driver (`post_round2.py`) finished at 02:59:39 with no aborts (no `forrtl`). Every run was evaluated
on the same 100 held-out seeds per test set (in-region 30000–30099, outside-region 31000–31099).

| policy | in | ood |
|---|---|---|
| BC umi+sim h8 (round 1) | 55 | 20 |
| BC + wall frame | 63 | 47 |
| **BC + wall frame + GMM** | **93** | **74** |
| IQL fix β3 (round 1 best) | 65 | 26 |
| IQL fix β3 + wall frame | 63 | 53 |
| IQL fix β30 | 41 | 7 |
| IQL fix β30 + wall frame | 41 | 16 |
| IQL fix β30 + wall frame + GMM | 75 | 25 |

Paired exact McNemar (only-A / only-B, p):
- β30 vs β3: in 14/38 p=0.0012, ood 3/22 p=1.6e-4. β30 vs β10: in p=0.04, ood p=0.017. **β30 hurts**:
  a sharper AWR weight overfits the critic's errors.
- IQL β3 + wall vs β3: in 21/23 p=0.88, ood 37/10 p=9.8e-5. **The wall frame doubles IQL ood**, the same
  effect as for BC.
- IQL β30 + wall vs β30: in p=1, ood 15/6 p=0.078 (same direction, not significant at β30).
- IQL β30 + wall + GMM vs β30 + wall: in 41/7 p=6.2e-7, ood 21/12 p=0.16. **GMM helps IQL too.**
- IQL β3 + wall vs BC + wall (same input frame, both unimodal): in 24/24 p=1, ood 22/16 p=0.42.
  **There is no significant difference**: the offline-RL reweighting neither helps nor hurts at matched architecture.
- IQL β3 + wall vs BC + wall + GMM: in p=2.8e-6, ood p=1.1e-3. IQL β30 + wall + GMM vs BC + wall + GMM:
  in p=5.3e-4, ood p=5.9e-12. **BC + wall + GMM stays the best learned policy.**

Conclusion: the representation choices (wall frame, multimodal head) matter far more than the
learning algorithm. At a matched architecture IQL ties BC. The untested fair cell is IQL β3 + wall + GMM
(about 5 h); it has not been launched and awaits the user's decision.

Charts: `media/hard_challenge/comparison_round2_{in,ood}.png`; training curves
`media/hard_challenge/training_{iql_umi_sim_fix_b30,iql_umi_sim_fix_b3_wall,iql_umi_sim_fix_b30_wall,iql_umi_sim_fix_b30_wall_gmm,bc_umi_sim_h8_wall,bc_umi_sim_h8_wall_gmm}.png`.
The McNemar script is the scratchpad `mcnemar2.py` (exact binomial, `math.comb`; scipy is not installed).
