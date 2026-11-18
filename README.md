# Drywall panel placement — sim environment, UMI-style demos, BC and offline RL (Parts A–C)

**Part B (UMI → BC) and Part C (offline RL / IQL)** start at "Part B" below; the **hard challenge** (wall and
rack anywhere around the robot) has its own section. Headlines on 100 held-out test episodes each:
- **Easy:** scripted 100 %, **BC on UMI demos 80 %**, **IQL on UMI + sim data 71 %**. RL removed BC's grasp
  failures but added release failures.
- **Hard:** BC on the limited demos 6 % inside / **0 % outside** their region. Adding sim data: BC 55 % / 20 %,
  **IQL 39 % / 22 %**. Sim data is what lets a policy work outside the demos' coverage; offline RL still
  didn't beat BC on the same data. Scripted controller 99 % / 92 %.
- **Part D** deployment proposal: [`docs/deployment_proposal.md`](docs/deployment_proposal.md).

A Franka Panda with a suction plate picks a 0.6 × 0.4 × 0.01 m drywall panel off a rack,
carries it to a wall, aligns it with a target region, places it flush against the wall and
releases it onto a ledge fixture. Gymnasium-style API, Cartesian end-effector delta actions,
deployable vs. sim-only observations, separately logged reward terms, seeded randomization, and
a scripted waypoint controller that uses only the policy interface.

Demo video: [`media/scripted_seed0.mp4`](media/scripted_seed0.mp4), the **scripted controller** (no learned policy) on seed 0, with the overview on the left and the side view on the right.
Hard challenge: [`media/hard_challenge/scripted_hard_seed1.mp4`](media/hard_challenge/scripted_hard_seed1.mp4), the scripted controller (demo mode) picking from a rack *behind* the robot and placing on a rotated wall, with an overhead view on the left and a wall-relative side view on the right.

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

## Datasets (not in the repository — regenerate them)

`data/` is git-ignored. The files are 21 MB to 7 GB, and some exceed GitHub's 100 MB limit.
- **Deterministic:** every dataset is generated from fixed seeds, so the commands below rebuild them exactly on
  the same machine and MuJoCo build.
- **Already committed:** the trained checkpoints, training logs and per-episode test results (`checkpoints/`,
  `runs/`, `results/`).
- **Only needed for retraining.** Times are for a 24-thread CPU + RTX 4070.

```bash
# --- easy challenge (Parts B and C as first reported)
python scripts/record_umi.py --n 500 --out data/umi_sim.npz                       # ~48 min, 7.1 GB  (seeds 0-499)
python scripts/convert_to_umi_zarr.py --npz data/umi_sim.npz --out data/umi_sim.zarr.zip   # ~7 min, 4.1 GB
python scripts/umi_to_transitions.py --replay 500                                  # ~4.5 min -> data/umi_transitions.npz (21 MB)
python scripts/generate_offline_data.py --bc checkpoints/easy_challenge/bc_umi_h8.pt  # ~2 min -> data/offline_sim.npz (44 MB, seeds 1000-1999)

# --- hard challenge (wall and rack anywhere around the robot)
python scripts/record_umi.py --challenge hard --region demo --mode demo --n 500 --start-seed 20000 \
       --max-noise 0 --out data/hard/umi_hard.npz                                  # ~1 h (seeds 20000-20499)
python scripts/convert_to_umi_zarr.py --npz data/hard/umi_hard.npz --out data/hard/umi_hard.zarr.zip
python scripts/umi_to_transitions.py --challenge hard --region demo --zarr data/hard/umi_hard.zarr.zip \
       --sidecar data/hard/umi_hard_sim_meta.npz --out data/hard/umi_transitions_hard.npz --replay 500
python scripts/generate_offline_data.py --challenge hard --out data/hard/offline_sim_hard.npz   # ~4 min, 173 MB (seeds 21000-22999)
```

## UMI-style demonstration dataset

```bash
python scripts/record_umi.py --n 500 --out data/umi_sim.npz   # ~40 min, ~6-7 GB
```

The script writes scripted demonstrations as if they had been recorded with a UMI hand-held
gripper (Chi et al. 2024, arXiv:2402.10329). The scripted controller stands in for the human.
The file contains only what UMI records, plus the target-relative pose and an external camera:

| key (`episode_XXX/…`) | shape, dtype | rate | meaning |
|---|---|---|---|
| `timestamp` | (T,) float64 | 60 Hz | s, session clock; episodes run back to back |
| `ee_pos`, `ee_quat` | (T,3), (T,4) float32 | 60 Hz | absolute EE (suction face) pose, world frame, wxyz |
| `attached` | (T,) uint8 | 60 Hz | gripper state: 1 = suction holding the panel |
| `panel_pos_in_target`, `panel_quat_in_target` | (T,3), (T,4) float32 | 60 Hz | panel pose in the target frame |
| `image_timestamp` | (K,) float64 | 10 Hz | image k was taken at pose sample 6k |
| `wrist_rgb` | (K,224,224,3) uint8 | 10 Hz | 155° equidistant fisheye on the plate (UMI's GoPro view) |
| `external_rgb` | (K,224,224,3) uint8 | 10 Hz | fixed overview camera |
| `start_time`, `end_time` | () float64 | per episode | episode boundaries |
| `success` | () bool | per episode | the env's success criterion at the end of the episode |

`meta/` holds `schema_version`, `n_episodes`, `pose_hz`, `image_hz` and `fisheye_fov_deg`. Load the file
with `np.load(path)["episode_000/ee_pos"]`; arrays are read lazily.

**Camera calibration** (in `meta/`, OpenCV conventions: x right, y down, z forward, pixel centres
at integers; `T_a_b` = pose of b in a):
- `wrist_K`, `wrist_D` (zeros), `wrist_model = "opencv_fisheye"`: equidistant, so `cv2.fisheye`
  with D = 0. Also `T_ee_wrist`, the fixed mount in the EE frame.
- `external_K`, `external_model = "opencv_pinhole"`, `T_world_external`.
- `mirror_polygons` (normalised u, v), plus `mirror_normals`, `mirror_points` and
  `mirror_virtual_origins` in the wrist camera frame (m).

`tests/test_umi_recorder.py` checks this against the renderer. It ray-casts from each camera,
projects the hit points with the stored calibration, and compares the result with a
segmentation render: 98–99 % of pixels agree, and the misses are object edges. To add the
calibration to a file recorded before it existed, run
`python scripts/record_umi.py --out data/umi_sim.npz --append-calibration`.

**Conversion to UMI's own format** (so UMI's diffusion-policy training code can read it directly):

```bash
python scripts/convert_to_umi_zarr.py --npz data/umi_sim.npz --out data/umi_sim.zarr.zip
```

This writes UMI's replay buffer (zarr v2 `ZipStore`, the layout from UMI's
`07_generate_replay_buffer.py`): `data/camera0_rgb` (JpegXl level 99, one frame per chunk),
`robot0_eef_pos`, `robot0_eef_rot_axis_angle`, `robot0_gripper_width`,
`robot0_demo_start_pose`, `robot0_demo_end_pose`, and `meta/episode_ends`.
- **Row rate:** rows are at 10 Hz, the image rate, so set `down_sample_steps = 1` in UMI's task config.
- **Gripper width:** `robot0_gripper_width` is the suction state as 0.0/1.0, standing in for width.
- **Extras UMI ignores:** `external_rgb`, `timestamp`, `panel_*_in_target`,
  `meta/episode_success`, and the calibration as JSON in the root attrs.
- **Relative trajectories:** UMI computes these at training time from the absolute poses, so they
  aren't stored. `drywall_env.umi_zarr.relative_trajectory(pos, rotvec, base_index)` does the same
  conversion, `T_rel = inv(T_base) @ T`.

- **No actions, rewards or done flags.** UMI doesn't record them; they come from the poses.
- **Sim-only metadata goes in a sidecar.** `data/umi_sim_sim_meta.npz` holds per-episode `seed`,
  `action_noise_std`, randomization factors, `dropped`, `truncated` and `n_control_steps`, for
  per-factor breakdowns.
- **Demo quality varies.** Each episode draws scripted-controller action noise from U[0, 0.2],
  so the dataset contains failures.
- **Wrist camera.** It is mounted 5 cm past the panel's long edge and 5 cm behind the plate face,
  looking along the approach direction. The panel would block a view along the plate normal.
  MuJoCo has no fisheye, so the script renders a 768² pinhole at 155° and remaps it with
  r = f·θ (2× supersampled).
- **Side mirrors (UMI's implicit stereo).** Two flat mirrors appear in UMI's mirror regions,
  lower left and lower right. The polygons are taken from UMI's `cv_util.py`.
  - **Rendering:** each mirror pixel's ray is reflected off the mirror plane, and the scene is
    sampled from the virtual camera, which is the lens reflected across that plane. This is
    exact planar optics, so mirror content is mirror-flipped, as on a real GoPro.
  - **Placement:** UMI's build guide only lists "plastic stick-on mirror" (the geometry is in
    their CAD, which we don't have). So placement is chosen here: 6 cm from the lens, reflecting
    to a forward view tilted 15° inward. That gives a stereo baseline of 11.9 cm between the two
    virtual cameras.
  - **Outside the circle:** 4.6% of each UMI polygon falls outside our inscribed 155° image circle
    and stays black.
  - **Not modelled:** mirror frames, tint and dirt.
- **Mirror swap — NOTE:** the `.npz` keeps the **raw** view. `convert_to_umi_zarr.py` applies UMI's
  digital reflection ("mirror swap") to `camera0_rgb` by default, which is what UMI's paper found
  best for policies. So inside the two mirror regions, `camera0_rgb` ≠ npz `wrist_rgb`. Use
  `--no-mirror-swap` to keep the raw view in the zarr. The root attr `mirror_swap` records which
  was used.
- **Assumptions:** no SLAM drift (exact poses), perfect gripper tracking, and human motion
  approximated by the scripted controller.

## Dataset viewer

```bash
python scripts/view_dataset.py --zarr data/umi_sim.zarr.zip --episodes 0-49
python scripts/view_dataset.py --episodes 0-19 --camera wrist --overlay attached,error --3d
```

Opens a Tk window for browsing many episodes of the UMI zarr at once. All episodes start
together, and an episode that has ended holds its last frame.
- **Trajectory plots:** EE paths of the selected episodes in a top view and a side view, with the
  wall and rack drawn in. Green = success, red = failure. A marker per episode moves with the
  time slider (Play / speed).
- **Camera grid** (`--camera wrist|external`, or the radio buttons): one tile per episode with a
  green/red border.
- **Overlays** (checkboxes or `--overlay`):
  - `attached`: suction runs are drawn as thick path segments, plus a SUCTION badge on the tiles.
  - `error`: markers are coloured by panel-to-target error (blue 0, yellow 50, red ≥ 100 mm),
    with the error in mm and degrees shown on the tiles.
- **3D replay** (button or `--3d`): a MuJoCo viewer showing the robot for the "focus ep",
  re-posed by IK from its recorded EE poses, with the scene layout restored from its seed. All
  selected episodes appear as EE path lines and translucent ghost panels at their recorded
  panel pose. Ghost panels need the sidecar `*_sim_meta.npz` for each episode's target offset;
  the viewer finds it next to the zarr automatically.

## Part B — UMI demonstrations → robot data → behaviour cloning

```bash
python scripts/umi_to_transitions.py --replay 500       # B.B: zarr -> data/umi_transitions.npz (~4.5 min)
python scripts/train_bc.py --chunk 8 --n-exec 4 --out checkpoints/easy_challenge/bc_umi_h8.pt   # B.C (~7 min, GPU)
python scripts/train_bc.py --out checkpoints/easy_challenge/bc_umi.pt                          # single-step baseline
python scripts/eval_policy.py --policy scripted checkpoints/easy_challenge/bc_umi_h8.pt checkpoints/easy_challenge/bc_umi.pt --results-dir results/easy_challenge
python scripts/eval_policy.py --policy checkpoints/easy_challenge/bc_umi_h8.pt --video media/easy_challenge/bc_umi_h8 --results-dir results/easy_challenge --name "BC on UMI demos (chunked, H=8)"
python scripts/plot_training.py runs/easy_challenge/bc_umi_h8 runs/easy_challenge/bc_umi --out-dir media/easy_challenge
python scripts/view_policy.py                                                    # interactive, see below
```

**Learning state (37-D, base frame), shared by Parts B and C.** It's the env's deployable observation without
the joint states, because UMI has no robot joints:
`ee_pos 3 | ee_rot6 6 | attached 1 | panel_pos 3 | panel_rot6 6 | target_pos 3 | target_rot6 6 | panel_in_target_pos 3 | panel_in_target_rot6 6`.
- **Rotations:** stored as the 6D representation (first two rotation-matrix columns; Zhou et al., CVPR 2019,
  as UMI does). Quaternions are not usable here: the tool-down EE orientation has w ≈ 0, and the w ≥ 0
  convention flipped the EE quaternion's sign 10 times in one episode.
- **One builder for both sources:** `transitions.build_state` builds the state from UMI data and from a
  live env, so any policy trained on it runs in the env.

**Conversion (`drywall_env/umi_to_robot.py`), each B.B requirement:**
1. **World → base frame:** `p_b = R_bw p_w + t_bw`. The identity in this sim, but the test runs it with a
   rotated and translated "SLAM map" frame and gets the same states.
2. **Temporal resampling** to the 10 Hz control rate: linear for position, slerp for rotation, zero-order
   hold for the gripper. This is exact here because UMI rows are already 10 Hz, and it works for any rate.
3. **Absolute → relative:** the action is the base-frame pose delta between consecutive samples, scaled by the
   env's 2 cm / 3° step limits and clipped to [−1, 1]. 0.9 % of actions were clipped.
4. **Gripper mapping:** `gripper_width` (attached 0/1) becomes the attach bit +1 / −1, using the state after
   the step.
5. **Workspace:** EE samples outside the env's workspace box are counted: 0 in this dataset.

**Rewards** use only the terms UMI data supports: position error, orientation error, action, jerk and
success. Collision, contact force and drop need force or contact sensing, so they are 0 and documented.
Every term is stored separately so the data can be relabelled.

**Terminals:** `terminal = True` on the last step of successful episodes. Failed episodes end as `timeout`s.

**Assumptions** needed to treat the synthetic trajectories as UMI demonstrations: poses without SLAM drift,
perfect gripper sensing, a human approximated by the noisy scripted controller, and the target pose from the
sim sidecar as a stand-in for perception (fiducials or a laser-registered wall layout).

**Conversion check:**
- **States match exactly:** rebuilding a recorded episode's states from the zarr matches a re-run of the same
  episode in the env to 1.2e-7.
- **Open-loop replay** of the derived actions on all 500 episodes:
  - 73.4 % success vs 74.2 % recorded; outcome agreement 87 %.
  - 91 % of recorded successes succeed again; EE tracking error 7.2 mm.
  - A one-step latency shift made replay worse (13 mm), so lag = 0.

**File format** (`drywall_env/transitions.py`), D4RL-style `.npz`:
- **Training arrays:** `observations, actions, rewards, next_observations, terminals, timeouts`.
- **Episode arrays:** `episode_ends, episode_success, episode_seed`.
- **Extras:** `reward/<term>` and `meta_json`.
- Nothing else; images stay in the zarr.

**Policy (`drywall_env/policies.py`):** MLP 3×256 + LayerNorm on normalized states.
- **Output:** 6 motion dims (tanh) and 1 attach logit.
- **Loss:** MSE + BCE, which is the negative log-likelihood of a fixed-variance Gaussian × Bernoulli.
- **Action chunking:** the network predicts H = 8 actions, 4 are executed, then it re-plans (ACT /
  Diffusion Policy style).
- **Data:** filtered BC on the 371 successful demos (334 train, 37 held out), 30k steps.

**Results on held-out test seeds 10000–10099 (100 episodes each, never used for training or model choice):**

| policy | success | 95 % CI | pos err (mm, mean) | ori err (°) | collision | length (steps) |
|---|---|---|---|---|---|---|
| scripted controller | 100 % | 96–100 % | 3.1 | 0.04 | 0 % | 171 |
| **BC, chunked H = 8** (`bc_umi_h8`) | **80 %** | 71–87 % | 132.7 | 14.6 | 9 % | 196 |
| BC, single-step (`bc_umi`) | 23 % | 16–32 % | 619.4 | 68.6 | 2 % | 272 |

**Model choice:** chunked vs single-step was chosen on validation seeds 5000–5049 (84 % vs 16 %); both are
shown here. Training curves: `media/easy_challenge/training_bc_umi_h8.png`, `media/easy_challenge/training_bc_umi.png`. Videos:
`media/easy_challenge/bc_umi_h8_success.mp4`, `media/easy_challenge/bc_umi_h8_failure.mp4` (labelled in the frame).

**BC failure analysis.** All 20 failures of `bc_umi_h8` are 300-step time-outs, of two kinds:
- **Never grasped (16):** the policy descends to about 9 mm above the panel face and stops. It never
  commands suction (the env would accept up to 30 mm). Near the goal the demonstrated motions are tiny, so
  the regressed motion goes to ≈ 0 before reaching the state where the demos switch suction on: a BC fixed
  point.
- **Released early (4):** the panel is carried to the target but released about 7 mm short of the wall. It
  lands on the ledge, but the fixture needs ≤ 2 mm, so it isn't counted as placed.

**Two more findings:**
- **Held-out loss is a poor proxy for success:** the held-out attach BCE rises after about 3k steps
  (overfitting the suction timing) while closed-loop success keeps rising. That's why training also logs
  validation rollouts.
- **Why chunking helps so much:** single-step BC is badly hurt by phase ambiguity (hovering and starting to
  descend look identical), and committing to 8-step segments fixes most of it.

**From this MLP to a pretrained VLA.** The same data and interface would post-train a pretrained
vision-language-action model (for example π0, OpenVLA or Octo). Each transition supplies:
- **Image observations:** the zarr's wrist fisheye + external camera at 10 Hz.
- **Proprioception:** the EE pose and suction state in this repo's 37-D layout (or its EE-only part).
- **Action target:** the same base-frame delta chunk (H × 7), plus a language instruction ("place the panel
  on the wall target").

Fine-tuning replaces this MLP's loss with the VLA's action head (flow matching, diffusion or tokenized
actions) on the same (observation, action-chunk) pairs. The UMI → robot conversion, the action space and the
evaluation harness stay the same. What changes is input normalization (the VLA's own statistics, or
per-embodiment statistics) and the action horizon, which must match the VLA's chunk length.

## Part C — Simulation-based offline RL (IQL)

```bash
python scripts/generate_offline_data.py --bc checkpoints/easy_challenge/bc_umi_h8.pt     # C.A -> data/offline_sim.npz (~2 min, 22 procs)
python scripts/train_iql.py --data data/offline_sim.npz --beta 3  --chunk 8 --n-exec 4 --out checkpoints/easy_challenge/iql_sim_b3.pt
python scripts/train_iql.py --data data/offline_sim.npz --beta 10 --chunk 8 --n-exec 4 --out checkpoints/easy_challenge/iql_sim_b10.pt
python scripts/train_iql.py --data data/umi_transitions.npz data/offline_sim.npz --reward-terms umi \
       --beta 3  --chunk 8 --n-exec 4 --out checkpoints/easy_challenge/iql_umi_sim_b3.pt          # ~1 h each on an RTX 4070
python scripts/train_iql.py --data data/umi_transitions.npz data/offline_sim.npz --reward-terms umi \
       --beta 10 --chunk 8 --n-exec 4 --out checkpoints/easy_challenge/iql_umi_sim_b10.pt
python scripts/train_bc.py --data data/umi_transitions.npz data/offline_sim.npz --chunk 8 --n-exec 4 \
       --out checkpoints/easy_challenge/bc_umi_sim_h8.pt                                          # control: BC on the same data
python scripts/eval_policy.py --policy scripted checkpoints/easy_challenge/*.pt --results-dir results/easy_challenge
python scripts/plot_training.py runs/easy_challenge/iql_* --compare results/easy_challenge/*.json --out-dir media/easy_challenge
```

**C.A Offline dataset** (`data/offline_sim.npz`): 1,000 episodes on seeds 1000–1999, 234,717 transitions,
same 37-D state and file format as Part B.
- **Behaviour policies, mixed per episode:**
  - the scripted controller (477 episodes, 55 % success);
  - the Part B BC policy (523 episodes, 59 % success).
  - Both add Gaussian action noise with std ~ U[0, 0.3].
- **Randomized factors:** panel position and yaw on the rack, target offset, panel mass, panel friction, and
  action noise.
- **Overall:** 57 % success, 2.5–6 % drops, and collisions in 20 % (scripted) and 63 % (BC) of episodes, so
  plenty of failures to learn from.
- **Rewards:** the env's full reward, every term stored separately.
- **Stored action:** the one actually executed, after noise and clipping.

**C.B Algorithm: Implicit Q-Learning, implemented from scratch in `scripts/train_iql.py`.** It follows
Kostrikov et al., ICLR 2022 (arXiv:2110.06169); the docstring derives each loss.
- **Value, eq. 5:** `L_V = E[ L2^τ(Q̂(s,a) − V(s)) ]` with the expectile loss `L2^τ(u) = |τ − 1(u<0)| u²`.
- **Q-function, eq. 6:** `L_Q = E[ (r + γ(1−done) V(s') − Q(s,a))² ]`.
- **Policy, eq. 7:** `L_π = −E[ exp(β(Q̂ − V)) log π(a|s) ]`, with the weight clipped at 100.
- **Settings:** twin target critics with Polyak 0.005, τ = 0.7, γ = 0.99, lr 3e-4, batch 256, 200k steps,
  reward × 0.1.
- **The actor is the Part B network** (chunk 8, execute 4), so IQL's actor loss is BC's loss times a per-sample
  weight. Each chunk is weighted by the advantage of its *first* action; this is an approximation, see
  limitations.

**Why this is offline RL, not online:**
- **No environment interaction during learning:** the update loop reads only the fixed arrays and never
  calls `env.step`.
- **No unseen actions:** IQL never evaluates Q at an action outside the dataset. V is an upper expectile of Q
  over *dataset* actions, and the TD target uses V(s′), not max_a′ Q(s′, a′). So it avoids the
  out-of-distribution over-estimation that CQL handles with a penalty.
- **Validation rollouts are measurement only:** every 20k steps, 50 episodes on seeds 5000–5049. They're
  never added to the data, never used for a gradient, and never used to choose a checkpoint (the final model
  is always the last step).

**C.C Comparison.**
- **Protocol:** every policy runs on the same 100 held-out test seeds, 10000–10099. These seeds were never
  used for data, training or model choice; the data scripts refuse them.
- **Same conditions for all:** deterministic policies, no observation noise, success as defined in Part A.
- **Uncertainty:** 95 % Wilson intervals, because 100 trials give about ±9 % at 70 %.

| policy | training data | success (95 % CI) | median pos err | collisions | failure modes (of 100) |
|---|---|---|---|---|---|
| scripted controller | — | 100 % (96–100) | 3.1 mm | 0 % | — |
| BC, single-step | UMI demos | 23 % (16–32) | 796 mm | 2 % | never grasped 76, held at end 1 |
| **BC, chunked H=8** — demos only | UMI demos | **80 % (71–87)** | 3.7 mm | 9 % | never grasped 16, released off wall 4 |
| BC, chunked — control | UMI + sim successes | 69 % (59–77) | 3.8 mm | 2 % | never grasped 27, released off wall 4 |
| IQL β = 3 — sim only | sim offline data | 58 % (48–67) | 7.3 mm | 22 % | never grasped 14, held 12, off wall 14, misaligned 2 |
| IQL β = 10 — sim only | sim offline data | 61 % (51–70) | 7.3 mm | 14 % | never grasped 5, held 10, off wall 21, misaligned 3 |
| **IQL β = 3 — combined** | UMI + sim | **71 % (61–79)** | 4.9 mm | 8 % | never grasped 9, held 6, off wall 14 |
| IQL β = 10 — combined | UMI + sim | 68 % (58–76) | 5.2 mm | 12 % | never grasped 3, held 6, off wall 22, misaligned 1 |

Mean episode lengths: 171 (scripted), 196 (BC H=8) and 210–231 (IQL) steps; 300 is the time-out. No policy
dropped a panel on the test seeds. Charts: `media/easy_challenge/comparison.png` and `media/easy_challenge/training_*.png`. Videos:
`media/easy_challenge/iql_umi_sim_b3_success.mp4` and `media/easy_challenge/iql_umi_sim_b3_failure.mp4`. Per-episode results:
`results/easy_challenge/*.json`.

**Findings.**
1. **Offline RL did not beat behaviour cloning on success rate.** The best IQL policy (71 %) and BC on the
   demos (80 %) overlap in their confidence intervals, and BC's point estimate is higher. With 100 trials we
   can't distinguish them; we can say IQL is not clearly better.
2. **RL changes *how* the policy fails.** IQL largely removes BC's fixed point: "never grasped" drops from
   16–27 to 3–9. That's the failure a reward can fix and a regression loss can't. But IQL adds release
   failures: still holding the panel at time-out (6–12), or releasing it more than 2 mm off the wall (14–22,
   vs BC's 4).
3. **Same data, RL vs BC:** IQL-combined (71 %) ≈ the BC control on the same data's successes (69 %). The
   advantage weighting buys a different failure profile, not a higher success rate.
4. **Combining data helps IQL:** UMI + sim (68–71 %) beats sim alone (58–61 %). The clean demonstrations
   anchor the actor where noisy sim data is ambiguous. For plain BC, adding noisier sim successes *hurts*
   (80 % → 69 %).
5. **IQL policies collide more:** 8–22 % vs 2–9 % for BC. The UMI-subset reward has no collision term, yet
   the sim-only runs, whose reward does penalize collisions, collide most. So it isn't just a missing penalty.
   They imitate the noisy, collision-heavy behaviour data (63 % of noisy-BC episodes collided) with only weak
   advantage reweighting (mean weight 1.1–1.4 at β = 3).
6. **β:** β = 10 shifts failures from grasping to releasing and lowers position error, but doesn't change
   success within the noise.

**Why IQL underperformed: hypotheses, not verified.**
- **The critic didn't converge in 200k steps.** For UMI + sim, mean V rose to about +1.0 at 45k steps and was
  still drifting down at 200k (≈ +0.15), against −1.03 for the data's empirical discounted return. Validation
  success swung ±15–20 % between checkpoints, beyond the ±14 % sampling noise. The paper trains for 1M steps.
- **The release decision is a single sparse event.** Success (+100) arrives 5 stable steps after release,
  only if the panel is within 2 mm of the wall, and the Q-function has to resolve a 7-D action whose deciding
  dimension is one bit.
- **Chunk weighting:** a chunk of 8 actions is weighted by A(s_t, a_t) of its first action only.
- **The reward doesn't separate "touching" from "almost touching" while the panel is held.** pos_error
  changes by about 5 mm between them, which is −0.0005 per step after scaling.

**Next steps if continued:**
- Train longer (1M steps).
- Use an EMA of the actor weights, which deploys a smoothed model rather than picking a checkpoint.
- Use a chunk-level critic Q(s, a_t:t+H).
- Shape the reward with a dense contact or wall-distance term.
- Weight demonstrations higher in the combined data.

## Hard challenge — pick and place anywhere around the robot

**Why a second experiment.** In the easy setup, the demos, the sim data and the tests all came from the same
narrow distribution (±5 cm), so the demos already covered everything. The task doc's premise ("human
demonstrations may be … insufficient to cover variations encountered during deployment") was never tested.
The hard challenge tests it:
- **Demos:** cover only part of the layout distribution.
- **Sim data:** covers all of it.
- **Two test sets:** one inside the demos' region and one outside it.

**Task** (`EnvConfig(challenge="hard")`; `challenge="easy"` is unchanged and bit-identical):
- **Wall:** rotates around the robot (θ ∈ ±150°), with the target anywhere in the arm's wall-facing reach band
  (±0.2 m sideways, 0.55–0.75 m high).
- **Rack:** anywhere on a 0.6–0.7 m ring, including behind the robot, at least 60° from the wall. The panel
  yaw spans 120°.
- **Where the ranges come from:** an IK reachability sweep (LABNOTEBOOK 2026-11-18). Racks closer than 0.6 m
  behind the robot, panel yaws in a +15°…+75° wrist-limit band, and wall placements above 0.8 m are
  unreachable.
- **Episode limit:** 500 steps.
- **Posture-aware IK:** the null-space posture follows the EE's direction around the base; otherwise joint6
  hits its limit at walls away from 0°.
- **Success reward:** **+1000** (×10; the easy task keeps +100), see "Reward" below.
- **Video:** [`media/hard_challenge/scripted_hard_seed1.mp4`](media/hard_challenge/scripted_hard_seed1.mp4).

**Scripted demonstrator** (`drywall_env/scripted_hard.py`):
- **Wall frame:** the easy controller's wall-side logic runs in the wall's frame. Two new *swing* phases carry
  the EE along arcs around the base, with the panel flat and its long side tangential.
- **Two modes, with setpoints sampled per episode:**
  - **`demo`:** moderate variance: hover, lift and swing heights and radii, standoff, speeds, small smooth
    noise. 90–93 % success on the full distribution, 98–99 % in the demo region.
  - **`explore`:** large variance, smooth AR(1) noise up to 0.2, random arc detours, early suction attempts,
    release delays, negative push depths. 55–64 % success.

**Data** (regeneration commands under "Datasets"):
- **UMI demos:** 500 `demo`-mode episodes in the **demo region only** (wall within ±45° of straight ahead,
  rack 60–120° to its right; this region contains the easy layout). 99.8 % success. Converted to 109,260
  transitions; open-loop replay 99.6 % vs 99.8 % recorded, 99.4 % outcome agreement.
- **Sim offline data:** 2,000 `explore`-mode episodes on the **full** distribution. 649,725 transitions,
  64 % success, 10 % drops, collisions in 41 % of episodes.
- **State:** **v2 (46-D)** = v1 + the panel pose in the EE frame, computed from the EE pose and the panel
  *estimate* only; not privileged.

**Results.**
- **Protocol:** 100 held-out test episodes per cell.
  - **in** = seeds 30000–30099 inside the demos' region;
  - **out** = seeds 31000–31099 outside it (`eval_policy.py --challenge hard --test in|ood`).
- **Network and IQL settings:** identical to the easy task (chunk 8 / execute 4, 200k steps, τ 0.7,
  reward × 0.1).
- **Validation:** seeds 25000–25049 over the full distribution.

| policy | training data | success **in** (95 % CI) | success **out** (95 % CI) | collisions in / out | main failures |
|---|---|---|---|---|---|
| scripted (`demo` mode) | — | 99 % (95–100) | 92 % (85–96) | 0 / 6 % | — |
| BC, demos only | UMI demo region | 6 % (3–12) | **0 % (0–4)** | 84 / 65 % | in: **dropped 50**; out: **never grasped 91** |
| **BC control** | UMI + sim successes | **55 % (45–64)** | 20 % (13–29) | 19 / 34 % | held at end 36 / 46 |
| IQL β3, success 100 | sim | 21 % (14–30) | 11 % (6–19) | 56 / 52 % | held 53 / 60 |
| IQL β10, success 100 | sim | 2 % (1–7) | 10 % (6–17) | 70 / 63 % | held 71 / 64 |
| IQL β3, success 1000 | sim | 8 % (4–15) | 14 % (9–22) | 59 / 57 % | held 40 / 66, never grasped 30 / 5 |
| IQL β10, success 1000 | sim | 16 % (10–24) | 19 % (13–28) | 51 / 44 % | held 67 / 61 |
| IQL β3, success 1000 | UMI + sim | 19 % (13–28) | 16 % (10–24) | 49 / 53 % | held 54 / 65 |
| **IQL β10, success 1000** | UMI + sim | **39 % (30–49)** | **22 % (15–31)** | 36 / 36 % | held 38 / 64 |

- **Charts:** `media/hard_challenge/comparison_in.png`, `comparison_ood.png` and `training_*.png`.
- **Videos (labelled):** `media/hard_challenge/*_success.mp4` / `*_failure.mp4`.
- **Per-episode results:** `results/hard_challenge/*.json`.
- **Snapshots** of every run at every validation step: `python scripts/view_policy.py --challenge hard --snapshots`.

**Findings.**
1. **Demos alone don't cover deployment, and sim data fixes coverage.** BC on the demos scores **0 %
   outside** their region: 91 of 100 episodes never even grasp, because the policy has never seen a rack
   there. Every policy that saw sim data succeeds outside the demo region (20–22 % for the best). This is the
   task doc's central claim, measured.
2. **On this task, few demos aren't enough even inside their own region.** BC on 449 demos reaches only 6 %
   in-region. It learns the motion, but the **suction decision generalizes badly** across widely varying
   layouts: premature mid-air release (50 drops) and on/off flicker at the target. Adding the sim successes
   (BC control) raises it to 55 %.
3. **Offline RL vs BC on the same data:**
   - The best IQL (UMI + sim, β10) is **below** the BC control in-region (39 % vs 55 %, CIs barely overlap) and
     **equal** outside it (22 % vs 20 %).
   - As in the easy task, IQL did not beat filtered BC.
   - Combining UMI demos with sim data again helped IQL (39 / 22 % vs 16 / 19 % for sim only at β10).
4. **"Held at end" dominates every learned policy: the panel is carried but never released.**
   - BC shows it too (36–46 episodes), and BC has no reward, so it is **not only reward hacking**. The
     learned policies struggle with the release decision and with the last centimetres of alignment.
   - **Reward ablation** (success +100 → +1000, everything else equal):
     - β10, sim only: in-region success 2 % → 16 % (CIs don't overlap), out 10 % → 19 %;
     - β3, sim only: 21 % → 8 % in-region (more never-grasped), 11 % → 14 % out.
     - The bigger success reward helps the sharper actor (β10) and not the softer one (β3); held-at-end stays
       the main failure either way.
5. **IQL collides more** (36–70 % of episodes vs 19–34 % for the BC control): it imitates the explore data's
   collision-heavy behaviour with only partial reweighting.
6. **The scripted controller stays far ahead** (99 / 92 %), because its contact and release logic is explicit.
   This supports the Part D recommendation: learned policies for gross motion and carrying, and a verified,
   force-controlled routine for the final contact and release.

**Limitations specific to the hard runs:**
- **Critics not converged:** the values still drift at 200k steps, as in the easy task.
- **No hyperparameter tuning per task.**
- **The success-1000 runs are the main comparison; the success-100 runs are an ablation.** For IQL on UMI +
  sim, only the success-1000 runs exist: the success-100 versions were stopped minutes in, when the reward was
  changed.
- **One seed per configuration.**
- **Physics instabilities:** MuJoCo reported 13 "simulation is unstable" warnings (NaN or huge velocity /
  acceleration on the panel's DOFs) and reset the state mid-episode. About 4 of the 2,000 hard sim episodes
  are affected, plus a handful of late-episode evaluation runs where a policy pressed the panel into the wall.
  They count as failures, so the effect on the numbers is negligible. A proper fix is to detect MuJoCo's
  warning and end such episodes with an `unstable` flag (LABNOTEBOOK 2026-11-18).

## Policy viewer (Parts B and C)

```bash
python scripts/view_policy.py                                  # easy: scripted + checkpoints/easy_challenge/*.pt
python scripts/view_policy.py --challenge hard --snapshots     # hard models + their validation-step snapshots
```

The MuJoCo 3D viewer plus a Tk panel, using the same single-thread pattern as `view_interactive.py`.
- **Single test:** pick a model and a seed (default 10000, the first test seed), then Play / Step / Reset with
  a live readout.
- **Many tests:** **Run N** runs N consecutive seeds with the chosen model. **watch** shows each episode in
  3D at the speed slider's rate; untick it to run fast. Every finished episode becomes a row (seed, model,
  result, errors, collision, steps), and a summary line per model shows success rate, median error,
  collision rate and length.
- **Replay:** click a row to replay that episode in 3D. The replay is added as a new row.
- **Stop:** ends a batch.
- **Hard challenge** (`--challenge hard`):
  - A **region** selector: `demo` = inside the UMI demos' coverage, `ood` = outside it, `full`. Choosing
    `demo` / `ood` jumps to that test set's first seed (30000 / 31000).
  - Only hard checkpoints are listed (46-D state).
  - The scripted model is the `demo`-mode hard controller.
- **Snapshots** (`--snapshots`): also lists the actor saved at every validation step of each run
  (`hard/<run>@<step>`). They are for watching how a policy develops during training; the reported results
  always use the final model.

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
| mean final position error | 3.06 mm (the 3 mm is the designed drop onto the ledge) |
| mean final orientation error | 0.041° |
| collision rate (episodes with ≥1 collision step) | 0 % |
| mean episode length | 171 steps (17.1 s) |
| peak panel–wall force | mean 13 N, worst 46 N |

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
  umi_recorder.py    UMI-style dataset recorder (60 Hz poses, 10 Hz fisheye + external images)
  umi_zarr.py        conversion to UMI's zarr replay buffer, relative_trajectory()
  transitions.py     37-D learning state (6D rotations), transition file format, seed allocation
  umi_to_robot.py    Part B.B: UMI -> robot actions (frame, resampling, deltas, gripper, workspace)
  policies.py        actor network (MLP + action chunking), TorchPolicy / ScriptedPolicy, checkpoints
  evaluation.py      rollouts, task-doc metrics, Wilson CI, failure-mode classification
  training_utils.py  JSONL logging, validation rollouts
scripts/run_scripted.py, scripts/eval_scripted.py, scripts/record_umi.py,
  scripts/convert_to_umi_zarr.py, scripts/view_dataset.py
scripts/umi_to_transitions.py (B.B), train_bc.py (B.C), generate_offline_data.py (C.A),
  train_iql.py (C.B), eval_policy.py + plot_training.py (C.C), view_policy.py (viewer)
tests/test_env.py, tests/test_umi_recorder.py, tests/test_transitions.py, tests/test_policies.py,
  tests/test_iql.py
checkpoints/<challenge>/*.pt   final actors (+ *.critics.pt for IQL)   challenge = easy_challenge | hard_challenge
runs/<challenge>/<name>/       training logs (log.jsonl) + console output (<name>.txt)
results/<challenge>/*.json     per-episode test results
media/<challenge>/             training charts, comparison chart, labelled videos
media/scripted_seed0.mp4       Part A demo of the environment (scripted controller)
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

**Timing:** physics dt 1/600 s, 60 substeps per step → 10 Hz control, max 300 steps (30 s).
The dt was 0.002 s until 2026-11-16 and was changed so the UMI recorder can sample at exactly 60 Hz
(every 10 substeps; see LABNOTEBOOK). `media/scripted_seed0.mp4` was re-rendered at the new dt.

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

**Who reads which group.**
- **Policies and the scripted controller:** read only `policy` or `deployable`. `policy` is built from deployable values alone.
- **Privileged:** reserved for training-time helpers (an asymmetric critic, a teacher policy) and for evaluation and debugging. It must never be an input to a deployed policy.
- **Wrapping the whole dict:** for example with `gymnasium.wrappers.FlattenObservation`. This would leak the privileged values into the policy, so training code selects `obs["policy"]` explicitly.
- **Cameras:** not part of the env observation. Policies here use pose estimates in their place. The wrist and external cameras are rendered by the UMI recorder.

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

**Why one arm instead of the spec's bimanual robot.**
- **Allowed by the spec:** it allows simplifying to a single arm. The scaled panel (0.6 × 0.4 m, 1–3 kg) can be carried and pressed flush by one centred suction plate, so a second arm adds coordination work without changing what the env tests.
- **Payload:** the Panda is rated for 3 kg, while plate plus the heaviest panel is 3.5 kg. The sim does not enforce payload limits, so this is not modelled.
- **Full-size panels:** these (1.2 × 2.4 m, ~25 kg) would need the second arm, or a lift, to support the far edge.
- **Interface stays the same:** the action and observation interface extends to two arms by stacking per-arm EE deltas and attach bits.

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
  Wall forces come from position-servo stiffness, not a force or impedance controller, so the scripted contact phase reaches up to about 46 N.
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
