# Learning drywall panel placement from UMI-style demonstrations and simulation-based offline RL

*Technical report (draft). Repository: this repo; full log of decisions and failures: `LABNOTEBOOK.md`.*

## 1. System architecture and design decisions

**Simulation (Part A).**
- **Cell:** a MuJoCo 4 work cell with a Franka Panda (Menagerie, no hand) carrying a 24 × 18 cm suction
  plate, a 0.6 × 0.4 × 0.01 m panel (1–3 kg), a rack, and a wall with a target region and a support ledge.
- **Suction:** a weld constraint that engages only when the plate is within 3 cm of the panel face, centred
  and within 30° of parallel, at the *current* relative pose (no snapping).
- **External fixture:** a released panel resting on the ledge within 2 mm of the wall is held by a second weld.
- **Action:** a 7-D action at 10 Hz: base-frame end-effector deltas (≤ 2 cm, ≤ 3° per step) plus a suction
  bit.
  - The deltas are integrated into a commanded pose, clipped to a workspace box and tracked by
    damped-least-squares IK feeding position actuators.
  - On hardware the same commanded pose would be streamed to the vendor's Cartesian impedance controller.
- **Observations:** split into *deployable* (joint states, EE pose, suction state, panel and target pose
  *estimates*) and *privileged* (true poses, velocities, contact forces).
- **Reward terms** (logged separately): position and orientation error, collisions, wall force above 50 N,
  action magnitude and jerk, drop (−50) and success (+100). Success means within 20 mm / 5°, against the
  wall, released, and still for 5 steps.

**Two task distributions.**
- **Easy:** the rack and wall are fixed, with ±5 cm / ±10° randomization plus mass and friction.
- **Hard:** the wall rotates anywhere within ±150° around the robot and the rack sits anywhere on a
  0.6–0.7 m ring, including behind the robot. The ranges come from an IK reachability sweep: the arm can face
  the wall only at panel heights of about 0.55–0.75 m, and a +15…+75° panel-yaw band is unreachable at the
  wrist.
- **Hard needed two additions:** a posture-aware IK (the null-space posture follows the EE azimuth, otherwise
  joint 6 hits its limit at rotated walls) and a 500-step limit.

**Demonstrations (Part B).**
- **Demonstrator:** a scripted waypoint controller that uses only the action interface and deployable
  observations. It stands in for the human.
- **Hard variant:** works in the wall's frame, swings around the base, and has two modes with per-episode
  setpoint variance: `demo` (moderate, mostly successful) and `explore` (large variance and smooth noise,
  many failures).
- **Recording:** episodes are recorded as a UMI dataset: 60 Hz absolute EE poses in a world/SLAM frame, a
  gripper (suction) state, a 155° fisheye wrist camera (with mirrors), an external camera, the panel pose in
  the target frame, episode boundaries and success labels.
- **Format:** stored in UMI's own zarr replay-buffer format (JpegXL images, `episode_ends`).
- **No actions or rewards are stored.**

**UMI → robot conversion.**
1. World → robot-base frame (tested with an arbitrary rotated and translated map frame).
2. Timestamp resampling to the 10 Hz control rate (linear position, slerp rotation, zero-order-hold gripper).
3. Absolute poses → base-frame deltas scaled to the action limits.
4. Gripper width → suction bit.
5. A workspace check.

The learning state is **37-D** (EE pose, suction, panel and target poses, panel-in-target pose), and **46-D**
for hard (+ panel pose in the EE frame). All rotations use the continuous 6D representation (Zhou et al.
2019): the tool-down EE quaternion has w ≈ 0 and flipped sign 10 times in one episode under a w ≥ 0
convention.

**Validation of the conversion:**
- States rebuilt from the zarr match the simulator to 1e-7.
- Replaying the derived actions open-loop reproduces the recorded outcomes: easy 73 % vs 74 %; hard 99.6 % vs
  99.8 %, with 99.4 % outcome agreement.

**Assumptions:** no SLAM drift, perfect gripper sensing, the scripted controller as the human, and the target
pose from the simulator (standing in for fiducial or BIM registration). Panel and target poses are perfect
estimates; no camera-based perception is implemented.

## 2. Training methodology

**Policy.**
- **Network:** MLP 3 × 256 with LayerNorm on normalized states.
- **Outputs:** 6 tanh motion dims and 1 suction logit, as **action chunks** of 8 steps, of which 4 are
  executed before re-planning.
- **Loss:** MSE + BCE, the negative log-likelihood of a fixed-variance Gaussian × Bernoulli.
- **Why chunking:** single-step BC reached 16 % validation success on the easy task; chunking reached 84 %
  (phase ambiguity and compounding error).

**Behaviour cloning:** trained on successful episodes ("filtered BC"), 30k steps.

**Offline RL:** Implicit Q-Learning (Kostrikov et al. 2021), **implemented from scratch** with each loss
documented against the paper's equations:
- expectile value regression (τ = 0.7);
- TD targets bootstrapped from V(s′), never from max_a′ Q;
- twin target critics;
- advantage-weighted regression for the actor with exp(βA), clipped at 100, β ∈ {3, 10}.

The actor *is* the BC network, so BC and IQL differ only in the per-sample weight. Training reads only the
fixed dataset; no `env.step` is called for learning.

**Hard-task IQL fixes** came after a diagnosis (Section 4):
- reward scale 0.01 instead of 0.1;
- 50/50 demo/sim batches;
- release targets weighted ×10 in the suction loss;
- 1M steps.

Optionally, the release threshold p(attach) is tuned on validation seeds only.

**Offline datasets.**
- **Easy:** 1,000 episodes from scripted plus noisy-BC behaviour (57 % success).
- **Hard:** 2,000 `explore` episodes (64 % success, 10 % drops, collisions in 41 %).
- **Stored as** D4RL-style transitions with every reward term kept separately. When UMI and sim data are
  combined, both are relabelled to the reward terms UMI data supports, since UMI has no collision, force or
  drop sensing.

**Protocol.**
- **Seed sets, disjoint:** data, validation and test seeds; the data scripts refuse test seeds.
- **Validation:** 50 episodes every 5k (BC) / 20k (IQL) steps, measurement only. Snapshots are saved for
  inspection, never for checkpoint selection; the reported model is always the last step.
- **Test:** 100 episodes per policy with 95 % Wilson intervals and an automatic failure-mode classifier.
- **Hard test sets:** one *inside* and one *outside* the UMI demos' region.

## 3. Experimental results

**Easy task** (test seeds 10000–10099):

| policy | data | success (95 % CI) | collisions |
|---|---|---|---|
| scripted controller | — | 100 % (96–100) | 0 % |
| BC, single-step | UMI demos | 23 % (16–32) | 2 % |
| **BC, chunked** | UMI demos | **80 % (71–87)** | 9 % |
| BC control | UMI + sim successes | 69 % (59–77) | 2 % |
| IQL β3 / β10 | sim | 58 % / 61 % | 22 / 14 % |
| **IQL β3** / β10 | UMI + sim | **71 %** / 68 % | 8 / 12 % |
| IQL β3, release threshold tuned (0.65) | UMI + sim | **81 % (72–87)** | 8 % |

**Hard task** (in = seeds 30000–30099 inside the demos' region; out = 31000–31099 outside):

| policy | data | in | out |
|---|---|---|---|
| scripted controller (demo mode) | — | 99 % (95–100) | 92 % (85–96) |
| BC, chunked | UMI demos (demo region) | 6 % (3–12) | **0 % (0–4)** |
| **BC control** | UMI + sim successes | **55 % (45–64)** | 20 % (13–29) |
| IQL β10, success 100 → 1000 | sim | 2 % → 16 % | 10 % → 19 % |
| IQL β3, success 100 → 1000 | sim | 21 % → 8 % | 11 % → 14 % |
| IQL β10, success 1000 | UMI + sim | 39 % (30–49) | 22 % (15–31) |
| IQL β10, same, 1M steps (ablation) | UMI + sim | 33 % (25–43) | 17 % (11–26) |
| **IQL fixed β3** / β1 / β10 | UMI + sim, balanced | **65 % (55–74)** / 50 / 57 % | 26 / **32 %** / 19 % |

Charts: `media/*/comparison*.png` and `media/*/training_*.png`. Per-episode results: `results/`.

**Findings.**
1. **Simulation data provides coverage that demonstrations lack.** BC on demos from a limited region scores
   0 % outside it (91 of 100 episodes never grasp). Every policy trained with simulated data generalizes
   outside the demo region (20–32 % for the best).
2. **Untuned offline RL did not beat behaviour cloning; tuned, it matched or edged past it.**
   - **Before tuning:** easy 71 % vs 69 % (BC control) and 80 % (BC on demos); hard 39 % / 22 % vs
     55 % / 20 %.
   - **After the fixes:** hard IQL reached 65 % / 26 %. Paired on the same test seeds that is +36 / −10
     episodes vs the old IQL (p < 0.001) and +31 / −21 vs the BC control (p = 0.21).
   - On the easy task, a validation-tuned release threshold lifted IQL to 81 % (+10 / −0, p = 0.002).
   - **Caveat:** at the new reward scale the β3 weights are nearly uniform (mean 1.01). Much of the gain may
     come from the balanced, release-weighted data rather than from the advantage weighting itself.
3. **Combining demos with sim data helps IQL** in both tasks (easy 71 vs 58–61 %; hard 39 vs 16 % in-region).
   For plain BC on the easy task, adding noisier sim successes hurt (80 → 69 %), while on the hard task it was
   essential (6 → 55 %).
4. **Reward balance matters, but isn't the whole story.** With success worth +100, the discounted success
   (≈ 22 from 150 steps out) was worth only about 70 steps of moderate error, and IQL learned to keep
   holding. Raising success ×10 improved β10 (2 → 16 % in-region, intervals disjoint) but not β3 (21 → 8 %).

## 4. Failure analysis

- **The last centimetres and the suction decision.** Nearly every learned-policy failure is there:
  - **Easy BC:** stalled about 9 mm above the panel without grasping (a regression fixed point: tiny
    demonstrated motions near the goal average to zero), or released 7 mm short of the wall.
  - **Easy IQL:** fixed the grasp (never-grasped 16–27 → 3–9) but released off the wall.
  - **Hard, BC on demos:** released mid-air (50 drops) or toggled suction at the target.
  - **Hard, sim-trained policies:** dominated by **"held at end"**, the panel carried but never placed.
    - Measured, it is mostly an approach failure: IQL's held episodes end a median 10–20 cm from the target.
      The BC control's end 3 cm away.
    - It occurs in BC too, which has no reward, so it is not only reward hacking.
- **Diagnosing hard IQL** (which led to the fixes):
  - **Saturated weights:** with success 1000 at reward × 0.1, 11–17 % of samples hit the exp(βA) clip.
  - **Rare release:** 1 in 144–224 suction-on states.
  - **Under-training:** validation was still rising at 200k steps.
  - **Training longer alone did not help** (39 → 33 %); the combination of fixes did (65 %).
- **Critic convergence.** At 200k steps the values still drift and validation swings ±15–20 % between
  checkpoints. At 1M steps the values are nearly flat, but validation still swings about ±10 %.
- **Collisions.** IQL policies collide more than the BC control trained on the same data (hard: 36–70 % vs
  19–34 % of episodes); they imitate the collision-heavy exploration data with only partial advantage
  reweighting. BC on the demos alone collides most (65–84 %), because it is mostly outside its training
  distribution.
- **Other limitations:**
  - A rigid panel and ideal suction.
  - Perfect pose estimates.
  - One training seed per configuration.
  - About 0.2 % of hard sim episodes hit a MuJoCo instability and were reset mid-episode (detected after the
    fact, documented).

## 5. Proposed deployment approach

Summarized from `docs/deployment_proposal.md`.
- **Hybrid architecture.** The learned policy (10 Hz, 8-step chunks, 4 executed) handles gross motion,
  carrying and approach, streamed as Cartesian pose targets to a 1 kHz impedance controller.
- **Verified final step.** The final approach becomes a force-controlled push to a set point, and the
  suction release is a *request* granted only when contact and support are confirmed. Our results show this
  is exactly where every learned policy fails, while explicit logic succeeds 92–100 %.
- **Main sim-to-real risks and mitigations:**
  - perception (fiducials, registration, tracking through the grasp);
  - contact dynamics (force control, system identification);
  - panel flex and suction slip (vacuum monitoring, a multi-cup frame or the second arm);
  - site localization (registration against BIM, a reachability check before acting).
- **Failure handling:** slip → safe set-down; wall mismatch → re-register or escalate; early contact → back
  off and re-align; policy uncertainty (ensemble disagreement, low critic value) → pause and hand over.
- **Field adaptation with 10–20 UMI demos:** co-train on site demos plus prior data, build a site digital
  twin for evaluation, then a staged validation ladder: open-loop replay → twin → dry run → dummy panel →
  supervised real panels. Each stage has explicit pass gates (for example ≥ 95 % success with Wilson lower
  bound ≥ 85 %, zero safety violations).

**Bottom line.** A small number of human demonstrations teaches the motion; simulation is what extends it to
layouts the demonstrations never covered. In this prototype offline RL added
success only after careful diagnosis and tuning, and not significantly beyond BC. The safety-critical
contact and release should not be left to the learned policy.
