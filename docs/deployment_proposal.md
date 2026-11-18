# Part D — Deployment and sim-to-real proposal

How the policy pipeline in this repository (UMI-style demonstrations → robot actions → BC / offline RL,
evaluated in a MuJoCo work cell) would be taken to a physical robot on a construction site. Where a
recommendation comes from something we measured in simulation, the result is cited.

**The main recommendation.** Use the learned policy for the parts of the task where it is reliable:
gross motion, carrying, approaching the wall. Hand the last centimetres of contact and the suction
release to a force-controlled, verifiable routine. In simulation every learned policy failed mostly in those
last millimetres:
- **BC** stalled about 9 mm above the panel and never grasped, or released 7 mm short of the wall.
- **IQL** fixed the grasp but released off the wall.
- The scripted controller with explicit contact logic succeeded 100 % on the same test seeds.

## A. Sim-to-real transfer

| difference | why it matters here | mitigation |
|---|---|---|
| **Perception** (largest gap) | The sim gives exact panel and target poses. A real system estimates them, and the panel is occluded by the plate while carried. | Fiducials on the panels and a registered wall layout (total station or BIM); a 6D pose estimator on the external RGB-D camera as backup. Once grasped, track the panel through kinematics and the grasp transform recorded at attach (the 46-D state's panel-in-hand feature). Train with perception noise and latency. |
| **Contact dynamics** | The sim pushes with a stiff position servo (up to 46 N), and success depends on a 2 mm wall gap. Real walls are uneven (studs, tape, mud), compliant and not exactly where the plan says. | Force-controlled final approach (see B). Randomize wall compliance and flatness in sim. Measure the real contact stiffness (system ID). |
| **Panel deformation** | A rigid 0.6 × 0.4 m panel in sim; a real 1.2 × 2.4 m, ~25 kg sheet sags and flexes. | Multi-cup vacuum frame or the second arm (the spec's bimanual robot); flexible-body sim or a stiffness model for the full-size panel; slow accelerations. |
| **Grasp slip** | Suction is an ideal weld in sim, with no leak, peel or slip under moment load. | Vacuum-pressure monitoring (the `attached` signal becomes a pressure threshold); carry orientations that limit peel moment; slip detection (see C). |
| **Camera calibration** | The wrist fisheye and external camera are perfectly calibrated in sim. | Per-site calibration with targets; check reprojection error every shift; augment camera extrinsics in training if images are used. |
| **Wall geometry / localization** | The robot base and wall are known exactly in sim. On site, the base is wherever it was parked. | Register base and wall on every set-up (fiducials or a laser tracker against BIM); check reachability before acting (our sweep: the arm faces the wall only at panel heights ≈ 0.5–0.75 m and about ±0.25 m sideways). Reposition the base rather than stretch. |

The UMI side has its own gaps: SLAM drift, fiducial-based gripper sensing, and real human hesitation and
multimodality. We assumed none of them. The coordinate conversion is frame-agnostic: it was tested with an
arbitrary map frame, and only relative poses matter.

## B. Closed-loop control

- **Loop structure:** the policy runs at **10 Hz** on the latest observation. It predicts an **8-step action
  chunk (0.8 s)** of base-frame end-effector deltas and executes **4 steps** before re-planning, the receding
  horizon that took BC from 16 % to 84 % in validation.
  - Deltas are integrated into a *commanded* pose on the host (as in the env) and clipped to the workspace.
  - The pose is streamed to the robot's **1 kHz Cartesian impedance controller** (libfranka, or `servoL` on a
    UR).
  - Joint-level IK is the vendor's, not ours.
- **Latency matching (UMI):** measure camera-to-action latency on the real robot and drop the predicted
  actions whose timestamps have already passed. Re-planning every 0.4 s keeps the open-loop segment short.
- **Force limits:**
  1. Impedance with low translational stiffness along the wall normal during approach and contact.
  2. Below about 5 cm from the wall, the policy's normal-axis command is replaced by a **force-controlled push
     to a set point** (for example 20–30 N), stopping on force, not position.
  3. A hard wrist F/T limit (our sim penalty starts at 50 N) triggers stop-and-retract.
  4. Velocity and acceleration caps scaled by panel mass; the per-step delta limits (2 cm, 3°) stay as a
     safety envelope.
- **Release:** only when wall contact is confirmed (force set point reached and stable, panel-to-wall gap
  ≤ 2 mm from perception) **and** the fixture or ledge is supporting the panel. The suction bit from the
  policy is a *request*; the controller checks the conditions before acting on it.

## C. Handling failure

| event | detection | response |
|---|---|---|
| **Panel slips during transport** | Vacuum pressure drop; panel pose in the hand frame drifting from the grasp recorded at attach. | Stop arm motion, keep vacuum, move slowly to the nearest safe set-down (rack or floor pad) along a pre-checked path; never continue to the wall with a degraded grasp. Log the episode for retraining. |
| **Target wall location differs from the simulated environment** | Registered wall pose vs plan; the target outside the reachability band or the training distribution. | Small differences are handled by design: the target pose is an input, and the hard challenge trains on walls all around the robot. Large ones (outside the band, or a novelty score such as distance to the training-state support or critic value well below normal): stop and ask the operator to move the base or confirm. Our hard experiment measures in- vs out-of-demo-distribution layouts. |
| **Panel touches the wall before alignment** | Unexpected F/T contact; perceived panel more than 5 cm (in-plane) from the target while touching (the env's collision definition). | Back off 3–5 cm along the wall normal, re-run pre-alignment, retry up to N times, then escalate. |
| **Policy uncertain or unsafe** | Disagreement across an ensemble or across overlapping action chunks; low critic value (IQL's V/Q); action or force limits hit repeatedly; IK residual high. | A safety filter clips every command (workspace, velocity, force). On sustained uncertainty, pause in a safe pose and hand over to a scripted routine or a teleoperator. A human supervisor holds an e-stop at all times during supervised deployment. |

## D. Field deployment with 10–20 new UMI demonstrations

1. **Site set-up (½ day):** register robot base, wall, racks and panel fiducials against the drawings;
   calibrate cameras; run the reachability check for every planned panel position.
2. **Collect 10–20 UMI demos on this site**, spread over the variations that matter here: rack positions, wall
   heights, lighting, and the actual panel type. Convert them with the same pipeline (map → base frame,
   resample to 10 Hz, deltas, gripper mapping, workspace check).
3. **Adapt:** fine-tune the pretrained policy on these demos mixed with its original data (co-training,
   low learning rate or adapters), so 20 demos shift it without overwriting it. In parallel, build a **digital
   twin** of the site (wall pose, rack positions) and generate simulated offline data there.
   - Use it as validation first. As offline-RL data, only after a check on held-out layouts: in our easy
     experiment, IQL on sim + demo data didn't beat BC on the demos alone (71 % vs 80 %).
4. **Validate in stages** with explicit gates:
   1. Open-loop replay of the new demos.
   2. ≥ 100 sim episodes on held-out twin layouts.
   3. Real dry runs with no panel.
   4. A lightweight dummy panel.
   5. Real panels at reduced speed with the e-stop held.

   Each stage must pass before the next: for example ≥ 95 % success with Wilson lower bound ≥ 85 %, zero force
   or workspace violations, and no unexplained failure modes.
5. **Supervised deployment readiness** is decided on the real-panel stage alone, with every failure classified
   (the same categories as our evaluation: never grasped / released early / misaligned / dropped / collision)
   and fed back into the data. Unsupervised operation is out of scope until the force-controlled contact,
   perception and slip detection have their own validation.
