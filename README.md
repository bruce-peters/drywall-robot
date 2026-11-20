# drywall-robot

This is a MuJoCo simulation of a Franka Panda arm with a suction gripper that picks a drywall panel off a rack and places it flush against a wall. On top of it are a UMI-style demonstration dataset, behaviour cloning (BC) and offline RL (IQL). This README only covers getting it running. The full development log (every decision, failure and fix) is in [LABNOTEBOOK.md](LABNOTEBOOK.md).

I developed and tested everything on Windows 11 with Python 3.12.4. Nothing in the code is Windows-specific, but I have not tested it on Linux or macOS, so the notes for those are best effort.

## 1. What you need

- **Python 3.10 or newer** (I used 3.12.4). On Linux, also install Tk for the viewers: `sudo apt install python3-tk`. The python.org installers for Windows and macOS already include it.
- **git**, plus an internet connection the first time you run anything (it downloads the robot model, see step 3).
- **About 5 GB of disk** for the Python packages (PyTorch with CUDA is most of it). Retraining needs about 10 GB more for the datasets.
- **A GPU is optional.** Testing the trained policies runs fine on a CPU. Retraining is where a GPU matters (an IQL run is about 5 h on an RTX 4070).

## 2. Install

```bash
git clone https://github.com/bruce-peters/drywall-robot.git
cd drywall-robot
python -m venv .venv
```

Activate the virtual environment (this has to be done in every new terminal):

| shell | command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux / macOS | `source .venv/bin/activate` |

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` first (it only lasts for that terminal).

Then install the packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every version and pulls PyTorch from the CUDA 12.8 index. If you don't have an NVIDIA GPU, pip still installs a working build and it just runs on the CPU.

## 3. Check that it works

```bash
python -m pytest -q
```

You should see `42 passed` (about 40 s). The first run is slower because `robot_descriptions` clones MuJoCo Menagerie (which has the Panda model) into `~/.cache/robot_descriptions`. After that no network is needed.

## 4. Look at the simulation

```bash
python scripts/run_scripted.py --seed 0 --video out.mp4     # one episode of the scripted controller, saved as out.mp4
python scripts/view_interactive.py --seed 0                 # 3D viewer with Play / Step / Reset (and manual control)
```

The viewers open a MuJoCo 3D window plus a small Tk control panel. On macOS the 3D window needs `mjpython` instead of `python` (MuJoCo's rule, not mine), for example `mjpython scripts/view_interactive.py`.

There are two versions of the task:
- **easy**: the rack and wall stay roughly in place (a few cm and degrees of randomization).
- **hard**: the wall can be anywhere within ±150° around the robot and the rack anywhere on a ring around it (even behind the robot). Add `--challenge hard` to most scripts to use it.

## 5. Test the trained policies (no data or GPU needed)

The trained checkpoints are already in the repo (`checkpoints/`), so you can test them right away. Each command below runs 100 held-out test episodes and took about 4 to 5 min on my machine (add `--n 5` for a quick check).

```bash
# best policy: BC with the wall frame and mixture head (93 % inside the demos' region, 74 % outside it)
python scripts/eval_policy.py --challenge hard --test in  --policy checkpoints/hard_challenge/bc_umi_sim_h8_wall_gmm.pt
python scripts/eval_policy.py --challenge hard --test ood --policy checkpoints/hard_challenge/bc_umi_sim_h8_wall_gmm.pt

# best offline RL policy: IQL with the wall frame (63 % / 53 %)
python scripts/eval_policy.py --challenge hard --test in  --policy checkpoints/hard_challenge/iql_umi_sim_fix_b3_wall.pt

# the scripted controller, for reference (99 % / 92 %)
python scripts/eval_policy.py --challenge hard --test in  --policy scripted
```

- `--test in` uses test seeds 30000–30099, inside the region the demonstrations cover. `--test ood` uses seeds 31000–31099, outside it.
- Results print as a table and get saved as JSON in `results/` (one entry per episode). My own results are in `results/hard_challenge/` to compare against.
- Add `--video media/check` to also save one success and one failure clip.
- Any `.pt` file in `checkpoints/` works with `--policy` (the `*.critics.pt` files are IQL critics, not policies).

To watch policies instead of just scoring them:

```bash
python scripts/view_policy.py --challenge hard
```

Pick a model and a seed in the panel, then press Play. **Run N** runs N seeds in a row and keeps a table of the results, and clicking a row replays that episode.

## 6. Retrain from scratch (optional)

The datasets are not in the repo (`data/` is git-ignored because some files are several GB). Every dataset comes from fixed seeds, so these commands rebuild them exactly. Times are for a 24-thread CPU and an RTX 4070.

**Build the hard-task datasets** (about 1.5 h in total):

```bash
# 500 UMI-style demonstrations (inside the demo region), then convert to UMI's zarr format
python scripts/record_umi.py --challenge hard --region demo --mode demo --n 500 --start-seed 20000 --max-noise 0 --out data/hard/umi_hard.npz
python scripts/convert_to_umi_zarr.py --npz data/hard/umi_hard.npz --out data/hard/umi_hard.zarr.zip

# convert the UMI demos into robot actions (the Part B pipeline)
python scripts/umi_to_transitions.py --challenge hard --region demo --zarr data/hard/umi_hard.zarr.zip --sidecar data/hard/umi_hard_sim_meta.npz --out data/hard/umi_transitions_hard.npz --replay 500

# 2,000 noisy simulated episodes for offline RL (about 4 min)
python scripts/generate_offline_data.py --challenge hard --out data/hard/offline_sim_hard.npz
```

**Train** (each saves to `checkpoints/` and logs to `runs/`):

```bash
# BC with the wall frame and mixture head (about 20 min on the GPU)
python scripts/train_bc.py --challenge hard --data data/hard/umi_transitions_hard.npz data/hard/offline_sim_hard.npz --chunk 8 --n-exec 4 --frame wall --head gmm --out checkpoints/hard_challenge/my_bc.pt

# IQL with the wall frame (about 5 h on the GPU; --resume picks up where it left off if it gets killed)
python scripts/train_iql.py --challenge hard --data data/hard/umi_transitions_hard.npz data/hard/offline_sim_hard.npz --reward-terms umi --chunk 8 --n-exec 4 --steps 1000000 --rollout-every 100000 --beta 3 --reward-scale 0.01 --balance --release-weight 10 --frame wall --resume --out checkpoints/hard_challenge/my_iql.pt
```

Then test them with `eval_policy.py` like in step 5, and plot the training curves with `python scripts/plot_training.py runs/hard_challenge/my_bc --out-dir media/hard_challenge`. Expect the trained numbers to come out a little different from mine on another machine (GPU training isn't bit-exact).

The easy task works the same way without `--challenge hard`. Its exact commands are in LABNOTEBOOK.md.

## 7. Where things are

```
drywall_env/        the environment (env.py), IK, scripted controller, UMI recording and conversion, policies, evaluation
scripts/            everything you run (record, convert, train, evaluate, plot, view)
tests/              pytest tests
checkpoints/        trained policies, per task (easy_challenge/, hard_challenge/)
results/            per-episode test results (JSON) for every policy
runs/               training logs and console output for every run
media/              training charts, comparison charts and labelled videos
data/               generated datasets (not in the repo, see step 6)
LABNOTEBOOK.md      the full development log
```

## Troubleshooting

- **`MUJOCO_LOG.TXT` shows up with "simulation unstable" warnings.** This happens in a handful of episodes when a policy presses the panel hard into the wall. Those episodes count as failures and the run keeps going, so it's safe to ignore.
- **The 3D viewer doesn't open on macOS.** Use `mjpython` instead of `python` (see step 4).
- **`No module named tkinter` on Linux.** Run `sudo apt install python3-tk` and recreate the virtual environment.
- **The first run hangs on "cloning".** That's the one-time Menagerie download (step 3). It needs git and network access.
