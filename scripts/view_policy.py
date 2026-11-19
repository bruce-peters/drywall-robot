"""Interactive policy viewer: run the final models (scripted, BC, IQL) one episode at a time or as a
batch of consecutive test seeds, in the MuJoCo 3D viewer with a Tk control panel.

    python scripts/view_policy.py                  # models = scripted + checkpoints/**/*.pt
    python scripts/view_policy.py --seed 10000

Single test: choose a model and a seed, then Play / Step / Reset (live readout as in
view_interactive.py).
Many tests: "Run N" runs N consecutive seeds from the seed box with the chosen model. Tick "watch" to see
each episode in 3D at the speed slider's rate; untick it to run fast (the 3D view updates every few
steps). Each finished episode is added to the results table, and the summary line shows the totals. Click
a row to replay that episode (same model, same seed) in 3D.

Default seeds are the held-out TEST seeds (10000-10099), the same ones as the comparison table.
Threading: as in view_interactive.py, everything runs on the Tk thread (root.after loop), and env
access is wrapped in viewer.lock().
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, EnvConfig  # noqa: E402
from drywall_env.scripted_hard import make_scripted  # noqa: E402
from drywall_env.evaluation import summarize  # noqa: E402
from drywall_env.policies import ScriptedPolicy, load_policy  # noqa: E402
from drywall_env.transitions import HARD_TEST_IN_SEEDS, HARD_TEST_OOD_SEEDS, TEST_SEEDS  # noqa: E402

REGIONS = ["demo", "ood", "full"]  # hard challenge layout regions (demo = the UMI demos' coverage)

CONTROL_HZ = 10.0
IDLE_MS = 50
FAST_STEPS_PER_TICK = 25
READOUTS = ["model", "seed", "step", "phase", "pos error (mm)", "ori error (deg)", "attached",
            "fixture_active", "collision", "last reward", "status"]
COLUMNS = [("seed", 60), ("model", 240), ("result", 80), ("pos mm", 70), ("ori deg", 65),
           ("collision", 70), ("steps", 50)]


def find_models(ckpt_dir="checkpoints", challenge: str | None = None, snapshots: bool = False) -> list[str]:
    """Final checkpoints of a challenge; with snapshots=True also every validation-step snapshot
    (snapshots/<run>/step_*.pt; the resume file train_state.pt there is not an actor)."""
    root = Path(ckpt_dir) / f"{challenge}_challenge" if challenge else Path(ckpt_dir)
    finals = sorted(p for p in root.rglob("*.pt") if "snapshots" not in p.parts
                    and not p.name.endswith(".critics.pt"))
    snaps = sorted(root.rglob("snapshots/*/step_*.pt")) if snapshots else []
    return ["scripted"] + [str(p) for p in finals + snaps]  # finals first: the default model is a final one


class App:
    def __init__(self, root: tk.Tk, seed: int, models: list[str], challenge: str = "easy"):
        self.root = root
        self.challenge = challenge
        self.env = DrywallPlacementEnv(EnvConfig(challenge=challenge, region="demo" if challenge == "hard" else "full"))
        self.models = models
        self._policies: dict[str, object] = {}
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.playing = False  # stepping loop active (single episode or batch)
        self.batch: list[int] = []  # seeds still to run in the current batch
        self.closed = False
        self._after_id = None
        self.rows: list[dict] = []
        self._build_ui(seed)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset()
        self._tick()

    # ------------------------------------------------------------------ policies
    def policy(self, spec: str):
        if spec not in self._policies:
            if spec == "scripted":
                pol = ScriptedPolicy(self.env.cfg)
                if self.challenge == "hard":  # demo-mode hard controller
                    pol.ctrl = make_scripted(self.env.cfg, mode="demo", seed=0)
                self._policies[spec] = pol
            else:
                self._policies[spec] = load_policy(spec)
        return self._policies[spec]

    @staticmethod
    def short(spec: str) -> str:
        if spec == "scripted":
            return spec
        p = Path(spec)
        if "snapshots" in p.parts:  # checkpoints/<c>_challenge/snapshots/<run>/step_000123.pt
            tag = p.parents[2].name.replace("_challenge", "")
            return f"{tag}/{p.parent.name}@{p.stem.replace('step_', '')}"
        tag = p.parent.name.replace("_challenge", "")  # easy / hard
        return f"{tag}/{p.stem}" if tag and tag != "checkpoints" else p.stem

    # ------------------------------------------------------------------ UI
    def _build_ui(self, seed):
        r = self.root
        r.title("drywall-robot policy viewer")
        top = tk.Frame(r)
        top.pack(fill="x", padx=8, pady=4)
        tk.Label(top, text="model").pack(side="left")
        finals = [m for m in self.models if "snapshots" not in Path(m).parts]
        self.model_var = tk.StringVar(value=finals[-1])
        ttk.Combobox(top, textvariable=self.model_var, values=self.models, width=72,
                     state="readonly").pack(side="left", padx=4)
        tk.Label(top, text="seed").pack(side="left", padx=(8, 2))
        self.seed_var = tk.StringVar(value=str(seed))
        tk.Entry(top, textvariable=self.seed_var, width=7).pack(side="left")
        self.region_var = tk.StringVar(value="demo")
        if self.challenge == "hard":
            tk.Label(top, text="region").pack(side="left", padx=(8, 2))
            cb = ttk.Combobox(top, textvariable=self.region_var, values=REGIONS, width=6, state="readonly")
            cb.pack(side="left")
            # switching region also jumps to that region's test seeds
            cb.bind("<<ComboboxSelected>>", lambda _e: self.seed_var.set(str(
                {"demo": HARD_TEST_IN_SEEDS[0], "ood": HARD_TEST_OOD_SEEDS[0]}.get(self.region_var.get(),
                                                                                   self.seed_var.get()))))

        bar = tk.Frame(r)
        bar.pack(fill="x", padx=8, pady=2)
        self.play_btn = tk.Button(bar, text="Play", width=7, command=self.toggle_play)
        self.play_btn.pack(side="left")
        tk.Button(bar, text="Step", width=7, command=self.step).pack(side="left", padx=4)
        tk.Button(bar, text="Reset", width=7, command=self.reset).pack(side="left")
        tk.Label(bar, text="  N").pack(side="left")
        self.n_var = tk.StringVar(value="20")
        tk.Entry(bar, textvariable=self.n_var, width=5).pack(side="left")
        tk.Button(bar, text="Run N", width=7, command=self.run_batch).pack(side="left", padx=4)
        self.watch = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="watch", variable=self.watch).pack(side="left")
        tk.Button(bar, text="Stop", width=6, command=self.stop).pack(side="left", padx=4)
        tk.Button(bar, text="Clear", width=6, command=self.clear).pack(side="left")

        self.speed = tk.DoubleVar(value=1.0)
        tk.Scale(r, label="speed (x real time, watch mode)", variable=self.speed, from_=0.25, to=8.0,
                 resolution=0.25, orient="horizontal").pack(fill="x", padx=8)

        grid = tk.Frame(r)
        grid.pack(fill="x", padx=8, pady=4)
        self.readout = {}
        for i, name in enumerate(READOUTS):
            tk.Label(grid, text=name, anchor="w").grid(row=i % 6, column=2 * (i // 6), sticky="w", padx=(0, 6))
            self.readout[name] = tk.StringVar()
            tk.Label(grid, textvariable=self.readout[name], anchor="w", width=30,
                     font=("Courier", 10)).grid(row=i % 6, column=2 * (i // 6) + 1, sticky="w")

        self.summary_var = tk.StringVar(value="no finished episodes yet")
        tk.Label(r, textvariable=self.summary_var, anchor="w", font=("Courier", 10), justify="left").pack(
            fill="x", padx=8)
        frame = tk.Frame(r)
        frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.table = ttk.Treeview(frame, columns=[c for c, _ in COLUMNS], show="headings", height=14)
        for c, w in COLUMNS:
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.table.tag_configure("ok", foreground="#1f7a3a")
        self.table.tag_configure("bad", foreground="#b02a2a")
        self.table.bind("<<TreeviewSelect>>", self._on_select)

    def _refresh(self):
        info, ro = self.info, self.readout
        ro["model"].set(self.short(self.cur_model))
        ro["seed"].set(str(self.cur_seed))
        ro["step"].set("0" if info is None else str(info["steps"]))
        ro["phase"].set(getattr(self.pol, "phase", "-"))
        ro["pos error (mm)"].set("-" if info is None else f"{info['pos_error'] * 1000:.1f}")
        ro["ori error (deg)"].set("-" if info is None else f"{np.degrees(info['ori_error']):.2f}")
        ro["attached"].set(str(self.env.attached))
        ro["fixture_active"].set(str(self.env.fixture_active))
        ro["collision"].set(str(self.ep_collision))
        ro["last reward"].set("-" if self.reward is None else f"{self.reward:.3f}")
        ro["status"].set(self.status + (f"  (batch: {len(self.batch)} left)" if self.batch else ""))

    def _update_summary(self):
        if not self.rows:
            self.summary_var.set("no finished episodes yet")
            return
        by_model = {}
        for r in self.rows:
            by_model.setdefault(r["model"], []).append(r)
        lines = []
        for m, rs in by_model.items():
            s = summarize(rs)
            lines.append(f"{m:<16} n={s['n']:3d}  success {100 * s['success_rate']:5.1f} %  "
                         f"pos {s['pos_error_mm_median']:6.1f} mm (median)  collision {100 * s['collision_rate']:4.0f} %  "
                         f"steps {s['length_mean']:5.1f}")
        self.summary_var.set("\n".join(lines))

    # ------------------------------------------------------------------ episode control
    def _start_episode(self, model: str, seed: int):
        self.cur_model, self.cur_seed = model, seed
        if self.challenge == "hard":
            self.env.cfg.region = self.region_var.get()  # layout sampling region for this reset
        self.pol = self.policy(model)
        with self.viewer.lock():
            self.obs, _ = self.env.reset(seed=seed)
        self.pol.reset()
        self.viewer.sync()
        self.info, self.reward, self.done, self.status = None, None, False, "running"
        self.ep_collision, self.ep_max_force = False, 0.0
        self._refresh()

    def reset(self):
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            self.status = "bad seed"
            return
        self.batch = []
        self._set_playing(False)
        self._start_episode(self.model_var.get(), seed)

    def step(self, sync: bool = True):
        if self.done or self.closed:
            return
        action = self.pol.act(self.obs)
        with self.viewer.lock():
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        if sync:
            self.viewer.sync()
        self.ep_collision |= self.info["collision"]
        self.ep_max_force = max(self.ep_max_force, self.info["max_panel_wall_force"])
        if terminated or truncated:
            self._finish_episode(truncated)
        if sync:
            self._refresh()

    def _finish_episode(self, truncated: bool):
        self.done = True
        info = self.info
        self.status = "SUCCESS" if info["success"] else "DROPPED" if info["dropped"] else "TRUNCATED"
        row = {"seed": self.cur_seed, "model": self.short(self.cur_model), "spec": self.cur_model,
               "success": bool(info["success"]), "dropped": bool(info["dropped"]), "truncated": truncated,
               "pos_error_mm": 1000 * info["pos_error"], "ori_error_deg": float(np.degrees(info["ori_error"])),
               "collision": self.ep_collision, "max_wall_force": self.ep_max_force, "length": info["steps"],
               "return": 0.0}
        self.rows.append(row)
        iid = self.table.insert("", "end", values=(
            row["seed"], row["model"], self.status, f"{row['pos_error_mm']:.1f}", f"{row['ori_error_deg']:.2f}",
            "yes" if row["collision"] else "no", row["length"]), tags=("ok" if row["success"] else "bad",))
        self.table.see(iid)
        self._update_summary()
        if self.batch:  # next episode of the batch
            self._start_episode(self.cur_model, self.batch.pop(0))
        else:
            self._set_playing(False)
            self.viewer.sync()
            self._refresh()

    def run_batch(self):
        try:
            seed0, n = int(self.seed_var.get()), int(self.n_var.get())
        except ValueError:
            self.status = "bad seed / N"
            return
        seeds = list(range(seed0, seed0 + max(1, n)))
        self.batch = seeds[1:]
        self._start_episode(self.model_var.get(), seeds[0])
        self._set_playing(True)

    def stop(self):
        self.batch = []
        self._set_playing(False)
        self._refresh()

    def clear(self):
        self.rows = []
        self.table.delete(*self.table.get_children())
        self._update_summary()

    def _on_select(self, _event):
        sel = self.table.selection()
        if not sel or self.playing:
            return
        idx = self.table.index(sel[0])
        row = self.rows[idx]
        self.batch = []
        self.watch.set(True)
        self._start_episode(row["spec"], row["seed"])
        self._set_playing(True)

    def toggle_play(self):
        if self.done:
            return
        self._set_playing(not self.playing)

    def _set_playing(self, on: bool):
        self.playing = on
        self.play_btn.config(text="Pause" if on else "Play")

    # ------------------------------------------------------------------ loop / shutdown
    def _tick(self):
        if self.closed:
            return
        if not self.viewer.is_running():
            self.close()
            return
        delay = IDLE_MS
        if self.playing:
            t0 = time.perf_counter()
            if self.watch.get():
                self.step()
                period_ms = 1000.0 / (CONTROL_HZ * self.speed.get())
                delay = max(1, int(period_ms - 1000.0 * (time.perf_counter() - t0)))
            else:
                for _ in range(FAST_STEPS_PER_TICK):
                    if not self.playing:
                        break
                    self.step(sync=False)
                self.viewer.sync()
                self._refresh()
                delay = 1
        self._after_id = self.root.after(delay, self._tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.viewer.close()
        self.env.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenge", choices=["easy", "hard"], default="easy")
    ap.add_argument("--seed", type=int, default=None, help="default: first test seed (10000 easy / 30000 hard)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="default: scripted + checkpoints/<challenge>_challenge/**/*.pt")
    ap.add_argument("--snapshots", action="store_true",
                    help="also list the actor saved at every validation step (viewing only)")
    args = ap.parse_args()
    models = args.models or find_models(challenge=args.challenge, snapshots=args.snapshots)
    seed = args.seed if args.seed is not None else (HARD_TEST_IN_SEEDS[0] if args.challenge == "hard" else TEST_SEEDS[0])
    root = tk.Tk()
    App(root, seed, models, args.challenge)
    root.mainloop()


if __name__ == "__main__":
    main()
