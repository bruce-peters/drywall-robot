"""Interactive 3D viewer (MuJoCo passive viewer) + Tkinter control panel.

    python scripts/view_interactive.py --seed 0

Threading: all env access happens on the Tk main thread (stepping loop driven by root.after).
The passive viewer renders on its own internal thread; env.step / env.reset are wrapped in
viewer.lock() so it never renders a half-updated model/data, and viewer.sync() is called after
every change.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv, ScriptedController  # noqa: E402

CONTROL_HZ = 10.0  # env control rate (EnvConfig.substeps * timestep = 0.1 s)
IDLE_MS = 50  # poll interval while paused (to notice the viewer window closing)
MANUAL_AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
READOUTS = ["step", "phase", "pos error (mm)", "ori error (deg)", "attached", "fixture_active",
            "collision", "last reward", "status"]


class App:
    def __init__(self, root: tk.Tk, seed: int):
        self.root = root
        self.env = DrywallPlacementEnv()
        self.ctrl = ScriptedController(self.env.cfg)
        self.obs, _ = self.env.reset(seed=seed)
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.viewer.sync()
        self.playing = False
        self.done = False
        self.closed = False
        self.info = None
        self.reward = None
        self.status = "running"
        self._after_id = None

        self._build_ui(seed)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh()
        self._tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, seed):
        r = self.root
        r.title("drywall-robot control")

        bar = tk.Frame(r)
        bar.pack(fill="x", padx=8, pady=4)
        self.play_btn = tk.Button(bar, text="Play", width=7, command=self.toggle_play)
        self.play_btn.pack(side="left")
        tk.Button(bar, text="Step", width=7, command=self.step).pack(side="left", padx=4)
        tk.Button(bar, text="Reset", width=7, command=self.reset).pack(side="left")
        tk.Label(bar, text="seed").pack(side="left", padx=(8, 2))
        self.seed_var = tk.StringVar(value=str(seed))
        tk.Entry(bar, textvariable=self.seed_var, width=6).pack(side="left")

        self.speed = tk.DoubleVar(value=1.0)
        tk.Scale(r, label="speed (x real time)", variable=self.speed, from_=0.25, to=4.0,
                 resolution=0.25, orient="horizontal").pack(fill="x", padx=8)

        grid = tk.Frame(r)
        grid.pack(fill="x", padx=8, pady=4)
        self.readout = {}
        for i, name in enumerate(READOUTS):
            tk.Label(grid, text=name, anchor="w").grid(row=i, column=0, sticky="w")
            self.readout[name] = tk.StringVar()
            tk.Label(grid, textvariable=self.readout[name], anchor="w", width=16,
                     font=("Courier", 10)).grid(row=i, column=1, sticky="w")

        self.manual = tk.BooleanVar(value=False)
        tk.Checkbutton(r, text="Manual", variable=self.manual,
                       command=self._on_manual_toggle).pack(anchor="w", padx=8)
        man = tk.Frame(r)
        man.pack(fill="x", padx=8, pady=(0, 8))
        self.axes = []
        for name in MANUAL_AXES:
            v = tk.DoubleVar(value=0.0)
            s = tk.Scale(man, label=name, variable=v, from_=-1.0, to=1.0, resolution=0.05,
                         orient="horizontal")
            s.bind("<ButtonRelease-1>", lambda _e, v=v: v.set(0.0))  # spring back to 0
            s.pack(fill="x")
            self.axes.append(v)
        self.suction = tk.BooleanVar(value=False)
        tk.Checkbutton(man, text="suction", variable=self.suction).pack(anchor="w")

    def _refresh(self):
        info, ro = self.info, self.readout
        ro["step"].set("0" if info is None else str(info["steps"]))
        ro["phase"].set("manual" if self.manual.get() else self.ctrl.phase)
        ro["pos error (mm)"].set("-" if info is None else f"{info['pos_error'] * 1000:.1f}")
        ro["ori error (deg)"].set("-" if info is None else f"{np.degrees(info['ori_error']):.2f}")
        ro["attached"].set(str(self.env.attached))
        ro["fixture_active"].set(str(self.env.fixture_active))
        ro["collision"].set("-" if info is None else str(info["collision"]))
        ro["last reward"].set("-" if self.reward is None else f"{self.reward:.3f}")
        ro["status"].set(self.status)

    # ------------------------------------------------------------------ actions
    def toggle_play(self):
        self.playing = not self.playing and not self.done
        self.play_btn.config(text="Pause" if self.playing else "Play")

    def _action(self):
        if self.manual.get():
            return np.array([v.get() for v in self.axes] + [1.0 if self.suction.get() else -1.0],
                            dtype=np.float32)
        return self.ctrl.act(self.obs)

    def step(self):
        """Advance exactly one control step (one env.step)."""
        if self.done or self.closed:
            return
        action = self._action()
        with self.viewer.lock():
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.viewer.sync()
        if terminated or truncated:
            self.done = True
            self.status = ("SUCCESS" if self.info["success"] else
                           "DROPPED" if self.info["dropped"] else "TRUNCATED")
            if self.playing:
                self.toggle_play()
        self._refresh()

    def reset(self):
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            self.status = "bad seed"
            self._refresh()
            return
        with self.viewer.lock():
            self.obs, _ = self.env.reset(seed=seed)
        self.ctrl.reset()
        self.viewer.sync()
        self.info, self.reward, self.done, self.status = None, None, False, "running"
        if self.playing:
            self.toggle_play()
        self._refresh()

    def _on_manual_toggle(self):
        if not self.manual.get():
            self.ctrl.reset()  # back to scripted: restart from "approach"
        self._refresh()

    # ------------------------------------------------------------------ loop / shutdown
    def _tick(self):
        if self.closed:
            return
        if not self.viewer.is_running():  # viewer window was closed
            self.close()
            return
        delay = IDLE_MS
        if self.playing:
            t0 = time.perf_counter()
            self.step()
            period_ms = 1000.0 / (CONTROL_HZ * self.speed.get())
            delay = max(1, int(period_ms - 1000.0 * (time.perf_counter() - t0)))
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
