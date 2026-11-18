"""Browse a UMI-style zarr dataset (scripts/convert_to_umi_zarr.py): many episodes at once.

    python scripts/view_dataset.py --zarr data/umi_sim.zarr.zip --episodes 0-49
    python scripts/view_dataset.py --episodes 0-19 --camera wrist --overlay attached,error --3d

Tk window:
  * trajectory plots: EE paths of the selected episodes, top view (x-y) and side view (x-z), green =
    success, red = failure, with a moving marker per episode at the current time
  * camera grid (optional): one tile per episode, wrist fisheye or external camera
  * overlays (toggle): attached = thick path segments / "SUCTION" badge on tiles;
    error = panel-to-target error as marker colour / text on tiles
  * 3D replay (button or --3d): MuJoCo viewer with the robot re-posed (IK) from one focus
    episode's recorded EE poses, plus translucent ghost panels and EE paths for all selected
    episodes. Ghost panels need the sidecar <name>_sim_meta.npz (per-episode target offsets).
Time is per-episode (all episodes start together); an episode that has ended holds its last frame.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drywall_env import DrywallPlacementEnv  # noqa: E402
from drywall_env.env import PANEL_HALF  # noqa: E402
from drywall_env.transforms import pose_mul, quat_to_mat, rotvec_to_quat  # noqa: E402

ROW_HZ = 10.0
GREEN, RED = "#2e9d4f", "#d1453b"
PLOT_W, PANEL_H = 620, 680  # trajectory canvas; two panels of PANEL_W x PANEL_H side by side
PANEL_W = PLOT_W // 2


def parse_episodes(spec: str, n_total: int) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    out = [e for e in dict.fromkeys(out) if 0 <= e < n_total]
    if not out:
        raise SystemExit(f"no valid episodes in '{spec}' (dataset has {n_total})")
    return out


def error_colour(err_m: float) -> str:
    """0 mm blue -> 50 mm pale yellow -> >=100 mm red."""
    stops = [(0.0, (44, 123, 182)), (0.05, (255, 255, 191)), (0.10, (215, 25, 28))]
    e = min(max(err_m, 0.0), stops[-1][0])
    for (e0, c0), (e1, c1) in zip(stops, stops[1:]):
        if e <= e1:
            t = (e - e0) / (e1 - e0)
            return "#%02x%02x%02x" % tuple(int(a + t * (b - a)) for a, b in zip(c0, c1))
    return "#d7191c"


class Dataset:
    def __init__(self, zarr_path: str, meta_path: str | None):
        import zarr
        from imagecodecs.numcodecs import register_codecs

        register_codecs(verbose=False)
        self.store = zarr.ZipStore(zarr_path, mode="r")
        root = zarr.open(self.store, mode="r")
        self.data = root["data"]
        ends = root["meta/episode_ends"][:]
        self.starts = np.concatenate([[0], ends[:-1]]).astype(int)
        self.lengths = (ends - self.starts).astype(int)
        self.n = len(ends)
        self.success = (root["meta/episode_success"][:] if "episode_success" in root["meta"]
                        else np.ones(self.n, bool))
        d = self.data
        self.pos = d["robot0_eef_pos"][:]
        self.rot = d["robot0_eef_rot_axis_angle"][:]
        self.grip = d["robot0_gripper_width"][:, 0]
        self.has_panel = "panel_pos_in_target" in d
        if self.has_panel:
            self.ppos = d["panel_pos_in_target"][:]
            self.prot = d["panel_rot_axis_angle_in_target"][:]
        self.cams = {k: k for k in ("camera0_rgb", "external_rgb") if k in d}
        self.meta = None
        if meta_path is None:
            cand = Path(zarr_path.replace(".zarr.zip", "_sim_meta.npz"))
            meta_path = str(cand) if cand.exists() else None
        if meta_path:
            m = np.load(meta_path)
            self.meta = {k: m[k] for k in m.files}
            if len(self.meta["seed"]) != self.n:
                print(f"warning: sidecar has {len(self.meta['seed'])} episodes, zarr {self.n}; ignoring it")
                self.meta = None
        self._img_cache: dict = {}

    def row(self, ep: int, k: int) -> int:
        return int(self.starts[ep] + min(k, self.lengths[ep] - 1))

    def image(self, cam: str, row: int) -> np.ndarray:
        key = (cam, row)
        if key not in self._img_cache:
            if len(self._img_cache) > 4000:
                self._img_cache.clear()
            self._img_cache[key] = self.data[cam][row]
        return self._img_cache[key]

    def panel_error(self, row: int) -> tuple[float, float]:
        if not self.has_panel:
            return float("nan"), float("nan")
        return float(np.linalg.norm(self.ppos[row])), float(np.linalg.norm(self.prot[row]))


class Replay3D:
    """MuJoCo passive viewer: focus episode re-posed by IK + ghost panels / paths for all episodes."""

    def __init__(self, ds: Dataset, episodes: list[int]):
        self.ds, self.episodes = ds, episodes
        self.env = DrywallPlacementEnv()
        self.focus = None
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        self.env.reset(seed=0)
        self._nom_target = self.env._nom_target_pos.copy()
        _, self._target_quat = self.env._target_pose_world()
        self._target_quat = self._target_quat.copy()

    def running(self) -> bool:
        return self.viewer.is_running()

    def _target_pose(self, ep: int):
        """World pose of an episode's target (nominal + its sampled offset)."""
        m = self.ds.meta
        if m is None:
            return None
        body_off = np.array([0.0, m["target_dy"][ep], m["target_dz"][ep]])
        site_off = self.env.data.site_xpos[self.env._target_site] - self.env.data.xpos[self.env._target_body]
        return self._nom_target + body_off + site_off, self._target_quat

    def _set_focus(self, ep: int):
        seed = int(self.ds.meta["seed"][ep]) if self.ds.meta is not None else 0
        self.env.reset(seed=seed)  # same target/rack layout as when the episode was recorded
        self._q = self.env.data.qpos[self.env._qadr].copy()
        self.focus = ep

    def update(self, k: int, focus: int, show_attached: bool, show_error: bool):
        ds, env = self.ds, self.env
        m, d = env.model, env.data
        with self.viewer.lock():
            if focus != self.focus:
                self._set_focus(focus)
            r = ds.row(focus, k)
            ee_p, ee_q = ds.pos[r].astype(float), rotvec_to_quat(ds.rot[r])
            for _ in range(3):  # warm-started DLS; extra passes absorb slider jumps
                self._q, _ = env.ik.solve(self._q, ee_p, ee_q)
            d.qpos[env._qadr] = self._q
            tp = self._target_pose(focus) if ds.has_panel else None
            if tp is None:  # no sidecar: approximate the target with the env's own reset layout
                tp = env._target_pose_world()
            pp, pq = pose_mul(tp[0], tp[1], ds.ppos[r], rotvec_to_quat(ds.prot[r])) if ds.has_panel else (None, None)
            if pp is not None:
                a = env._panel_qadr
                d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pp, pq
            d.eq_active[:] = 0
            mujoco.mj_forward(m, d)
            self._draw_ghosts(k, focus, show_attached, show_error)
        self.viewer.sync()

    def _draw_ghosts(self, k, focus, show_attached, show_error):
        ds, scn = self.ds, self.viewer.user_scn
        scn.ngeom = 0
        budget = scn.maxgeom - len(self.episodes) - 1
        seg_total = sum(ds.lengths[e] for e in self.episodes)
        stride = max(1, int(np.ceil(seg_total / max(budget, 1))))

        def add(gtype, size, pos, mat, rgba):
            if scn.ngeom >= scn.maxgeom:
                return None
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, gtype, size, pos, mat, rgba)
            scn.ngeom += 1
            return g

        for ep in self.episodes:
            s, L = ds.starts[ep], ds.lengths[ep]
            col = (0.18, 0.62, 0.31) if ds.success[ep] else (0.82, 0.27, 0.23)
            idx = list(range(0, L, stride)) + [L - 1]
            for i0, i1 in zip(idx, idx[1:]):
                g = add(mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), (*col, 0.6))
                if g is None:
                    break
                width = 4 if (show_attached and ds.grip[s + i0] > 0.5) else 1.5
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width,
                                     ds.pos[s + i0].astype(float), ds.pos[s + i1].astype(float))
            if ep == focus or not ds.has_panel:
                continue
            tp = self._target_pose(ep)
            if tp is None:
                continue
            r = ds.row(ep, k)
            pp, pq = pose_mul(tp[0], tp[1], ds.ppos[r], rotvec_to_quat(ds.prot[r]))
            if show_error:
                h = error_colour(ds.panel_error(r)[0]).lstrip("#")
                rgb = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
            else:
                rgb = col
            add(mujoco.mjtGeom.mjGEOM_BOX, PANEL_HALF, pp, quat_to_mat(pq).flatten(), (*rgb, 0.25))

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
        self.env.close()


class App:
    def __init__(self, root: tk.Tk, ds: Dataset, episodes: list[int], camera: str, overlays: set,
                 tile: int, open_3d: bool):
        self.root, self.ds, self.episodes = root, ds, episodes
        self.k = 0
        self.max_k = int(max(ds.lengths[e] for e in episodes)) - 1
        self.playing = False
        self.replay = None
        self.tile = tile
        self._photos = {}
        env = DrywallPlacementEnv()  # only for drawing the wall / rack outlines
        m = env.model
        mujoco.mj_forward(m, env.data)
        self.wall_x = float(env.data.geom_xpos[m.geom("wall").id][0] - m.geom_size[m.geom("wall").id][0])
        rg = m.geom("rack").id
        self.rack = (env.data.geom_xpos[rg].copy(), m.geom_size[rg].copy())
        env.close()

        root.title(f"dataset viewer: {len(episodes)} episodes")
        self.cam_var = tk.StringVar(value=camera)
        self.att_var = tk.BooleanVar(value="attached" in overlays)
        self.err_var = tk.BooleanVar(value="error" in overlays)
        self.speed = tk.DoubleVar(value=1.0)
        self.focus_var = tk.IntVar(value=episodes[0])
        self._build_ui()
        self._draw_static()
        root.protocol("WM_DELETE_WINDOW", self.close)
        if open_3d:
            self.open_3d()
        self.refresh()
        self._tick()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        bar = tk.Frame(r)
        bar.pack(fill="x", padx=6, pady=4)
        self.play_btn = tk.Button(bar, text="Play", width=6, command=self.toggle_play)
        self.play_btn.pack(side="left")
        tk.Label(bar, text="speed").pack(side="left", padx=(8, 0))
        tk.Scale(bar, variable=self.speed, from_=0.25, to=4.0, resolution=0.25, orient="horizontal",
                 length=110, showvalue=True).pack(side="left")
        tk.Label(bar, text="camera").pack(side="left", padx=(10, 2))
        for val, label in (("none", "off"), ("wrist", "wrist"), ("external", "external")):
            state = "normal" if val == "none" or self._cam_key(val) in self.ds.cams else "disabled"
            tk.Radiobutton(bar, text=label, value=val, variable=self.cam_var, state=state,
                           command=self.on_camera).pack(side="left")
        tk.Checkbutton(bar, text="attached", variable=self.att_var, command=self.on_overlay).pack(side="left", padx=(10, 0))
        err_state = "normal" if self.ds.has_panel else "disabled"
        tk.Checkbutton(bar, text="panel error", variable=self.err_var, state=err_state,
                       command=self.on_overlay).pack(side="left")
        tk.Button(bar, text="3D replay", command=self.open_3d).pack(side="left", padx=(10, 2))
        tk.Label(bar, text="focus ep").pack(side="left")
        tk.Spinbox(bar, values=self.episodes, textvariable=self.focus_var, width=5,
                   command=self.refresh).pack(side="left")

        self.time_var = tk.IntVar(value=0)
        tk.Scale(r, variable=self.time_var, from_=0, to=self.max_k, orient="horizontal",
                 label="time step (10 Hz)", command=self.on_slider).pack(fill="x", padx=6)
        self.status = tk.StringVar()
        tk.Label(r, textvariable=self.status, anchor="w", font=("Courier", 10)).pack(fill="x", padx=6)

        body = tk.Frame(r)
        body.pack(fill="both", expand=True)
        self.plot = tk.Canvas(body, width=PLOT_W, height=PANEL_H, bg="white", highlightthickness=0)
        self.plot.pack(side="left", padx=6, pady=6)
        n = len(self.episodes)
        self.cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / self.cols))
        self.grid = tk.Canvas(body, width=self.cols * self.tile, height=rows * self.tile, bg="black",
                              highlightthickness=0)
        self.on_camera(initial=True)

    @staticmethod
    def _cam_key(cam: str) -> str:
        return {"wrist": "camera0_rgb", "external": "external_rgb"}.get(cam, "")

    # ------------------------------------------------------------------ trajectory plots
    def _draw_static(self):
        ds, c = self.ds, self.plot
        allp = np.concatenate([ds.pos[ds.starts[e]:ds.starts[e] + ds.lengths[e]] for e in self.episodes])
        rp, rs = self.rack
        lo = np.minimum.reduce([allp.min(0), rp - rs, [self.wall_x - 0.1, -0.8, 0.0]]) - 0.05
        hi = np.maximum.reduce([allp.max(0), rp + rs, [self.wall_x + 0.05, 0.3, 1.0]]) + 0.05
        self.views = []  # (axis_u, axis_v, x_offset, scale, origin); two tall panels side by side
        for vi, (title, au, av) in enumerate((("top view (x right, y up)", 0, 1),
                                              ("side view (x right, z up)", 0, 2))):
            x0 = vi * PANEL_W
            s = min((PANEL_W - 30) / (hi[au] - lo[au]), (PANEL_H - 50) / (hi[av] - lo[av]))
            self.views.append((au, av, x0, s, lo))
            c.create_text(x0 + 8, 8, text=title, anchor="nw", fill="#444")
            if vi:
                c.create_line(x0, 0, x0, PANEL_H, fill="#ddd")
            # wall face and rack outline
            x = self._xy(vi, np.array([self.wall_x, -9, -9]))[0]
            c.create_line(x, 26, x, PANEL_H - 22, fill="#b8a77a", width=3)
            rp, rs = self.rack
            p0, p1 = self._xy(vi, rp - rs), self._xy(vi, rp + rs)
            c.create_rectangle(*p0, *p1, outline="#8b5a2b")
        self.path_items = {}
        self._draw_paths()
        self.markers = {}
        for e in self.episodes:
            self.markers[e] = [c.create_oval(0, 0, 0, 0, outline="black", width=1) for _ in self.views]
        c.create_text(8, PANEL_H - 4, anchor="sw", fill="#444", tags="legend", width=PLOT_W - 16,
                      text="lines: green = success, red = failure")

    def _xy(self, vi, p):
        au, av, x0, s, lo = self.views[vi]
        return x0 + 15 + (p[au] - lo[au]) * s, PANEL_H - 30 - (p[av] - lo[av]) * s

    def _draw_paths(self):
        c, ds = self.plot, self.ds
        c.delete("path")
        att = self.att_var.get()
        for e in self.episodes:
            s, L = ds.starts[e], ds.lengths[e]
            col = GREEN if ds.success[e] else RED
            for vi in range(len(self.views)):
                pts = [self._xy(vi, p) for p in ds.pos[s:s + L]]
                c.create_line(*np.ravel(pts), fill=col, width=1, tags="path")
                if att:  # thick overlay on the attached runs
                    g = ds.grip[s:s + L] > 0.5
                    edges = np.flatnonzero(np.diff(np.concatenate([[0], g.astype(int), [0]])))
                    for a, b in zip(edges[::2], edges[1::2]):
                        if b - a >= 2:
                            c.create_line(*np.ravel(pts[a:b]), fill=col, width=3, tags="path")
        c.tag_lower("path")
        c.itemconfigure("legend", text="lines: green = success, red = failure"
                        + ("; thick = suction attached" if att else "")
                        + ("; marker colour = panel-target error (blue 0, yellow 50, red >= 100 mm)"
                           if self.err_var.get() else ""))

    # ------------------------------------------------------------------ camera grid
    def on_camera(self, initial=False):
        if self.cam_var.get() == "none":
            self.grid.pack_forget()
        else:
            self.grid.pack(side="left", padx=6, pady=6, anchor="n")
        if not initial:
            self.refresh()

    def _draw_tiles(self):
        g, ds, t = self.grid, self.ds, self.tile
        g.delete("all")
        cam = self._cam_key(self.cam_var.get())
        for i, e in enumerate(self.episodes):
            r = ds.row(e, self.k)
            x, y = (i % self.cols) * t, (i // self.cols) * t
            img = Image.fromarray(ds.image(cam, r)).resize((t, t), Image.BILINEAR)
            self._photos[e] = ImageTk.PhotoImage(img)
            g.create_image(x, y, image=self._photos[e], anchor="nw")
            g.create_rectangle(x, y, x + t - 1, y + t - 1, outline=GREEN if ds.success[e] else RED)
            g.create_text(x + 3, y + 2, text=f"ep {e}", anchor="nw", fill="white", font=("Arial", 8, "bold"))
            if self.att_var.get() and ds.grip[r] > 0.5:
                g.create_rectangle(x + t - 52, y + 2, x + t - 3, y + 14, fill="#1f5fbf", outline="")
                g.create_text(x + t - 27, y + 8, text="SUCTION", fill="white", font=("Arial", 7, "bold"))
            if self.err_var.get() and ds.has_panel:
                pe, oe = ds.panel_error(r)
                g.create_rectangle(x + 2, y + t - 15, x + t - 3, y + t - 2, fill="black", outline="")
                g.create_text(x + 4, y + t - 9, anchor="w", fill=error_colour(pe), font=("Courier", 8, "bold"),
                              text=f"{pe * 1000:.0f}mm {np.degrees(oe):.1f}°")

    # ------------------------------------------------------------------ update loop
    def on_overlay(self):
        self._draw_paths()
        self.refresh()

    def on_slider(self, _=None):
        k = int(self.time_var.get())
        if k != self.k:  # programmatic set during playback already refreshed
            self.k = k
            self.refresh()

    def toggle_play(self):
        if not self.playing and self.k >= self.max_k:
            self.k = 0
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")

    def open_3d(self):
        if self.replay is None or not self.replay.running():
            self.replay = Replay3D(self.ds, self.episodes)
            self.refresh()

    def refresh(self):
        ds, c, k = self.ds, self.plot, self.k
        err_on = self.err_var.get() and ds.has_panel
        for e in self.episodes:
            r = ds.row(e, k)
            fill = error_colour(ds.panel_error(r)[0]) if err_on else (GREEN if ds.success[e] else RED)
            for vi, item in enumerate(self.markers[e]):
                x, y = self._xy(vi, ds.pos[r])
                c.coords(item, x - 4, y - 4, x + 4, y + 4)
                c.itemconfigure(item, fill=fill)
        if self.cam_var.get() != "none":
            self._draw_tiles()
        try:
            focus = int(self.focus_var.get())
        except (tk.TclError, ValueError):
            focus = self.episodes[0]
        if focus not in self.episodes:
            focus = self.episodes[0]
        if self.replay is not None:
            if self.replay.running():
                self.replay.update(k, focus, self.att_var.get(), err_on)
            else:
                self.replay.close()
                self.replay = None
        n_done = sum(k >= ds.lengths[e] - 1 for e in self.episodes)
        n_att = sum(ds.grip[ds.row(e, k)] > 0.5 for e in self.episodes)
        self.status.set(f"t = {k / ROW_HZ:5.1f} s   step {k}/{self.max_k}   attached {n_att}/{len(self.episodes)}"
                        f"   ended {n_done}/{len(self.episodes)}   success {int(ds.success[self.episodes].sum())}"
                        f"/{len(self.episodes)}")

    def _tick(self):
        t0 = time.perf_counter()
        if self.playing:
            if self.k < self.max_k:
                self.k += 1
                self.time_var.set(self.k)
                self.refresh()
            else:
                self.toggle_play()
        elif self.replay is not None and not self.replay.running():
            self.refresh()  # notices the 3D window closing
        spent = (time.perf_counter() - t0) * 1000
        delay = 1000 / (ROW_HZ * self.speed.get()) if self.playing else 100
        self._after = self.root.after(max(1, int(delay - spent)), self._tick)

    def close(self):
        if self.replay is not None:
            self.replay.close()
        self.ds.store.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="data/umi_sim.zarr.zip")
    ap.add_argument("--meta", default=None, help="sidecar _sim_meta.npz (default: next to the zarr)")
    ap.add_argument("--episodes", default="0-49", help="e.g. 0-49 or 0,3,10-19")
    ap.add_argument("--camera", choices=["none", "wrist", "external"], default="none")
    ap.add_argument("--overlay", default="", help="comma list: attached,error")
    ap.add_argument("--tile", type=int, default=0, help="camera tile size in px (default: fit about 700 px)")
    ap.add_argument("--3d", dest="open_3d", action="store_true", help="open the MuJoCo 3D replay at start")
    args = ap.parse_args()

    ds = Dataset(args.zarr, args.meta)
    eps = parse_episodes(args.episodes, ds.n)
    cols = int(np.ceil(np.sqrt(len(eps))))
    tile = args.tile or int(np.clip(700 // cols, 64, 224))
    overlays = {o.strip() for o in args.overlay.split(",") if o.strip()}
    root = tk.Tk()
    App(root, ds, eps, args.camera, overlays, tile, args.open_3d)
    root.mainloop()


if __name__ == "__main__":
    main()
