from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from annotation_pathing import astar, smooth_path, theta_star
from annotation_settings import DEFAULT_TURN_PENALTY
from common import clamp_xy_to_bounds, overlay_path_on_image


class SessionController:
    def __init__(
        self,
        image_rgba: np.ndarray,
        grid_binary: np.ndarray,
        move_cost: np.ndarray,
        dist_map: np.ndarray,
        trav01: Optional[np.ndarray],
        img_name: str,
        maps_dir: Path,
        eight_connected: bool,
        line_width: int,
        viz: bool,
        idx_label: str,
        use_theta_star: bool,
        min_clear_px: int,
        clear_margin_px: float,
        samples_per_px: float,
    ):
        self.image = image_rgba
        self.grid_bin = grid_binary
        self.move_cost = move_cost
        self.dist_map = dist_map
        self.trav01 = trav01
        self.img_name = img_name
        self.maps_dir = maps_dir
        self.eight_connected = eight_connected
        self.line_width = line_width
        self.viz = viz
        self.idx_label = idx_label
        self.use_theta_star = use_theta_star
        self.min_clear = int(min_clear_px)
        self.clear_margin = float(clear_margin_px)
        self.samples_per_px = float(samples_per_px)

        self.H, self.W = self.image.shape[:2]
        self.starts_xy: List[Tuple[int, int]] = []
        self.goal_xy: Optional[Tuple[int, int]] = None
        self.paths: List[List[Tuple[int, int]]] = []

        self.mode = "start"
        self.choice = None

        self.fig = None
        self.ax_img = None
        self.ax_bin = None
        self.ax_trav = None
        self.b_next = None
        self.b_skip = None

    def draw(self):
        self.fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        self.ax_img, self.ax_bin, self.ax_trav = axes
        self.fig.canvas.manager.set_window_title("A* Planner {} — {}".format(self.idx_label, self.img_name))
        self._redraw_all()

        ax_next = plt.axes([0.80, 0.01, 0.15, 0.06])
        ax_skip = plt.axes([0.62, 0.01, 0.15, 0.06])
        self.b_next = Button(ax_next, "Next [n]")
        self.b_skip = Button(ax_skip, "Skip [s]")
        self.b_next.on_clicked(lambda evt: self._on_choice("next"))
        self.b_skip.on_clicked(lambda evt: self._on_choice("skip"))

        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show(block=True)

    def _redraw_all(self):
        self.ax_img.clear()
        self.ax_img.imshow(self.image)
        self.ax_img.set_axis_off()
        title = "Image — Click START(s) on Image or Occupancy. Press 'g' then click GOAL."
        if self.mode == "goal":
            title = "Image — GOAL mode: click once to set GOAL."
        if not self.viz:
            title += " (viz=OFF)"
        self.ax_img.set_title(title)

        for sx, sy in self.starts_xy:
            self.ax_img.scatter([sx], [sy], c="g", s=25)
        if self.goal_xy is not None:
            gx, gy = self.goal_xy
            self.ax_img.scatter([gx], [gy], c="b", s=25)
        if self.viz and self.paths:
            overlay = self.image.copy()
            for path_rc in self.paths:
                overlay = overlay_path_on_image(overlay, path_rc, line_width=self.line_width)
            self.ax_img.imshow(overlay)

        self.ax_bin.clear()
        self.ax_bin.imshow(self.grid_bin, cmap="gray", vmin=0, vmax=1)
        self.ax_bin.set_title("Binary occupancy (1=free, 0=blocked)\n(clicks allowed here)")
        self.ax_bin.set_axis_off()
        for sx, sy in self.starts_xy:
            self.ax_bin.scatter([sx], [sy], c="g", s=25)
        if self.goal_xy is not None:
            gx, gy = self.goal_xy
            self.ax_bin.scatter([gx], [gy], c="b", s=25)
        if self.viz and self.paths:
            for path_rc in self.paths:
                if len(path_rc) >= 2:
                    rr, cc = zip(*path_rc)
                    self.ax_bin.plot(cc, rr, linewidth=self.line_width)

        self.ax_trav.clear()
        if self.trav01 is not None:
            self.ax_trav.imshow(self.trav01, cmap="viridis", vmin=0, vmax=1)
            self.ax_trav.set_title("Traversability (normalized view)")
        else:
            self.ax_trav.text(0.5, 0.5, "No traversability_map.npy", ha="center", va="center", fontsize=11)
            self.ax_trav.set_title("Traversability")
        self.ax_trav.set_axis_off()

        self.fig.canvas.draw_idle()

    def _onclick(self, event):
        if event.inaxes not in (self.ax_img, self.ax_bin):
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = clamp_xy_to_bounds(event.xdata, event.ydata, self.W, self.H)
        if self.mode == "start":
            self.starts_xy.append((x, y))
            self._redraw_all()
        else:
            self.goal_xy = (x, y)
            self._compute_paths()
            self._redraw_all()

    def _onkey(self, event):
        if event.key == "n":
            self._on_choice("next")
        elif event.key == "s":
            self._on_choice("skip")
        elif event.key == "g":
            self.mode = "goal"
            self._redraw_all()
        elif event.key == "c":
            self.starts_xy = []
            self.goal_xy = None
            self.paths = []
            self.mode = "start"
            self._redraw_all()

    def _on_choice(self, choice: str):
        self.choice = choice
        plt.close(self.fig)

    def _compute_paths(self):
        if not self.starts_xy or self.goal_xy is None:
            return

        gx, gy = self.goal_xy
        goal_rc = (gy, gx)
        paths = []
        for sx, sy in self.starts_xy:
            start_rc = (sy, sx)
            try:
                if self.use_theta_star:
                    p_raw = theta_star(self.grid_bin, start_rc, goal_rc, move_cost=self.move_cost)
                else:
                    p_raw = astar(
                        self.grid_bin,
                        start_rc,
                        goal_rc,
                        eight_connected=self.eight_connected,
                        move_cost=self.move_cost,
                        turn_penalty_scale=DEFAULT_TURN_PENALTY,
                    )
                p = smooth_path(
                    self.grid_bin,
                    p_raw,
                    dist_map=self.dist_map,
                    clear_margin_px=max(4, self.clear_margin),
                    samples_per_px=self.samples_per_px,
                )
            except Exception as exc:
                print("[planner] Failed for start {} -> goal {}: {}".format((sx, sy), (gx, gy), exc))
                p = []
            paths.append(p)
        self.paths = paths
