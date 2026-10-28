"""Particles and the trails they paint, independent of Tk: the stage runs one Flow, an exporter its own.

Time is in seconds. At `fps` frames per second a trail fades by exp(−1/(τ·fps)) per frame, particles live a
few seconds and the field time advances by frame_dt(), so 30 and 60 fps show the same motion.
"""
import math

import numpy as np

import kernels
import render

LIFE = (1.3, 4.7)   # seconds a particle lives before it respawns somewhere else


def frame_dt(field, view, style, vref, fps, t_rate):
    """Field time per frame. A field that uses t runs at t_rate units of t per second whatever the zoom;
    in a steady one, particles at the typical speed vref cover style.particle_speed points per second."""
    if field.uses_t:
        return t_rate / fps
    return style.particle_speed * style.px_per_pt / (fps * vref * view.scale)


class Particles:
    """Massless tracers carried by the field. Positions are in world coordinates, ages in seconds."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = None

    def _spawn(self, n, view):
        px = self.rng.uniform(0, view.w, n)
        py = self.rng.uniform(0, view.h, n)
        return np.stack(view.to_world(px, py), axis=1)

    def reset(self, n, view):
        self.pos = self._spawn(n, view)
        self.life = self.rng.uniform(*LIFE, n)
        self.age = self.rng.uniform(0, self.life)

    def step(self, grid, view, dt, seconds):
        """Moves every particle by one RK4 step of field time dt through the grid, ages it by `seconds` and
        respawns the ones that left the view or grew old. Returns (previous positions, speeds, respawned)."""
        old = self.pos.copy()
        x, y = old[:, 0].copy(), old[:, 1].copy()
        speed = np.empty(len(x))
        alive = np.empty(len(x), bool)
        kernels.advect(grid.U, grid.V, grid.x0, grid.y0, grid.h, x, y, dt, speed, alive)
        self.pos = np.stack([x, y], axis=1)
        self.age += seconds
        px, py = view.to_px(x, y)
        dead = ((self.age > self.life) | ~alive | ~np.isfinite(px) | ~np.isfinite(py)
                | (px < 0) | (px >= view.w) | (py < 0) | (py >= view.h))
        n = int(dead.sum())
        if n:
            self.pos[dead] = self._spawn(n, view)
            self.age[dead] = 0
            self.life[dead] = self.rng.uniform(*LIFE, n)
        return old, speed, dead


class Flow:
    """n particles in `view` and the trail image they paint (float32, 0–255), at `fps` frames per second."""

    def __init__(self, n, view, fps, seed=None):
        self.view, self.fps = view, fps
        self.parts = Particles(seed)
        self.parts.reset(n, view)
        self.trail = np.zeros((view.h, view.w, 3), np.float32)

    def clear(self):
        self.trail[:] = 0

    def image(self):
        return self.trail.astype(np.uint8)

    @staticmethod
    def _colors(rel, style):
        return render.vector_colors(rel, style.colored and style.background == "none",
                                    style.palette).astype(np.float32)

    def _dot(self, px, py, col, style):
        """Paints square dots of the style's particle size centred on the pixel points (px, py)."""
        n = max(2, round(style.px(style.particle_size)))
        h, w = self.trail.shape[:2]
        x0 = np.floor(px - (n - 1) / 2).astype(int)
        y0 = np.floor(py - (n - 1) / 2).astype(int)
        ok = (x0 >= 0) & (x0 <= w - n) & (y0 >= 0) & (y0 <= h - n)
        x0, y0, col = x0[ok], y0[ok], col[ok]
        for dy in range(n):
            for dx in range(n):
                self.trail[y0 + dy, x0 + dx] = np.maximum(self.trail[y0 + dy, x0 + dx], col)

    def paint(self, field, t, vref, style):
        """Paints the particles where they are now (after a reset, so the first frame is not empty)."""
        x, y = self.parts.pos.T
        u, v = field(x, y, t)
        self._dot(*self.view.to_px(x, y), self._colors(np.hypot(u, v) / vref, style), style)

    def step(self, grid, dt, vref, style):
        """One frame: fades the trails, moves the particles by field time dt and paints their paths."""
        tau = style.trail
        self.trail *= math.exp(-1 / (tau * self.fps)) if tau > 0 else 0.0
        view = self.view
        old, speed, dead = self.parts.step(grid, view, dt, 1 / self.fps)
        keep = ~dead
        ox, oy = view.to_px(*old[keep].T)
        nx, ny = view.to_px(*self.parts.pos[keep].T)
        col = self._colors(speed[keep] / vref, style)
        # Fill the gap between the old and new position, so fast particles leave lines, not dots.
        dist = np.hypot(nx - ox, ny - oy)
        nsub = int(np.clip(np.ceil(dist.max(initial=0) / style.px(style.substep)), 1, 12))
        for k in range(1, nsub + 1):
            a = k / nsub
            self._dot(ox + a * (nx - ox), oy + a * (ny - oy), col, style)
