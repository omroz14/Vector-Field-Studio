"""Resolution and frame-rate independence: an export at k× lays out the same lines, and 30 and 60 fps show
the same motion."""
import unittest

import numpy as np

from tests.helpers import preset
import field as F
import kernels
import particles
import render


class Resolution(unittest.TestCase):
    def test_same_streamlines_at_any_scale(self):
        for key in ("vanderpol", "dipole", "cells"):
            field, _ = preset(key)
            layouts = []
            for k in (1.0, 1.8, 2.25):
                v = F.View(0.0, 0.0, 3.0, round(480 * k), round(600 * k))
                s = render.Style(px_per_pt=k)
                g = F.FieldGrid(field, v, 0.0, spacing=s.px(s.sample))
                x, y, m, starts = F.layout_streamlines(g, F.reference_speed(field, v, 0.0),
                                                       s.px(s.stream_sep), s.px(s.step))
                layouts.append((g.x0 + x * g.h, g.y0 + y * g.h, starts))
            for x, y, starts in layouts[1:]:
                self.assertTrue(np.array_equal(starts, layouts[0][2]), f"{key}: the lines differ")
                self.assertLess(np.abs(np.r_[x - layouts[0][0], y - layouts[0][1]]).max(), 1e-9, key)


class FrameRate(unittest.TestCase):
    def test_same_motion_at_30_and_60_fps(self):
        field, view = preset("vortex", 600, 500)
        style = render.Style()
        vref = F.reference_speed(field, view, 0.0)
        grid = F.FieldGrid(field, view, 0.0)
        ends, times = [], []
        for fps in (30, 60):
            x, y, t = np.array([1.0]), np.array([0.0]), 0.0
            for _ in range(fps):   # one second
                dt = particles.frame_dt(field, view, style, vref, fps, 0.5)
                kernels.advect(grid.U, grid.V, grid.x0, grid.y0, grid.h, x, y, dt, np.empty(1), np.empty(1, bool))
                t += dt
            ends.append((x[0], y[0]))
            times.append(t)
        self.assertAlmostEqual(times[0], times[1], places=12)
        self.assertAlmostEqual(ends[0][0], ends[1][0], places=6)
        self.assertAlmostEqual(ends[0][1], ends[1][1], places=6)

    def test_trail_fade_per_second(self):
        for fps in (30, 60):
            tau = 0.65
            delta = np.exp(-1 / (tau * fps))
            self.assertAlmostEqual(delta ** fps, np.exp(-1 / tau), places=12)


class Blend(unittest.TestCase):
    def test_screen(self):
        a = np.array([[[200, 100, 0]]], np.uint8)
        b = np.array([[[100, 100, 255]]], np.uint8)
        want = np.round(255 * (1 - (1 - a / 255) * (1 - b / 255)))
        self.assertTrue(np.array_equal(render.screen(a, b), want.astype(np.uint8)))


if __name__ == "__main__":
    unittest.main()
