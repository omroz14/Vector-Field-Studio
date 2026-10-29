"""Seamless loops: the frame after the last one equals the first (within one 8-bit step)."""
import unittest

import numpy as np

from tests.helpers import mean_step, preset
import export
import render
import field as F


def frames(key, style=render.Style(), fps=30, seconds=2, mode="particles", period="auto", fx=None, fy=None):
    if key:
        field, stage = preset(key, 600, 460)
    else:
        field, stage = F.Field(fx, fy), preset("vortex", 600, 460)[1]
    scene = export.Scene(field, style, t=0.3, t_rate=0.5, streamlines=mode == "streamlines",
                         density=25 if mode == "particles" else 0)
    settings = export.Settings(360, 450, fps, seconds, "mp4", 1, False, True,
                               field.period if period == "auto" else period)
    r = export.Renderer(scene, stage, settings, seed=1)
    return r, [f for f in r.frames(extra=1) if f is not None]


class ExactLoops(unittest.TestCase):
    def check(self, key, **kw):
        r, out = frames(key, **kw)
        self.assertEqual(r.kind, "exact")
        n = r.settings.frames
        self.assertEqual(len(out), n + 1)
        diff = np.abs(out[n].astype(int) - out[0]).max()
        self.assertLessEqual(diff, 1, f"{key}: frame {n} differs from frame 0 by {diff}/255")

    def test_steady_particles(self):
        self.check("vortex")

    def test_steady_particles_60fps(self):
        self.check("cells", fps=60, seconds=1)

    def test_long_trails_into_a_sink(self):
        self.check("sink", style=render.Style(trail=2.0))

    def test_glow(self):
        self.check("vanderpol", style=render.Style(glow=True))

    def test_heatmap_without_trails(self):
        self.check("dipole", style=render.Style(background="magnitude", trail=0.0))

    def test_periodic_field_particles(self):
        self.check("gyre", seconds=2)

    def test_periodic_field_streamlines(self):
        self.check("gyre", mode="streamlines", style=render.Style(background="magnitude"), seconds=1)


class Crossfade(unittest.TestCase):
    def check(self, **kw):
        r, out = frames(**kw)
        self.assertEqual(r.kind, "crossfade")
        self.assertEqual(len(out), r.settings.frames)
        steps = [mean_step(out[i + 1], out[i]) for i in range(len(out) - 1)]
        seam = mean_step(out[0], out[-1])
        self.assertLessEqual(seam, 1.25 * np.median(steps) + 0.5,
                             f"the seam ({seam:.2f}) jumps more than a frame step ({np.median(steps):.2f})")

    def test_non_periodic_field(self):
        self.check(key=None, fx="-y + 0.3*sin(t)*sin(sqrt(2)*t)", fy="x")

    def test_period_cleared(self):
        self.check(key="gyre", period=None)


if __name__ == "__main__":
    unittest.main()
