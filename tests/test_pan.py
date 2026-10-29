"""The stage's cached static layers: one rendered after a pan agrees pixel for pixel with the one before,
wherever both show the same place, so swapping them never shows."""
import unittest

import numpy as np

from tests.helpers import moved, overlap, preset, static
import render
import viewport
import field as F


class PanIdentity(unittest.TestCase):
    SHIFTS = ((37, -21), (-150, 90), (3, 0))

    def check(self, key, style, **kw):
        field, stage = preset(key, 640, 480)
        vref = F.reference_speed(field, stage, 0.0)
        region = viewport.Region()
        c0 = viewport.cache_for(stage)
        a = static(field, c0, style, region, vref=vref, **kw)
        stamp = region.stamp
        for dx, dy in self.SHIFTS:
            c1 = viewport.cache_for(moved(stage, dx, dy))
            b = static(field, c1, style, region, vref=vref, **kw)
            self.assertEqual(region.stamp, stamp, "the region should not have been rebuilt")
            A, B = overlap(a, c0, b, c1)
            bad = int((np.abs(A - B).max(axis=2) > 0).sum())
            self.assertEqual(bad, 0, f"{key} {style.background}: {bad} pixels differ after a pan of ({dx}, {dy})")

    def test_arrows(self):
        self.check("dipole", render.Style(), arrows=True)

    def test_coloured_streamlines(self):
        self.check("vanderpol", render.Style(), streamlines=True)

    def test_streamlines_on_speed(self):
        self.check("vanderpol", render.Style(background="magnitude"), streamlines=True)

    def test_streamlines_curl_and_trace(self):
        self.check("cells", render.Style(background="curl"), streamlines=True, pins=((0.3, 0.4),))

    def test_arrows_on_divergence(self):
        self.check("spiral", render.Style(background="divergence"), arrows=True)


class Viewport(unittest.TestCase):
    def test_snapped_views_differ_by_whole_pixels(self):
        _, v = preset("vortex")
        w = moved(v, 13, -7)
        self.assertEqual((w.ox - v.ox, w.oy - v.oy), (13, -7))
        self.assertEqual(viewport.offset(v, viewport.cache_for(v)), (160, 120))

    def test_render_only_near_the_edge(self):
        _, v = preset("vortex")
        cache = viewport.cache_for(v)
        self.assertFalse(viewport.needs_render(cache, moved(v, 40, 0)))
        self.assertTrue(viewport.needs_render(cache, moved(v, 150, 0)))
        self.assertTrue(viewport.needs_render(cache, F.View.at(v.cx, v.cy, v.scale * 1.1, v.w, v.h)))


if __name__ == "__main__":
    unittest.main()
