"""The stage viewport: views on a world-anchored pixel lattice, the static layer cached with overscan (so a pan
is a crop) and a region of 3 × 3 views whose field samples and streamline layout outlive pans.

Pixel column X of the lattice covers world x in [X, X + 1) / scale, row Y covers y in (−Y − 1, −Y] / scale.
A snapped view starts at a whole lattice pixel, so two views at one zoom differ by whole pixels, and anything
drawn from world-anchored data (the region's grid, the arrow lattice) is the same in both where they overlap.
"""
import math

import field as fieldmod
import render

OVERSCAN = 0.25   # of the view size, on every side of the cached static layer
EDGE = 0.5        # a new cache is rendered once less than this part of the overscan is left on a side
REGION = 2.0      # the region is this many times the cache in each direction (3 × 3 stage views)


def snap(view):
    """The view moved by less than a pixel so that it starts at a whole lattice pixel."""
    x0, y0 = view.origin()
    cx = (round(x0) + view.w / 2) / view.scale
    cy = -(round(y0) + view.h / 2) / view.scale
    return fieldmod.View.at(cx, cy, view.scale, view.w, view.h)


def grown(view, f):
    """A snapped view with the same centre and zoom and round(f · size) more pixels on every side."""
    mx, my = round(f * view.w), round(f * view.h)
    return snap(fieldmod.View.at(view.cx, view.cy, view.scale, view.w + 2 * mx, view.h + 2 * my))


def offset(inner, outer):
    """(dx, dy): where `inner` starts inside `outer` in whole pixels, or None if they differ in zoom or
    `inner` does not lie inside `outer`."""
    if inner.scale != outer.scale:
        return None
    (ix, iy), (ox, oy) = inner.origin(), outer.origin()
    dx, dy = round(ix - ox), round(iy - oy)
    if 0 <= dx <= outer.w - inner.w and 0 <= dy <= outer.h - inner.h:
        return dx, dy
    return None


def margins(inner, outer):
    """The smallest gap in pixels between `inner` and the edge of `outer` (negative: it sticks out),
    or None at another zoom."""
    if inner.scale != outer.scale:
        return None
    (ix, iy), (ox, oy) = inner.origin(), outer.origin()
    return min(ix - ox, iy - oy, ox + outer.w - ix - inner.w, oy + outer.h - iy - inner.h)


def cache_for(view, overscan=OVERSCAN):
    return grown(view, overscan)


def needs_render(cache, view):
    """True if the cached layer `cache` (a view) is at another zoom or the view is near its edge."""
    m = margins(view, cache)
    return m is None or m < EDGE * OVERSCAN * min(view.w, view.h)


class Region:
    """F sampled over a region of REGION × the cache around the stage, and what is derived from it once per
    field and zoom rather than per render: the streamline layout (in grid units, so in the world), and the
    heatmap images with their colour scales. Used on the render thread only. Steady fields only."""

    def __init__(self):
        self._key = None
        self.view = None
        self.grid = None
        self._lines = (None, None)
        self._heat = {}
        self.stamp = 0      # changes whenever what the region draws changes (for crossfades)
        self.layout = None  # (grid, streamline layout) of the latest layout, for exports (read from any thread)

    def prepare(self, field, cache, style):
        """Covers the cache view `cache` for this field, zoom and sampling; resamples F if it has to."""
        key = (field, cache.scale, style.px(style.sample))
        m = margins(cache, self.view) if self.view is not None else None
        if key == self._key and m is not None and m >= 0:
            return self
        self._key = key
        self.view = grown(cache, (REGION - 1) / 2)
        self.grid = fieldmod.FieldGrid(field, self.view, 0.0, spacing=style.px(style.sample), overscan=0.0)
        self._lines = (None, None)
        self._heat = {}
        self.layout = None
        self.stamp += 1
        return self

    def lines(self, vref, style):
        """The evenly spaced streamline layout of the whole region (kernel output, grid units)."""
        key = (vref, style.px(style.stream_sep), style.px(style.step))
        if self._lines[0] != key:
            self._lines = (key, fieldmod.layout_streamlines(self.grid, vref, *key[1:]))
            self.layout = (self.grid, self._lines[1])
            self.stamp += 1
        return self._lines[1]

    def heat(self, kind, vref, palette, view):
        """The heatmap of the whole region as an image of the grid (see render.grid_image)."""
        key = (kind, vref, palette)
        if key not in self._heat:
            self._heat = {key: render.grid_image(self.grid, kind, vref, palette, view)}
        return self._heat[key]
