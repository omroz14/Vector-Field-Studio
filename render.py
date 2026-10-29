"""Drawing: color maps, the style, heatmaps, guides, arrows and streamlines into images of a view.

Two layers: the art (background, arrows, streamlines, traces and, on top, particles) is what an export shows;
the guides (grid and axes) help on the stage and stay out of exports. Numbers are drawn by Tk only.
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

import field as fieldmod
import kernels
from palettes import PALETTES, PALETTE_NAMES
from theme import *

GRID = "#1b1f2a"   # faint grid lines on the plain background


@dataclass(frozen=True)
class Style:
    """Every visual setting. Sizes are in points: one point is one stage pixel, and a render of the same
    region at k× the pixels uses px_per_pt = k, so an export looks like the preview, only sharper.
    Speeds and durations are per second, so the frame rate does not change the look either."""
    px_per_pt: float = 1.0
    palette: str = "studio"
    background: str = "none"     # none, magnitude, divergence or curl
    colored: bool = True         # vectors coloured by speed (on the plain background only)
    arrow_spacing: float = 36.0
    arrow_length: float = 1.0    # of the arrow cell
    equal_length: bool = False
    arrow_width: float = 1.5
    stream_sep: float = 22.0
    stream_heads: bool = True
    line_width: float = 1.3
    trace_width: float = 2.4
    particle_size: float = 2.0   # width of a particle's trail; never under 2 px, so it survives 4:2:0 video
    particle_gain: float = 2.5   # κ: trail brightness is 1 − exp(−κ · density)
    glow: bool = False           # a soft halo around the trails
    particle_speed: float = 90.0 # points per second at the typical speed |F| = vref (steady fields)
    trail: float = 0.65          # seconds: time constant of the trail fade (0 = no trail)
    substep: float = 1.2         # points between the dots that join a particle's old and new position
    sample: float = 2.0          # points between the field samples of the grid
    step: float = 1.5            # points per RK4 step of a streamline
    tick: float = 90.0           # points between grid lines, roughly

    def px(self, pt):
        return pt * self.px_per_pt


MAG_TOP = 200      # the magnitude heatmap stops short of the brightest color so vectors stay readable on it


def vector_colors(rel, colored, palette="studio"):
    """(N, 3) uint8 colors for relative speeds rel (1 = typical speed)."""
    if not colored:
        return np.tile(np.array(hex_to_rgb(TEXT), np.uint8), (len(rel), 1))
    return PALETTES[palette].line[(np.clip(rel, 0, 1) * 255).astype(int)]


def nice_step(raw):
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw * (1 - 1e-9))


def ticks(view, target_px=90.0):
    """(step, xs, ys): round world coordinates of the grid lines visible in the view."""
    step = nice_step(target_px / view.scale)
    x0, y1 = view.to_world(0, 0)
    x1, y0 = view.to_world(view.w, view.h)
    xs = np.arange(math.ceil(x0 / step), math.floor(x1 / step) + 1) * step
    ys = np.arange(math.ceil(y0 / step), math.floor(y1 / step) + 1) * step
    return step, xs, ys


def view_affine(old, new):
    """PIL AFFINE data that resamples an image of view `old` into view `new` (pan, zoom and resize)."""
    k = old.scale / new.scale
    return (k, 0, new.ox * k - old.ox, 0, k, new.oy * k - old.oy)


def _grid_to_view(img, grid, view, ss):
    """Resamples an image of the grid (row 0 = top row of the grid) onto the view at ss× (bilinear). For a
    snapped view at the grid's zoom every number here is exact, so neighbouring views resample identically."""
    ny = grid.U.shape[0]
    s = grid.unit(view)
    k = 1 / (ss * s)   # grid nodes per output pixel
    c = view.ox / s - grid.i0 + 0.5
    f = ny - 0.5 + view.oy / s + grid.j0
    return img.transform((view.w * ss, view.h * ss), Image.AFFINE, (k, 0, c, 0, k, f), Image.BILINEAR)


def heatmap(grid, view, kind, vref, palette="studio", ss=1, image=None):
    """Background image at ss× the view: plain, |F|, divergence or curl, read from the sampled grid
    (or from `image`, a grid_image() made earlier)."""
    if kind == "none":
        return Image.new("RGB", (view.w * ss, view.h * ss), SURFACE)
    if image is None:
        image = grid_image(grid, kind, vref, palette, view)
    return _grid_to_view(image, grid, view, ss)


def grid_image(grid, kind, vref, palette, view):
    """The heatmap at the grid's nodes, as an image with the top row first. `view` sets the floor of the
    div / curl colour scale."""
    U, V = grid.U, grid.V
    if kind == "magnitude":
        idx = (np.clip(np.hypot(U, V) / vref, 0, 1) * MAG_TOP).astype(np.uint8)
        img = PALETTES[palette].seq[idx]
    else:
        h = grid.h   # rows grow with y, so both derivatives use +h
        if kind == "divergence":
            q = np.gradient(U, h, axis=1) + np.gradient(V, h, axis=0)
        else:
            q = np.gradient(V, h, axis=1) - np.gradient(U, h, axis=0)
        # The floor keeps round-off noise of a divergence- or curl-free field at the neutral color.
        ref = max(float(np.percentile(np.abs(q[::4, ::4]), 95)), 1e-3 * vref * view.scale / view.h)
        idx = ((np.clip(q / ref, -1, 1) + 1) * 127.5).astype(np.uint8)
        img = PALETTES[palette].div[idx]
    return Image.fromarray(np.ascontiguousarray(img[::-1]))


def draw_guides(d, view, style, ss, grid):
    """Axes, and with `grid` the faint grid lines. Display only."""
    w, h = view.w * ss, view.h * ss
    lw = max(1, round(style.px(1) * ss))
    _, xs, ys = ticks(view, style.px(style.tick))
    if grid:
        for x in xs:
            px = view.to_px(x, 0)[0] * ss
            d.line([(px, 0), (px, h)], fill=GRID, width=lw)
        for y in ys:
            py = view.to_px(0, y)[1] * ss
            d.line([(0, py), (w, py)], fill=GRID, width=lw)
    ox, oy = view.to_px(0, 0)
    if 0 <= ox <= view.w:
        d.line([(ox * ss, 0), (ox * ss, h)], fill=BORDER, width=lw)
    if 0 <= oy <= view.h:
        d.line([(0, oy * ss), (w, oy * ss)], fill=BORDER, width=lw)


def _q(a):
    """Coordinates on a 1/64 px grid. Pillow rasterises in single precision, where a point's fraction rounds
    differently at different distances from the corner; on this grid it is exact, so a shape shifted by whole
    pixels is drawn identically (two cached layers of neighbouring views agree)."""
    return np.round(np.asarray(a, float) * 64) / 64


def _head(d, tip, direction, size, color):
    """A filled triangle with its point at `tip`, pointing along the unit vector `direction`."""
    ux, uy = direction
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    hw = size * 0.55
    pts = _q([tip, (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)])
    d.polygon([tuple(p) for p in pts], fill=color)


def draw_arrows(d, field, view, t, vref, style, colored, ss):
    """Arrows at the world lattice (i·s, j·s), s = one arrow spacing: they stay put when the view pans, and
    an export at k× shows the very arrows of the preview."""
    spacing, length, k = style.px(style.arrow_spacing), style.arrow_length, style.px_per_pt
    s = spacing / view.scale
    x0, y1 = view.to_world(-spacing, -spacing)
    x1, y0 = view.to_world(view.w + spacing, view.h + spacing)
    I, J = np.meshgrid(np.arange(math.ceil(x0 / s), math.floor(x1 / s) + 1),
                       np.arange(math.ceil(y0 / s), math.floor(y1 / s) + 1))
    X, Y = I * s, J * s
    U, V = field(X, Y, t)
    U, V = U.ravel(), V.ravel()
    m = np.hypot(U, V)
    cell = spacing
    rel = m / vref
    if style.equal_length:
        L = np.where(rel > 1e-6, 0.8 * cell * length, 0.0)
    else:
        L = 0.9 * cell * length * np.clip(rel, 0, 1.5)
    with np.errstate(all="ignore"):
        ux, uy = np.nan_to_num(U / m), np.nan_to_num(-V / m)
    px, py = I.ravel() * spacing - view.ox, -J.ravel() * spacing - view.oy   # lattice pixels, exactly
    colors = vector_colors(rel, colored, style.palette)
    width = max(1, round(style.px(style.arrow_width) * ss))
    for i in np.flatnonzero(L > 2 * k):
        hl = min(max(0.35 * L[i], 3.0 * k), 8.0 * k)
        dx, dy = ux[i], uy[i]
        tip = np.array([px[i] + 0.5 * L[i] * dx, py[i] + 0.5 * L[i] * dy])
        tail = (px[i] - 0.5 * L[i] * dx, py[i] - 0.5 * L[i] * dy)
        neck = tip - 0.8 * hl * np.array([dx, dy])
        c = tuple(int(v) for v in colors[i])
        d.line(_q([tail[0] * ss, tail[1] * ss, neck[0] * ss, neck[1] * ss]).tolist(), fill=c, width=width)
        _head(d, tip * ss, (dx, dy), hl * ss, c)


def draw_lines(d, lines, vref, colored, heads, ss, width, k=1.0, color=None, palette="studio"):
    """Streamlines from field.streamlines(), `width` px wide (px_per_pt k sizes the arrows).
    With `color`, every line gets that color."""
    w = max(1, round(width * ss))
    plain = color or hex_to_rgb(TEXT)
    size = (4 * k + width * 2) * ss
    for pts, speed, arrows, first in lines:
        if len(pts) < 2:
            continue
        p = _q(pts * ss)
        if colored and color is None:
            cols = vector_colors(speed / vref, True, palette)
            # Pieces of 8 points counted from the start of the whole line (field._lines_px cuts lines at piece
            # boundaries), so a line cut at another place by a neighbouring view gets the same colours.
            for s in range(-(first % 8), len(p) - 1, 8):
                s = max(s, 0)
                c = tuple(int(v) for v in cols[s:s + 9].mean(axis=0))
                d.line(p[s:s + 9].ravel().tolist(), fill=c, width=w, joint="curve")
        else:
            d.line(p.ravel().tolist(), fill=plain, width=w, joint="curve")
        if not heads:
            continue
        for i in arrows:
            a, b = p[max(i - 3, 0)], p[min(i + 3, len(p) - 1)]
            n = np.hypot(*(b - a))
            if n > 0:
                u = (b - a) / n
                c = tuple(int(v) for v in cols[i]) if colored and color is None else plain
                _head(d, p[i] + u * size / 2, u, size, c)


def _pad(style):
    """Pixels drawn around a layer and cut off afterwards: shapes that cross an image edge (negative
    coordinates) rasterise differently from the same shapes inside one, and downscaling needs neighbours."""
    return math.ceil(style.px(max(16.0, 1.4 * style.arrow_spacing * style.arrow_length + 8)))


def compose(field, grid, view, t, vref, style, arrows=False, lines=None, traces=(), ss=2, guides=True, heat=None):
    """The static layer: background, guides (if `guides`), streamlines, arrows (if `arrows`) and traced lines,
    drawn at `ss`× and scaled down for antialiasing. `heat` is a grid_image() of the background, if made.
    The grid must reach _pad(style) px beyond the view."""
    colored = style.colored and style.background == "none"   # on a heatmap, vectors are drawn in light color
    k = style.px_per_pt
    p = _pad(style)
    pv = fieldmod.View.at(view.cx, view.cy, view.scale, view.w + 2 * p, view.h + 2 * p)
    shift = lambda ls: [(pts + p, *rest) for pts, *rest in ls]   # view px -> padded px
    img = heatmap(grid, pv, style.background, vref, style.palette, ss, heat)
    d = ImageDraw.Draw(img)
    if guides:
        draw_guides(d, pv, style, ss, grid=style.background == "none")
    if lines:
        draw_lines(d, shift(lines), vref, colored, style.stream_heads, ss, style.px(style.line_width), k,
                   palette=style.palette)
    if arrows:
        draw_arrows(d, field, pv, t, vref, style, colored, ss)
    if traces:
        draw_lines(d, shift(traces), vref, False, True, ss, style.px(style.trace_width), k, color=(255, 255, 255))
    if ss > 1:
        img = img.resize((pv.w, pv.h), Image.LANCZOS)
    return np.asarray(img)[p:p + view.h, p:p + view.w]


def screen(a, b, gain=1.0):
    """Screen blend of two uint8 images, 1 − (1 − a)(1 − gain·b): light adds up without clipping to the brighter one."""
    out = np.empty_like(a)
    kernels.screen(np.ascontiguousarray(a), np.ascontiguousarray(b), float(gain), out)
    return out


def build_static(field, view, t, style, arrows=False, streamlines=False, grid_needed=False, pins=(), ss=2,
                 guides=True, vref=None, region=None, layout=None, overscan=0.25):
    """Samples the field and draws the static layer of `view`. Returns (image, grid or None, vref).
    `pins` are world points with a traced streamline; `grid_needed` asks for the grid even if nothing drawn
    here reads it (the particles do). vref defaults to the typical speed in the view.
    With a viewport.Region (steady fields on the stage), its grid, layout and heatmap are used, so layers
    built for neighbouring views agree pixel for pixel. `layout` = (grid, lines) reuses a streamline layout
    made elsewhere (an export showing the stage's lines); `overscan` is the grid's, if one is sampled here."""
    if vref is None:
        vref = fieldmod.reference_speed(field, view, t)
    step, sep = style.px(style.step), style.px(style.stream_sep)
    grid = heat = lines = None
    if region is not None:
        grid = region.grid
        if streamlines:
            lines = fieldmod.lines_in(grid, view, region.lines(vref, style), step, sep)
        if style.background != "none":
            heat = region.heat(style.background, vref, style.palette, view)
    elif grid_needed or streamlines or style.background != "none" or len(pins):
        overscan = max(overscan, (_pad(style) + 4) / min(view.w, view.h))   # compose draws a margin too
        grid = fieldmod.FieldGrid(field, view, t, spacing=style.px(style.sample), overscan=overscan)
    if streamlines and lines is None:
        if layout is not None:
            lines = fieldmod.lines_in(layout[0], view, layout[1], step, sep)
        else:
            lines = fieldmod.streamlines(grid, view, vref, sep, step)
    traces = ()
    if len(pins):
        traces = fieldmod.trace_through(grid, view, vref, pins, style.px(1.0))
    img = compose(field, grid, view, t, vref, style, arrows, lines, traces, ss=ss, guides=guides, heat=heat)
    return img, grid, vref
