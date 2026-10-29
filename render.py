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


def _ramp(stops, pos):
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    rgb = [hex_to_rgb(s) for s in stops]
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [s[c] for s in rgb])
    return lut


class Palette:
    """A color scheme: `seq` for the magnitude (0 -> fast), `div` for signed quantities (negative -> positive),
    and `line` for vectors on the plain background: the bright part of `seq`, so slow vectors stay visible."""

    def __init__(self, seq, div, seq_pos=None):
        self.seq = _ramp(seq, seq_pos or np.linspace(0, 1, len(seq)))
        self.div = _ramp(div, np.linspace(0, 1, len(div)))
        self.line = self.seq[np.linspace(90, 255, 256).astype(int)]


# Sequential stops run from slow to fast; diverging stops have the stage color in the middle, at zero.
PALETTES = {
    "studio": Palette([SURFACE, ACCENT, WARM, "#ffffff"],     # the motion map palette of Pixel Sort Studio
                      ["#d6ccff", ACCENT, SURFACE, WARM, "#ffe2b8"], [0.0, 0.4, 0.75, 1.0]),
    "viridis": Palette(["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
                       ["#b4c8ff", "#3b6fd8", SURFACE, "#d8503b", "#ffc4a8"]),
    "magma": Palette(["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
                     ["#a8f0e0", "#2a9d8f", SURFACE, "#de4968", "#fcc5c0"]),
    "ice": Palette([SURFACE, "#16425b", "#2a87b0", "#7fd3e6", "#f2fbff"],
                   ["#7fd3e6", "#2a87b0", SURFACE, "#c98a2e", "#ffe2b8"]),
    "fire": Palette([SURFACE, "#5c1a1a", "#c0392b", "#f39c12", "#fff3b0"],
                    ["#7fd3e6", "#2a87b0", SURFACE, "#c0392b", "#f39c12"]),
    "turbo": Palette(["#30123b", "#4669db", "#26bce1", "#5ffc73", "#d0e935", "#fb8022", "#7a0403"],
                     ["#26bce1", "#4669db", SURFACE, "#fb8022", "#d0e935"]),
}
PALETTE_NAMES = (("studio", "Studio"), ("viridis", "Viridis"), ("magma", "Magma"),
                 ("ice", "Ice"), ("fire", "Fire"), ("turbo", "Turbo"))
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
    c = (new.cx - new.w / (2 * new.scale) - old.cx) * old.scale + old.w / 2
    f = old.h / 2 - (new.cy - old.cy) * old.scale - new.h / 2 * k
    return (k, 0, c, 0, k, f)


def _grid_to_view(img, grid, view, ss):
    """Resamples an image of the grid (row 0 = top row of the grid) onto the view at ss× (bilinear)."""
    ny = grid.U.shape[0]
    k = 1 / (ss * view.scale * grid.h)   # grid nodes per output pixel
    c = (view.cx - view.w / (2 * view.scale) - grid.x0) / grid.h + 0.5
    f = ny - 0.5 - (view.cy + view.h / (2 * view.scale) - grid.y0) / grid.h
    return img.transform((view.w * ss, view.h * ss), Image.AFFINE, (k, 0, c, 0, k, f), Image.BILINEAR)


def heatmap(grid, view, kind, vref, palette="studio", ss=1):
    """Background image at ss× the view: plain, |F|, divergence or curl, read from the sampled grid."""
    if kind == "none":
        return Image.new("RGB", (view.w * ss, view.h * ss), SURFACE)
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
    return _grid_to_view(Image.fromarray(img[::-1]), grid, view, ss)


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


def _head(d, tip, direction, size, color):
    """A filled triangle with its point at `tip`, pointing along the unit vector `direction`."""
    ux, uy = direction
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    hw = size * 0.55
    d.polygon([tuple(tip), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], fill=color)


def draw_arrows(d, field, view, t, vref, style, colored, ss):
    spacing, length, k = style.px(style.arrow_spacing), style.arrow_length, style.px_per_pt
    nx, ny = max(1, round(view.w / spacing)), max(1, round(view.h / spacing))
    X, Y = view.grid(nx, ny)
    U, V = field(X, Y, t)
    U, V = U.ravel(), V.ravel()
    m = np.hypot(U, V)
    cell = min(view.w / nx, view.h / ny)
    rel = m / vref
    if style.equal_length:
        L = np.where(rel > 1e-6, 0.8 * cell * length, 0.0)
    else:
        L = 0.9 * cell * length * np.clip(rel, 0, 1.5)
    with np.errstate(all="ignore"):
        ux, uy = np.nan_to_num(U / m), np.nan_to_num(-V / m)
    px, py = view.to_px(X.ravel(), Y.ravel())
    colors = vector_colors(rel, colored, style.palette)
    width = max(1, round(style.px(style.arrow_width) * ss))
    for i in np.flatnonzero(L > 2 * k):
        hl = min(max(0.35 * L[i], 3.0 * k), 8.0 * k)
        dx, dy = ux[i], uy[i]
        tip = np.array([px[i] + 0.5 * L[i] * dx, py[i] + 0.5 * L[i] * dy])
        tail = (px[i] - 0.5 * L[i] * dx, py[i] - 0.5 * L[i] * dy)
        neck = tip - 0.8 * hl * np.array([dx, dy])
        c = tuple(int(v) for v in colors[i])
        d.line([(tail[0] * ss, tail[1] * ss), tuple(neck * ss)], fill=c, width=width)
        _head(d, tip * ss, (dx, dy), hl * ss, c)


def draw_lines(d, lines, vref, colored, heads, ss, width, k=1.0, color=None, palette="studio"):
    """Streamlines from field.streamlines(), `width` px wide (px_per_pt k sizes the arrows).
    With `color`, every line gets that color."""
    w = max(1, round(width * ss))
    plain = color or hex_to_rgb(TEXT)
    size = (4 * k + width * 2) * ss
    for pts, speed, arrows in lines:
        if len(pts) < 2:
            continue
        p = pts * ss
        if colored and color is None:
            cols = vector_colors(speed / vref, True, palette)
            for s in range(0, len(p) - 1, 8):
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


def compose(field, grid, view, t, vref, style, arrows=False, lines=None, traces=(), ss=2, guides=True):
    """The static layer: background, guides (if `guides`), streamlines, arrows (if `arrows`) and traced lines,
    drawn at `ss`× and scaled down for antialiasing."""
    colored = style.colored and style.background == "none"   # on a heatmap, vectors are drawn in light color
    k = style.px_per_pt
    img = heatmap(grid, view, style.background, vref, style.palette, ss)
    d = ImageDraw.Draw(img)
    if guides:
        draw_guides(d, view, style, ss, grid=style.background == "none")
    if lines:
        draw_lines(d, lines, vref, colored, style.stream_heads, ss, style.px(style.line_width), k,
                   palette=style.palette)
    if arrows:
        draw_arrows(d, field, view, t, vref, style, colored, ss)
    if traces:
        draw_lines(d, traces, vref, False, True, ss, style.px(style.trace_width), k, color=(255, 255, 255))
    if ss > 1:
        img = img.resize((view.w, view.h), Image.LANCZOS)
    return np.asarray(img)


def screen(a, b, gain=1.0):
    """Screen blend of two uint8 images, 1 − (1 − a)(1 − gain·b): light adds up without clipping to the brighter one."""
    out = np.empty_like(a)
    kernels.screen(np.ascontiguousarray(a), np.ascontiguousarray(b), float(gain), out)
    return out


def build_static(field, view, t, style, arrows=False, streamlines=False, grid_needed=False, pins=(), ss=2,
                 guides=True, vref=None):
    """Samples the field and draws the static layer of `view`. Returns (image, grid or None, vref).
    `pins` are world points with a traced streamline; `grid_needed` asks for the grid even if nothing drawn
    here reads it (the particles do). vref defaults to the typical speed in the view."""
    if vref is None:
        vref = fieldmod.reference_speed(field, view, t)
    grid = None
    if grid_needed or streamlines or style.background != "none" or len(pins):
        grid = fieldmod.FieldGrid(field, view, t, spacing=style.px(style.sample))
    lines = None
    if streamlines:
        lines = fieldmod.streamlines(grid, view, vref, style.px(style.stream_sep), style.px(style.step))
    traces = ()
    if len(pins):
        pts = np.array([view.to_px(x, y) for x, y in pins])
        traces = fieldmod.trace_through(grid, view, vref, pts, style.px(1.0))
    img = compose(field, grid, view, t, vref, style, arrows, lines, traces, ss=ss, guides=guides)
    return img, grid, vref
