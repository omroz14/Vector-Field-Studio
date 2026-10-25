"""Drawing: color maps, heatmaps, axes, arrows and streamlines into images the size of the stage."""
import math

import numpy as np
from PIL import Image, ImageDraw

from theme import *

GRID = "#1b1f2a"   # faint grid lines on the plain background


def _ramp(stops, pos):
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    rgb = [hex_to_rgb(s) for s in stops]
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [s[c] for s in rgb])
    return lut


# Magnitude: background -> violet -> warm orange -> white (the motion map palette of Pixel Sort Studio).
SEQ = _ramp([SURFACE, ACCENT, WARM, "#ffffff"], [0.0, 0.4, 0.75, 1.0])
# Signed quantities: violet below zero, orange above.
DIV = _ramp(["#d6ccff", ACCENT, SURFACE, WARM, "#ffe2b8"], [0.0, 0.25, 0.5, 0.75, 1.0])
# Vectors on the plain background: the bright part of SEQ, so slow vectors stay visible.
LINE = SEQ[np.linspace(90, 255, 256).astype(int)]
MAG_TOP = 200      # the magnitude heatmap stops short of white so vectors stay readable on it


def vector_colors(rel, colored):
    """(N, 3) uint8 colors for relative speeds rel (1 = typical speed)."""
    if not colored:
        return np.tile(np.array(hex_to_rgb(TEXT), np.uint8), (len(rel), 1))
    return LINE[(np.clip(rel, 0, 1) * 255).astype(int)]


def nice_step(raw):
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw * (1 - 1e-9))


def ticks(view, target_px=90):
    """(step, xs, ys): round world coordinates of the grid lines visible in the view."""
    step = nice_step(target_px / view.scale)
    x0, y1 = view.to_world(0, 0)
    x1, y0 = view.to_world(view.w, view.h)
    xs = np.arange(math.ceil(x0 / step), math.floor(x1 / step) + 1) * step
    ys = np.arange(math.ceil(y0 / step), math.floor(y1 / step) + 1) * step
    return step, xs, ys


def heatmap(field, view, t, kind, vref, res=3):
    """Background image: plain, |F|, divergence or curl, computed on a coarse grid and scaled up later."""
    if kind == "none":
        return Image.new("RGB", (view.w, view.h), SURFACE)
    nx, ny = max(2, view.w // res), max(2, view.h // res)
    X, Y = view.grid(nx, ny)
    U, V = field(X, Y, t)
    if kind == "magnitude":
        idx = (np.clip(np.hypot(U, V) / vref, 0, 1) * MAG_TOP).astype(np.uint8)
        return Image.fromarray(SEQ[idx])
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]   # dy < 0: row 0 is the top
    if kind == "divergence":
        q = np.gradient(U, dx, axis=1) + np.gradient(V, dy, axis=0)
    else:
        q = np.gradient(V, dx, axis=1) - np.gradient(U, dy, axis=0)
    # The floor keeps round-off noise of a divergence- or curl-free field at the neutral color.
    ref = max(float(np.percentile(np.abs(q), 95)), 1e-3 * vref * view.scale / view.h)
    idx = ((np.clip(q / ref, -1, 1) + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(DIV[idx])


def draw_axes(d, view, ss, grid):
    w, h = view.w * ss, view.h * ss
    _, xs, ys = ticks(view)
    if grid:
        for x in xs:
            px = view.to_px(x, 0)[0] * ss
            d.line([(px, 0), (px, h)], fill=GRID, width=ss)
        for y in ys:
            py = view.to_px(0, y)[1] * ss
            d.line([(0, py), (w, py)], fill=GRID, width=ss)
    ox, oy = view.to_px(0, 0)
    if 0 <= ox <= view.w:
        d.line([(ox * ss, 0), (ox * ss, h)], fill=BORDER, width=ss)
    if 0 <= oy <= view.h:
        d.line([(0, oy * ss), (w, oy * ss)], fill=BORDER, width=ss)


def _head(d, tip, direction, size, color):
    """A filled triangle with its point at `tip`, pointing along the unit vector `direction`."""
    ux, uy = direction
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    hw = size * 0.55
    d.polygon([tuple(tip), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], fill=color)


def draw_arrows(d, field, view, t, vref, spacing, length, equal, colored, ss):
    nx, ny = max(1, round(view.w / spacing)), max(1, round(view.h / spacing))
    X, Y = view.grid(nx, ny)
    U, V = field(X, Y, t)
    U, V = U.ravel(), V.ravel()
    m = np.hypot(U, V)
    cell = min(view.w / nx, view.h / ny)
    rel = m / vref
    if equal:
        L = np.where(rel > 1e-6, 0.8 * cell * length, 0.0)
    else:
        L = 0.9 * cell * length * np.clip(rel, 0, 1.5)
    with np.errstate(all="ignore"):
        ux, uy = np.nan_to_num(U / m), np.nan_to_num(-V / m)
    px, py = view.to_px(X.ravel(), Y.ravel())
    colors = vector_colors(rel, colored)
    width = max(1, round(1.5 * ss))
    for i in np.flatnonzero(L > 2):
        hl = min(max(0.35 * L[i], 3.0), 8.0)
        dx, dy = ux[i], uy[i]
        tip = np.array([px[i] + 0.5 * L[i] * dx, py[i] + 0.5 * L[i] * dy])
        tail = (px[i] - 0.5 * L[i] * dx, py[i] - 0.5 * L[i] * dy)
        neck = tip - 0.8 * hl * np.array([dx, dy])
        c = tuple(int(v) for v in colors[i])
        d.line([(tail[0] * ss, tail[1] * ss), tuple(neck * ss)], fill=c, width=width)
        _head(d, tip * ss, (dx, dy), hl * ss, c)


def draw_lines(d, lines, vref, colored, heads, ss, width=1.3, color=None):
    """Streamlines from field.streamlines(). With `color`, every line gets that color."""
    w = max(1, round(width * ss))
    plain = color or hex_to_rgb(TEXT)
    for pts, speed, mid in lines:
        if len(pts) < 2:
            continue
        p = pts * ss
        if colored and color is None:
            cols = vector_colors(speed / vref, True)
            for s in range(0, len(p) - 1, 6):
                c = tuple(int(v) for v in cols[s:s + 7].mean(axis=0))
                d.line(p[s:s + 7].ravel().tolist(), fill=c, width=w, joint="curve")
            head_color = tuple(int(v) for v in cols[mid])
        else:
            d.line(p.ravel().tolist(), fill=plain, width=w, joint="curve")
            head_color = plain
        if heads and len(p) > 10:
            a, b = p[max(mid - 3, 0)], p[min(mid + 3, len(p) - 1)]
            n = np.hypot(*(b - a))
            if n > 0:
                u = (b - a) / n
                size = (4 + width * 2) * ss
                _head(d, p[mid] + u * size / 2, u, size, head_color)


def compose(field, view, t, vref, background="none", arrows=None, lines=None, traces=(),
            colored=True, heads=True, ss=2):
    """The static layer: background, axes, streamlines, arrows and traced lines, drawn at `ss`× and
    scaled down for antialiasing. `arrows` holds the draw_arrows settings, or None to skip them."""
    colored = colored and background == "none"   # on a heatmap, vectors are drawn in plain light color
    img = heatmap(field, view, t, background, vref).resize((view.w * ss, view.h * ss), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    draw_axes(d, view, ss, grid=background == "none")
    if lines:
        draw_lines(d, lines, vref, colored, heads, ss)
    if arrows:
        draw_arrows(d, field, view, t, vref, colored=colored, ss=ss, **arrows)
    if traces:
        draw_lines(d, traces, vref, False, True, ss, width=2.4, color=(255, 255, 255))
    if ss > 1:
        img = img.resize((view.w, view.h), Image.LANCZOS)
    return np.asarray(img)
