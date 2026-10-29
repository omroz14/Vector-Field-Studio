"""Numba kernels: bilinear sampling of a gridded field, RK4 streamlines spaced by Jobard–Lefer, particle advection.

Positions are in grid units: node (i, j) holds F at the world point (x0 + i·h, y0 + j·h), so the row index j
grows with y. The kernels only read the grid, never the field's expressions, so they are compiled once and
cached on disk; `nogil` lets them run on the render thread while Tk keeps the main one.
"""
import math

import numba as nb
import numpy as np

_jit = nb.njit(cache=True, nogil=True)


@_jit
def _sample(U, V, gx, gy):
    """F at the grid position (gx, gy), interpolated bilinearly. ok is False outside the grid (and for NaN)."""
    ny, nx = U.shape
    if not (0.0 <= gx <= nx - 1 and 0.0 <= gy <= ny - 1):
        return 0.0, 0.0, False
    i = min(int(gx), nx - 2)
    j = min(int(gy), ny - 2)
    fx = gx - i
    fy = gy - j
    a = (1.0 - fx) * (1.0 - fy)
    b = fx * (1.0 - fy)
    c = (1.0 - fx) * fy
    d = fx * fy
    u = a * U[j, i] + b * U[j, i + 1] + c * U[j + 1, i] + d * U[j + 1, i + 1]
    v = a * V[j, i] + b * V[j, i + 1] + c * V[j + 1, i] + d * V[j + 1, i + 1]
    return u, v, True


@_jit
def _unit(U, V, gx, gy, sign, tiny):
    """Unit direction sign·F/|F| at (gx, gy) and |F|. ok is False outside the grid or where |F| <= tiny."""
    u, v, ok = _sample(U, V, gx, gy)
    m = math.sqrt(u * u + v * v)
    if not ok or m <= tiny:
        return 0.0, 0.0, m, False
    return sign * u / m, sign * v / m, m, True


@_jit
def _rk4(U, V, gx, gy, k1x, k1y, sign, ds, tiny):
    """One RK4 step of arc length ds along the unit field, given its direction k1 at the start point."""
    k2x, k2y, _, ok2 = _unit(U, V, gx + 0.5 * ds * k1x, gy + 0.5 * ds * k1y, sign, tiny)
    k3x, k3y, _, ok3 = _unit(U, V, gx + 0.5 * ds * k2x, gy + 0.5 * ds * k2y, sign, tiny)
    k4x, k4y, _, ok4 = _unit(U, V, gx + ds * k3x, gy + ds * k3y, sign, tiny)
    if not (ok2 and ok3 and ok4):
        return gx, gy, False
    dx = (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    dy = (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    if dx * dx + dy * dy < 0.25:   # the unit directions disagree: a singular point lies within the step
        return gx, gy, False
    return gx + ds * dx, gy + ds * dy, True


# ---------- occupancy: a spatial hash of line points, cells of size `cell` ----------
# occ = (head (ncy, ncx), next, x, y, line id, arc position, count[1]); each cell is a linked list, newest first.

def _new_occ(nx, ny, cell, cap):
    ncx, ncy = int((nx - 1) / cell) + 1, int((ny - 1) / cell) + 1
    return (np.full((ncy, ncx), -1, np.int64), np.empty(cap, np.int64), np.empty(cap), np.empty(cap),
            np.empty(cap, np.int64), np.empty(cap), np.zeros(1, np.int64))


@_jit
def _occ_add(occ, cell, x, y, line, s):
    head, nxt, px, py, pl, ps, cnt = occ
    k = cnt[0]
    if k >= px.size:
        return False
    cx, cy = int(x / cell), int(y / cell)
    px[k], py[k], pl[k], ps[k] = x, y, line, s
    nxt[k] = head[cy, cx]
    head[cy, cx] = k
    cnt[0] = k + 1
    return True


@_jit
def _occ_pop(occ, cell, k0):
    """Removes the points added after the first k0, newest first, which keeps every list consistent."""
    head, nxt, px, py, pl, ps, cnt = occ
    for k in range(cnt[0] - 1, k0 - 1, -1):
        head[int(py[k] / cell), int(px[k] / cell)] = nxt[k]
    cnt[0] = k0


@_jit
def _occ_near(occ, cell, x, y, r, line, s, s_skip, others):
    """The index of a stored point closer than r (<= cell) to (x, y), or -1. Points of `line` within arc s_skip
    of s are its own neighbours and do not count; points of other lines count only if `others`."""
    head, nxt, px, py, pl, ps, cnt = occ
    cx, cy = int(x / cell), int(y / cell)
    r2 = r * r
    for j in range(max(cy - 1, 0), min(cy + 2, head.shape[0])):
        for i in range(max(cx - 1, 0), min(cx + 2, head.shape[1])):
            k = head[j, i]
            while k >= 0:
                dx, dy = px[k] - x, py[k] - y
                if dx * dx + dy * dy < r2:
                    if pl[k] == line:
                        if abs(ps[k] - s) >= s_skip:
                            return k
                    elif others:
                        return k
                k = nxt[k]
    return -1


# ---------- tracing ----------

@_jit
def _half(U, V, occ, cell, gx, gy, sign, ds, tiny, d_test, s_skip, others, max_steps, max_turn, line, bx, by, bm):
    """Traces from (gx, gy) along sign·F until the line leaves the grid, stalls, turns back at a fixed point,
    winds more than max_turn radians, or comes within d_test of another line (if `others`) or of its own
    earlier part. Stores the points in occ and in the buffers bx, by, bm (|F|); returns their count."""
    k1x, k1y, m, ok = _unit(U, V, gx, gy, sign, tiny)
    if not ok:
        return 0
    n = 0
    turn = 0.0
    while n < max_steps:
        qx, qy, ok = _rk4(U, V, gx, gy, k1x, k1y, sign, ds, tiny)
        if not ok:
            break
        nkx, nky, m, ok = _unit(U, V, qx, qy, sign, tiny)
        dot = nkx * k1x + nky * k1y
        if not ok or dot < 0.0:
            break
        turn += abs(math.atan2(k1x * nky - k1y * nkx, dot))
        s = sign * (n + 1) * ds
        if turn > max_turn:
            break
        hit = _occ_near(occ, cell, qx, qy, d_test, line, s, s_skip, others)
        if hit >= 0:
            # Back at its own seed: a closed orbit. End exactly on the seed instead of a gap short of it.
            if occ[4][hit] == line and abs(occ[5][hit]) < d_test and n < max_steps:
                bx[n], by[n], bm[n] = occ[2][hit], occ[3][hit], m
                n += 1
            break
        if not _occ_add(occ, cell, qx, qy, line, s):
            break
        bx[n], by[n], bm[n] = qx, qy, m
        n += 1
        gx, gy, k1x, k1y = qx, qy, nkx, nky
    return n


@_jit
def _grow(U, V, occ, cell, sx, sy, ds, tiny, d_test, s_skip, others, max_steps, max_turn, min_len,
          fx, fy, fm, bx, by, bm, lx, ly, lm, starts, seed_at, nout, nl):
    """Traces the line through the seed (sx, sy) both ways and appends it to the output (lx, ly, lm in field
    direction; line nl spans starts[nl]:starts[nl + 1], its seed is at seed_at[nl]). A line shorter than
    min_len is dropped again. Returns the new (point count, line count)."""
    u, v, ok = _sample(U, V, sx, sy)
    m = math.sqrt(u * u + v * v)
    if not ok or m <= tiny:
        return nout, nl
    k0 = occ[6][0]
    if not _occ_add(occ, cell, sx, sy, nl, 0.0):
        return nout, nl
    nf = _half(U, V, occ, cell, sx, sy, 1.0, ds, tiny, d_test, s_skip, others, max_steps, max_turn, nl, fx, fy, fm)
    nb = _half(U, V, occ, cell, sx, sy, -1.0, ds, tiny, d_test, s_skip, others, max_steps, max_turn, nl, bx, by, bm)
    if (nf + nb) * ds < min_len or nout + nf + nb + 1 > lx.size or nl + 2 > starts.size:
        _occ_pop(occ, cell, k0)
        return nout, nl
    for i in range(nb - 1, -1, -1):
        lx[nout], ly[nout], lm[nout] = bx[i], by[i], bm[i]
        nout += 1
    seed_at[nl] = nout
    lx[nout], ly[nout], lm[nout] = sx, sy, m
    nout += 1
    for i in range(nf):
        lx[nout], ly[nout], lm[nout] = fx[i], fy[i], fm[i]
        nout += 1
    nl += 1
    starts[nl] = nout
    return nout, nl


@_jit
def _evenly_spaced(U, V, occ, d_sep, ds, tiny, min_len, max_steps, lat_k, lat_ox, lat_oy, cap):
    ny, nx = U.shape
    d_test = 0.5 * d_sep
    s_skip = math.pi * d_test       # own points closer along the line than this are its neighbours
    max_turn = 4.0 * math.pi        # two turns: stops lines circling a centre smaller than d_test
    fx, fy, fm = np.empty(max_steps), np.empty(max_steps), np.empty(max_steps)
    bx, by, bm = np.empty(max_steps), np.empty(max_steps), np.empty(max_steps)
    lx, ly, lm = np.empty(cap), np.empty(cap), np.empty(cap)
    starts = np.zeros(cap // 4 + 2, np.int64)
    seed_at = np.zeros(cap // 4 + 2, np.int64)

    # Seeds of last resort: world-anchored lattice nodes, nearest to the centre first.
    gi = np.arange(lat_ox, nx, lat_k)
    gj = np.arange(lat_oy, ny, lat_k)
    sx = np.empty(gi.size * gj.size)
    sy = np.empty(gi.size * gj.size)
    for a in range(gj.size):
        for b in range(gi.size):
            sx[a * gi.size + b] = gi[b]
            sy[a * gi.size + b] = gj[a]
    order = np.argsort((sx - 0.5 * (nx - 1)) ** 2 + (sy - 0.5 * (ny - 1)) ** 2)

    every = max(1, int(0.5 * d_sep / ds))   # candidate seeds every half separation along a line
    nout, nl, q, li = 0, 0, 0, 0
    while True:
        if q < nl:
            # Jobard–Lefer: new seeds one separation to either side of the lines already laid out.
            for p in range(starts[q], starts[q + 1], every):
                u, v, ok = _sample(U, V, lx[p], ly[p])
                m = math.sqrt(u * u + v * v)
                if m <= tiny:
                    continue
                for side in (-1.0, 1.0):
                    cx = lx[p] - side * d_sep * v / m
                    cy = ly[p] + side * d_sep * u / m
                    if (0.0 <= cx <= nx - 1 and 0.0 <= cy <= ny - 1
                            and _occ_near(occ, d_sep, cx, cy, d_sep, -1, 0.0, 0.0, True) < 0):
                        nout, nl = _grow(U, V, occ, d_sep, cx, cy, ds, tiny, d_test, s_skip, True, max_steps,
                                         max_turn, min_len, fx, fy, fm, bx, by, bm, lx, ly, lm, starts, seed_at,
                                         nout, nl)
            q += 1
        elif li < order.size:
            k = order[li]
            li += 1
            if _occ_near(occ, d_sep, sx[k], sy[k], d_sep, -1, 0.0, 0.0, True) < 0:
                nout, nl = _grow(U, V, occ, d_sep, sx[k], sy[k], ds, tiny, d_test, s_skip, True, max_steps,
                                 max_turn, min_len, fx, fy, fm, bx, by, bm, lx, ly, lm, starts, seed_at, nout, nl)
        else:
            break
    return lx[:nout], ly[:nout], lm[:nout], starts[:nl + 1], seed_at[:nl]


def evenly_spaced(U, V, d_sep, ds, tiny, lattice=(1, 0, 0), min_len=None):
    """Evenly spaced streamlines (Jobard & Lefer 1997) over the whole grid, in grid units. Lines stop where
    they come within d_sep/2 of another line. `lattice` = (k, ox, oy): fallback seeds sit at the grid nodes
    (ox + a·k, oy + b·k), so they stay put in the world when the grid moves by whole nodes.
    Returns (x, y, |F|, starts, seed_at): line n is x[starts[n]:starts[n + 1]] in field direction."""
    ny, nx = U.shape
    d_sep = max(float(d_sep), 2.0 * ds)
    max_steps = int(4 * (nx + ny) / ds) + 16
    cap = int(3.0 * nx * ny / (d_sep * ds)) + 2 * max_steps + 64
    occ = _new_occ(nx, ny, d_sep, cap)
    k, ox, oy = lattice
    return _evenly_spaced(U, V, occ, d_sep, float(ds), float(tiny), float(0.75 * d_sep if min_len is None else min_len),
                          max_steps, int(k), int(ox), int(oy), cap)


@_jit
def _through(U, V, occ, cell, sx, sy, ds, tiny, d_self, max_steps, max_turn, cap):
    fx, fy, fm = np.empty(max_steps), np.empty(max_steps), np.empty(max_steps)
    bx, by, bm = np.empty(max_steps), np.empty(max_steps), np.empty(max_steps)
    lx, ly, lm = np.empty(cap), np.empty(cap), np.empty(cap)
    starts = np.zeros(sx.size + 1, np.int64)
    seed_at = np.full(sx.size, -1, np.int64)
    nout = 0
    for n in range(sx.size):
        starts[n] = nout
        nout2, nl = _grow(U, V, occ, cell, sx[n], sy[n], ds, tiny, d_self, math.pi * d_self, False, max_steps,
                          max_turn, 0.0, fx, fy, fm, bx, by, bm, lx, ly, lm, starts, seed_at, nout, n)
        nout = nout2
        starts[n + 1] = nout
    return lx[:nout], ly[:nout], lm[:nout], starts, seed_at


def trace_through(U, V, sx, sy, ds, tiny, d_self=1.0, max_turn=20 * math.pi):
    """The full streamline through each seed (grid units), ignoring the other lines. A line stops where it
    comes back within d_self of itself (a closed orbit, or a spiral onto a limit cycle). Same output as
    evenly_spaced; a seed outside the grid or at a fixed point gives an empty line with seed_at -1."""
    ny, nx = U.shape
    sx, sy = np.asarray(sx, float).ravel(), np.asarray(sy, float).ravel()
    max_steps = int(4 * (nx + ny) / ds) + 16
    cap = sx.size * (2 * max_steps + 1) + 1
    cell = max(float(d_self), 1.0)
    occ = _new_occ(nx, ny, cell, cap)
    return _through(U, V, occ, cell, sx, sy, float(ds), float(tiny), float(d_self), max_steps, float(max_turn), cap)


# ---------- particles ----------

@_jit
def advect(U, V, x0, y0, h, X, Y, dt, speed, alive):
    """Moves the world points (X, Y) in place by one RK4 step of time dt through the gridded field.
    speed gets |F| of the step; alive is False for points whose step left the grid."""
    inv = 1.0 / h
    for n in range(X.size):
        gx = (X[n] - x0) * inv
        gy = (Y[n] - y0) * inv
        a = 0.5 * dt * inv
        u1, v1, ok1 = _sample(U, V, gx, gy)
        u2, v2, ok2 = _sample(U, V, gx + a * u1, gy + a * v1)
        u3, v3, ok3 = _sample(U, V, gx + a * u2, gy + a * v2)
        u4, v4, ok4 = _sample(U, V, gx + 2.0 * a * u3, gy + 2.0 * a * v3)
        if not (ok1 and ok2 and ok3 and ok4):
            alive[n] = False
            speed[n] = 0.0
            continue
        du = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
        dv = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        X[n] += dt * du
        Y[n] += dt * dv
        speed[n] = math.sqrt(du * du + dv * dv)
        alive[n] = True


# ---------- particle trails ----------

@_jit
def splat_paths(D, S, x0, y0, x1, y1, w, speed, nsub, rho):
    """Draws each path (x0, y0) -> (x1, y1) in pixels as nsub samples (the start point belongs to the previous
    frame) into the density D and the speed-weighted density S. A sample adds a tent of radius rho px with
    peak w: rho = 1 is a bilinear splat, a larger rho a proportionally wider one. Pixel i is centred at i + 0.5."""
    h, wd = D.shape
    r = int(math.ceil(rho))
    for n in range(x0.size):
        wn = w[n]
        if wn <= 0.0:
            continue
        sn = wn * speed[n]
        for k in range(1, nsub + 1):
            a = k / nsub
            x = x0[n] + a * (x1[n] - x0[n]) - 0.5
            y = y0[n] + a * (y1[n] - y0[n]) - 0.5
            if not (-rho < x < wd - 1 + rho and -rho < y < h - 1 + rho):
                continue
            ix, iy = int(math.floor(x)), int(math.floor(y))
            for j in range(max(iy - r + 1, 0), min(iy + r + 1, h)):
                wy = 1.0 - abs(y - j) / rho
                if wy <= 0.0:
                    continue
                for i in range(max(ix - r + 1, 0), min(ix + r + 1, wd)):
                    wx = 1.0 - abs(x - i) / rho
                    if wx > 0.0:
                        D[j, i] += wn * wx * wy
                        S[j, i] += sn * wx * wy


@nb.njit(cache=True, nogil=True, parallel=True)
def tonemap(D, S, kappa, inv_vref, lut, delta, out):
    """Trail colours into out (uint8 h × w × 3): brightness 1 − exp(−κD), colour lut(S / (D·vref)), so dense
    trails saturate softly and every pixel shows the mean speed of what passed it. Then fades D and S by delta
    for the next frame (tiny values go to zero, so they never turn into slow denormals)."""
    h, w = D.shape
    for j in nb.prange(h):
        for i in range(w):
            d = D[j, i]
            if d > 1e-4:
                b = 1.0 - math.exp(-kappa * d)
                q = min(max(S[j, i] / d * inv_vref, 0.0), 1.0) * 255.0
                q0 = min(int(q), 254)   # between two LUT entries: the colour changes smoothly with speed
                f = q - q0
                for c in range(3):
                    out[j, i, c] = np.uint8(b * ((1.0 - f) * lut[q0, c] + f * lut[q0 + 1, c]) + 0.5)
                D[j, i] = d * delta
                S[j, i] *= delta
            else:
                out[j, i, 0] = out[j, i, 1] = out[j, i, 2] = 0
                D[j, i] = 0.0
                S[j, i] = 0.0


@nb.njit(cache=True, nogil=True, parallel=True)
def screen(a, b, gain, out):
    """Screen blend of uint8 images, out = 1 − (1 − a)(1 − gain·b): light adds up without clipping. out may be a."""
    h, w, nc = a.shape
    for j in nb.prange(h):
        for i in range(w):
            for c in range(nc):
                x = float(a[j, i, c])
                y = gain * b[j, i, c]
                out[j, i, c] = np.uint8(min(x + y - x * y / 255.0, 255.0) + 0.5)


def warm_up():
    """Compiles (or loads from the disk cache) every kernel on a tiny grid."""
    g = np.linspace(-1, 1, 16)
    X, Y = np.meshgrid(g, g)
    U, V = (-Y).astype(np.float32), X.astype(np.float32)
    evenly_spaced(U, V, 3.0, 0.5, 1e-9)
    trace_through(U, V, [8.0], [4.0], 0.5, 1e-9)
    px, py = np.array([0.5]), np.array([0.0])
    advect(U, V, -1.0, -1.0, 2 / 15, px, py, 0.01, np.empty(1), np.empty(1, np.bool_))
    D, S = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32)
    one = np.ones(1)
    splat_paths(D, S, one, one, 2 * one, 2 * one, one, one, 2, 1.0)
    rgb = np.zeros((4, 4, 3), np.uint8)
    tonemap(D, S, 2.0, 1.0, np.zeros((256, 3), np.float32), 0.9, rgb)
    screen(rgb, rgb, 1.0, rgb)
