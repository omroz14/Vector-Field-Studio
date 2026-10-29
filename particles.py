"""Particles and the trails they paint, independent of Tk: the stage runs one Flow, an exporter its own.

Time is in seconds. At `fps` frames per second the trails fade by exp(−1/(τ·fps)) per frame, particles live a
few seconds and the field time advances by frame_dt(), so 30 and 60 fps show the same motion.

Trails are two float buffers: the density D of everything that passed a pixel and S, the same weighted by
speed. A pixel shows brightness 1 − exp(−κD) in the colour of the mean speed S/D, so crowded trails saturate
softly and do not clip. Each particle fades in and out over 15 % of its life, so nothing pops.

Particles live in the world, in a domain a margin larger than the view, and keep a constant density on the
screen: a coarse grid of cells counts them every frame, the excess in crowded cells fades out early and new
particles are born in the emptiest cells. That one mechanism evens out zooming, panning, resizing, sinks and
sources. Particles that stall at a fixed point fade out too.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

import kernels
import render
from theme import TEXT, hex_to_rgb

LIFE = (1.3, 4.7)   # seconds a particle lives before it is born again elsewhere
FADE = 0.15         # part of the life spent fading in, and again fading out
MARGIN = 64.0       # points of off-screen domain around the view, where particles are born and flow in from
CELL = 32.0         # points: the smallest density cell
PER_CELL = 8        # particles a cell holds at the target density, at least (fewer would be all noise)
CROWDED = 1.5       # a cell with more than CROWDED × target + 2 particles sheds its excess
MOVES = 0.08        # at most this part of all particles starts fading out for balance in one frame
MOVE_FADE = 0.2     # seconds: a particle moved for balance fades out this fast, so the rim fills quickly
STALL = 0.02        # a particle slower than STALL × vref ...
STALL_S = 0.5       # ... for this many seconds fades out
GLOW_RADIUS = 6.0   # points
GLOW = 0.6          # strength of the glow


def frame_dt(field, view, style, vref, fps, t_rate):
    """Field time per frame. A field that uses t runs at t_rate units of t per second whatever the zoom;
    in a steady one, particles at the typical speed vref cover style.particle_speed points per second."""
    if field.uses_t:
        return t_rate / fps
    return style.particle_speed * style.px_per_pt / (fps * vref * view.scale)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class Flow:
    """Particles at `density` per 100 × 100 points over `view`, and their trails, at `fps` frames per second.
    `px_per_pt` is the style's: it sizes the margin and the density cells."""

    _ARRAYS = ("x", "y", "age", "life", "fin", "fout", "still", "fading")

    def __init__(self, density, view, fps, px_per_pt=1.0, seed=None):
        self.density, self.fps, self.k = density, fps, px_per_pt
        self.rng = np.random.default_rng(seed)
        self.view = view
        self._layout()
        self.D = np.zeros((view.h, view.w), np.float32)
        self.S = np.zeros((view.h, view.w), np.float32)
        self.rgb = np.zeros((view.h, view.w, 3), np.uint8)
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.age = np.empty(0)
        self.life = np.empty(0)
        self.fin = np.empty(0)       # seconds of fading in
        self.fout = np.empty(0)      # seconds of fading out
        self.still = np.empty(0)     # seconds spent stalled
        self.fading = np.empty(0, bool)
        self._add(self.n, None)
        self.age = self.rng.uniform(0, self.life)   # desynchronised: they do not all fade in together

    # ---------- the domain and its cells ----------

    def _layout(self):
        """The domain (the view plus a margin, in whole cells), the cells and the particle budget n."""
        v, k = self.view, self.k
        m = MARGIN * k
        area = (v.w + 2 * m) * (v.h + 2 * m)
        self.n = round(self.density * area / k ** 2 / 1e4)
        c = max(CELL * k, math.sqrt(PER_CELL * area / max(self.n, 1)))
        self.ncx, self.ncy = math.ceil((v.w + 2 * m) / c), math.ceil((v.h + 2 * m) / c)
        self.cell = c
        self.ox, self.oy = (v.w - self.ncx * c) / 2, (v.h - self.ncy * c) / 2   # px of the domain's corner
        self.target = self.n / (self.ncx * self.ncy)

    def _cells(self, px, py):
        """Cell index of each pixel point, −1 outside the domain (or not finite)."""
        with np.errstate(invalid="ignore"):
            cx = np.floor((px - self.ox) / self.cell)
            cy = np.floor((py - self.oy) / self.cell)
            ok = (cx >= 0) & (cx < self.ncx) & (cy >= 0) & (cy < self.ncy)
        return np.where(ok, cy * self.ncx + cx, -1).astype(np.int64)

    def _counts(self, cell, mask):
        return np.bincount(cell[mask], minlength=self.ncx * self.ncy)

    def _birth_cells(self, m, counts):
        """Cells for m new particles: stratified over the domain at first, later the emptiest cells."""
        nc = self.ncx * self.ncy
        if counts is None:
            reps = math.ceil(m / nc)
            return np.concatenate([self.rng.permutation(nc) for _ in range(reps)])[:m]
        deficit = np.clip(self.target - counts, 0, None)
        if deficit.sum() <= 0:
            return self.rng.integers(0, nc, m)
        return self.rng.choice(nc, m, p=deficit / deficit.sum())

    def _born(self, m, counts):
        """World positions of m new particles, each at a random point of its cell."""
        cells = self._birth_cells(m, counts)
        px = self.ox + (cells % self.ncx + self.rng.random(m)) * self.cell
        py = self.oy + (cells // self.ncx + self.rng.random(m)) * self.cell
        return self.view.to_world(px, py)

    # ---------- particle bookkeeping ----------

    def _add(self, m, counts):
        if m <= 0:
            return
        x, y = self._born(m, counts)
        life = self.rng.uniform(*LIFE, m)
        new = dict(x=x, y=y, age=np.zeros(m), life=life, fin=FADE * life, fout=FADE * life, still=np.zeros(m),
                   fading=np.zeros(m, bool))
        for name in self._ARRAYS:
            setattr(self, name, np.concatenate([getattr(self, name), new[name]]))

    def _rebirth(self, idx, counts):
        m = len(idx)
        if m == 0:
            return
        x, y = self._born(m, counts)
        life = self.rng.uniform(*LIFE, m)
        self.x[idx], self.y[idx], self.age[idx], self.life[idx] = x, y, 0.0, life
        self.fin[idx], self.fout[idx], self.still[idx], self.fading[idx] = FADE * life, FADE * life, 0.0, False

    def _drop(self, idx):
        keep = np.ones(len(self.x), bool)
        keep[idx] = False
        for name in self._ARRAYS:
            setattr(self, name, getattr(self, name)[keep])

    def _fade_out(self, mask, seconds=None):
        """Starts the fade-out of these particles now, over `seconds` (default: their own fade time); they die
        when it ends. One that would end sooner anyway keeps its course."""
        idx = np.flatnonzero(mask)
        dur = self.fin[idx] if seconds is None else np.minimum(self.fin[idx], seconds)
        sooner = self.age[idx] + dur < self.life[idx]
        idx, dur = idx[sooner], dur[sooner]
        self.life[idx] = self.age[idx] + dur
        self.fout[idx] = dur
        self.fading |= mask

    def _envelope(self):
        """Opacity of each particle: smooth fade in after birth and out before death."""
        return _smoothstep(self.age / self.fin) * _smoothstep((self.life - self.age) / self.fout)

    def _balance(self, cell, dead):
        """Density control: the excess particles of crowded cells (and beyond the budget) start fading out,
        at most MOVES of all particles per frame."""
        ok = ~dead & ~self.fading
        counts = self._counts(cell, ok)
        crowded = counts > CROWDED * self.target + 2
        out = np.zeros(len(self.x), bool)
        if crowded.any():
            cand = np.flatnonzero(ok & crowded[np.maximum(cell, 0)] & (cell >= 0))
            cand = cand[self.rng.permutation(len(cand))]
            cand = cand[np.argsort(cell[cand], kind="stable")]
            cs = cell[cand]
            first = np.searchsorted(cs, cs)                 # rank of each particle within its cell
            rank = np.arange(len(cand)) - first
            out[cand[rank >= math.ceil(self.target)]] = True
        extra = int(ok.sum()) - self.n                      # the budget shrank (a smaller stage)
        if extra > 0:
            rest = np.flatnonzero(ok & ~out)
            out[self.rng.choice(rest, min(extra, len(rest)), replace=False)] = True
        cap = max(1, int(MOVES * max(self.n, 1)))
        idx = np.flatnonzero(out)
        if len(idx) > cap:
            out[:] = False
            out[self.rng.choice(idx, cap, replace=False)] = True
        self._fade_out(out, MOVE_FADE)

    def _respawn(self, dead, cell):
        """Dead particles are born again in the emptiest cells; the count follows the budget."""
        counts = self._counts(cell, ~dead & ~self.fading & (cell >= 0))
        idx = np.flatnonzero(dead)
        surplus = len(self.x) - self.n
        if surplus > 0:
            gone, idx = idx[:surplus], idx[surplus:]
        else:
            gone = idx[:0]
        self._rebirth(idx, counts)
        if len(gone):
            self._drop(gone)
        if surplus < 0:
            self._add(-surplus, counts)

    # ---------- per frame ----------

    def _lut(self, style):
        if style.colored and style.background == "none":
            return render.PALETTES[style.palette].line.astype(np.float32)
        return np.tile(np.array(hex_to_rgb(TEXT), np.float32), (256, 1))

    def _tonemap(self, vref, style, delta):
        kernels.tonemap(self.D, self.S, style.particle_gain, 1.0 / vref, self._lut(style), delta, self.rgb)
        if style.glow:   # blurred at a quarter of the size: a wide halo for a few milliseconds
            h, w = self.rgb.shape[:2]
            small = Image.fromarray(self.rgb).reduce(4)
            blur = small.filter(ImageFilter.GaussianBlur(style.px(GLOW_RADIUS) / 4)).resize((w, h), Image.BILINEAR)
            kernels.screen(self.rgb, np.asarray(blur), GLOW, self.rgb)

    def _rho(self, style):
        """Radius of a particle's splat, px: never under 1, so trails are at least 2 px wide."""
        return max(1.0, style.px(style.particle_size) / 2)

    def paint(self, field, t, vref, style):
        """Shows the particles where they are, without moving them (the first frame, or a pause)."""
        u, v = field(self.x, self.y, t)
        px, py = self.view.to_px(self.x, self.y)
        kernels.splat_paths(self.D, self.S, px, py, px, py, self._envelope(), np.hypot(u, v), 1, self._rho(style))
        self._tonemap(vref, style, 1.0)

    def step(self, grid, dt, vref, style):
        """One frame: moves the particles by field time dt through the grid, paints their paths, keeps the
        density even and fades the trails."""
        view, sec = self.view, 1 / self.fps
        ox, oy = view.to_px(self.x, self.y)
        speed = np.empty(len(self.x))
        alive = np.empty(len(self.x), bool)
        kernels.advect(grid.U, grid.V, grid.x0, grid.y0, grid.h, self.x, self.y, dt, speed, alive)
        self.age += sec
        px, py = view.to_px(self.x, self.y)
        cell = self._cells(px, py)
        dead = ~alive | (cell < 0) | (self.age >= self.life)

        self.still = np.where(speed < STALL * vref, self.still + sec, 0.0)
        self._fade_out(~dead & ~self.fading & (self.still > STALL_S))
        self._balance(cell, dead)

        w = np.where(dead, 0.0, self._envelope())
        with np.errstate(invalid="ignore"):
            dist = np.where(dead, 0.0, np.hypot(px - ox, py - oy))
        nsub = int(np.clip(np.ceil(dist.max(initial=0) / style.px(style.substep)), 1, 12))
        kernels.splat_paths(self.D, self.S, ox, oy, px, py, w, speed, nsub, self._rho(style))
        self._respawn(dead, cell)

        tau = style.trail
        self._tonemap(vref, style, math.exp(-1 / (tau * self.fps)) if tau > 0 else 0.0)

    def retarget(self, view, vref, style):
        """Follows a pan, zoom or resize: the trails are resampled into the new view (so the picture moves
        with it), the particles stay where they are in the world and the density control evens them out."""
        old = self.view
        if (old.cx, old.cy, old.scale, old.w, old.h) == (view.cx, view.cy, view.scale, view.w, view.h):
            return
        a = render.view_affine(old, view)
        self.D, self.S = (np.array(Image.fromarray(b, "F").transform(
            (view.w, view.h), Image.AFFINE, a, Image.BILINEAR, fillcolor=0), np.float32) for b in (self.D, self.S))
        self.rgb = np.zeros((view.h, view.w, 3), np.uint8)
        self.view = view
        self._layout()
        self._tonemap(vref, style, 1.0)

    def restyle(self, vref, style):
        """Recolours the trails for a new palette, background or glow, without moving anything."""
        self._tonemap(vref, style, 1.0)

    def image(self):
        return self.rgb


def loop_lifetimes(frames, fps):
    """Lifetimes (in frames) that divide a loop of `frames`, within LIFE if any do, else the longest that
    fits under it. Dividing the loop makes every particle's schedule repeat with it."""
    divs = [d for d in range(1, frames + 1) if frames % d == 0]
    lo, hi = LIFE[0] * fps, LIFE[1] * fps
    fit = [d for d in divs if lo <= d <= hi]
    return np.array(fit or [max([d for d in divs if d <= hi] or [frames])])


class LoopFlow(Flow):
    """Particles on a fixed schedule that repeats every `frames` frames, for a seamless loop. Particle j lives
    L_j frames (L_j divides the loop), starting at its own offset φ_j; its m-th life starts at a seed s_jm
    chosen once. At frame f it is (f − φ_j) mod L_j frames old and on life ((f − φ_j) mod loop) // L_j, so
    where it is depends on f mod loop only, and so does everything painted, once each particle has been
    reborn at least once. There is no density control: it would break the repeat."""

    def __init__(self, density, view, fps, frames, px_per_pt=1.0, seed=None):
        super().__init__(density, view, fps, px_per_pt, seed)
        n, rng = len(self.x), self.rng
        self.loop = frames
        self.L = rng.choice(loop_lifetimes(frames, fps), n)
        self.phi = rng.integers(0, self.L)
        lives = frames // self.L
        self.base = np.concatenate([[0], np.cumsum(lives)[:-1]])
        sx, sy = self._born(int(lives.sum()), None)   # stratified over the domain
        order = rng.permutation(len(sx))              # so a particle's lives are not neighbours
        self.sx, self.sy = sx[order], sy[order]
        self.ok = np.ones(n, bool)                    # False once a particle has left the grid, until rebirth

    def warm_up(self, style):
        """Frames to run before frame 0. First every particle is reborn once, so positions repeat; then, with
        trails, one whole loop is painted from empty buffers and scaled to its periodic sum (see step)."""
        longest = int(self.L.max(initial=0))
        return longest + (self.loop if style.trail > 0 else 0)

    def step(self, grid, dt, vref, style, frame):
        """Frame number `frame` (any integer; only frame mod loop matters once warmed up).

        Trails are linear: B_f = δ·B_(f−1) + S_f, with S repeating every loop of T frames. So the periodic
        state is B_(−1) = Σ_(i<T) δ^i S_(−1−i) / (1 − δ^T): frames −T … −1 paint into empty buffers, which are
        then scaled by 1 / (1 − δ^T). No residue of the time before is left, the loop is exact."""
        view = self.view
        tau = style.trail
        delta = math.exp(-1 / (tau * self.fps)) if tau > 0 else 0.0
        if frame == -self.loop and delta > 0:
            self.D[:] = 0
            self.S[:] = 0
        a = (frame - self.phi) % self.L
        life = ((frame - self.phi) % self.loop) // self.L
        ox, oy = view.to_px(self.x, self.y)
        speed = np.empty(len(self.x))
        alive = np.empty(len(self.x), bool)
        kernels.advect(grid.U, grid.V, grid.x0, grid.y0, grid.h, self.x, self.y, dt, speed, alive)
        self.ok &= alive
        born = a == 0
        idx = self.base[born] + life[born]
        self.x[born], self.y[born], self.ok[born] = self.sx[idx], self.sy[idx], True
        px, py = view.to_px(self.x, self.y)

        u = a / (FADE * self.L)
        w = _smoothstep(u) * _smoothstep((self.L - a) / (FADE * self.L))
        w *= _smoothstep(speed / (2 * STALL * vref))   # stalled particles fade away, as in Flow
        w = np.where(born | ~self.ok, 0.0, w)
        with np.errstate(invalid="ignore"):
            dist = np.where(w > 0, np.hypot(px - ox, py - oy), 0.0)
        nsub = int(np.clip(np.ceil(dist.max(initial=0) / style.px(style.substep)), 1, 12))
        kernels.splat_paths(self.D, self.S, ox, oy, px, py, w, speed, nsub, self._rho(style))
        self._tonemap(vref, style, delta)
        if frame == -1 and delta > 0:
            scale = np.float32(1 / (1 - delta ** self.loop))
            self.D *= scale
            self.S *= scale
