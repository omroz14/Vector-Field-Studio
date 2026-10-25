"""Vector fields: presets, safe expression parsing, the view transform, streamline tracing and particles."""
import ast

import numpy as np

# key, label, Fx, Fy, view center, half of the view height (world units)
PRESETS = (
    ("uniform", "Uniform", "1", "0.5", (0, 0), 3),
    ("source", "Source", "x", "y", (0, 0), 3),
    ("sink", "Sink", "-x", "-y", (0, 0), 3),
    ("vortex", "Vortex", "-y", "x", (0, 0), 3),
    ("saddle", "Saddle", "x", "-y", (0, 0), 3),
    ("spiral", "Spiral sink", "-0.3*x - y", "x - 0.3*y", (0, 0), 3),
    ("shear", "Shear", "y", "0", (0, 0), 3),
    ("dipole", "Dipole",
     "(x-1)/((x-1)^2 + y^2) - (x+1)/((x+1)^2 + y^2)",
     "y/((x-1)^2 + y^2) - y/((x+1)^2 + y^2)", (0, 0), 2.5),
    ("cells", "Cellular", "sin(x)*cos(y)", "-cos(x)*sin(y)", (0, 0), 4),
    ("pendulum", "Pendulum", "y", "-sin(x)", (0, 0), 4),
    ("vanderpol", "Van der Pol", "y", "(1 - x^2)*y - x", (0, 0), 4),
    ("gyre", "Double gyre (t)",
     "-0.1*pi*sin(pi*(0.25*sin(0.63*t)*x^2 + (1 - 0.5*sin(0.63*t))*x))*cos(pi*y)",
     "0.1*pi*cos(pi*(0.25*sin(0.63*t)*x^2 + (1 - 0.5*sin(0.63*t))*x))*sin(pi*y)"
     "*(0.5*sin(0.63*t)*x + 1 - 0.5*sin(0.63*t))", (1, 0.5), 0.6),
)

VARIABLES = ("x", "y", "t", "r", "theta")
FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "atan2": np.arctan2, "arctan2": np.arctan2, "hypot": np.hypot,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "sign": np.sign, "floor": np.floor, "min": np.minimum, "max": np.maximum,
    "pi": np.pi, "e": np.e,
}


# Syntax allowed in an expression: arithmetic, comparisons (as 0/1, e.g. (x > 0)*x) and calls of FUNCTIONS.
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
          ast.Constant, ast.operator, ast.unaryop, ast.cmpop)


def _compile(expr, name):
    """Compiles one component. Only the variables and FUNCTIONS may appear: no attributes, no indexing."""
    src = expr.strip().replace("^", "**")
    if not src:
        raise ValueError(f"{name} is empty.")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{name}: {exc.msg}.") from None
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise ValueError(f"{name}: “{ast.get_source_segment(src, node) or type(node).__name__}” "
                             "is not supported.")
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS and node.id not in VARIABLES:
            raise ValueError(f"{name}: unknown name “{node.id}”.")
        if isinstance(node, ast.Call) and (node.keywords or not isinstance(node.func, ast.Name)):
            raise ValueError(f"{name}: only plain calls like sin(x) are supported.")
    return compile(tree, name, "eval")


class Field:
    """F(x, y, t) = (Fx, Fy) given as two expressions. Raises ValueError on a bad expression."""

    def __init__(self, fx, fy):
        self.fx, self.fy = fx.strip(), fy.strip()
        self._code = (_compile(fx, "Fx"), _compile(fy, "Fy"))
        self.uses_t = any("t" in c.co_names for c in self._code)
        self(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))   # surfaces runtime errors now

    def __call__(self, x, y, t=0.0):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        with np.errstate(all="ignore"):
            ns = {"__builtins__": {}, **FUNCTIONS,
                  "x": x, "y": y, "t": t, "r": np.hypot(x, y), "theta": np.arctan2(y, x)}
            out = []
            for name, code in zip(("Fx", "Fy"), self._code):
                try:
                    val = np.asarray(eval(code, ns), float)
                    val = np.broadcast_to(val, x.shape)
                except Exception as exc:
                    raise ValueError(f"{name}: {exc}") from None
                out.append(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
        return out[0], out[1]


class View:
    """Maps world coordinates to stage pixels. y grows upward in the world and downward on screen."""

    def __init__(self, cx, cy, half_h, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.scale = h / (2 * half_h)   # pixels per world unit

    def to_px(self, x, y):
        return self.w / 2 + (x - self.cx) * self.scale, self.h / 2 - (y - self.cy) * self.scale

    def to_world(self, px, py):
        return self.cx + (px - self.w / 2) / self.scale, self.cy - (py - self.h / 2) / self.scale

    def grid(self, nx, ny):
        """World coordinates of the centers of an nx × ny grid of cells covering the stage (row 0 on top)."""
        px = (np.arange(nx) + 0.5) * self.w / nx
        py = (np.arange(ny) + 0.5) * self.h / ny
        return self.to_world(*np.meshgrid(px, py))


def reference_speed(field, view, t):
    """A typical |F| in the view (90th percentile), so singular points do not flatten the color scale."""
    u, v = field(*view.grid(48, 32), t)
    m = np.hypot(u, v)
    ref = float(np.percentile(m, 90))
    if ref <= 1e-12:
        ref = float(m.max())
    return ref if ref > 1e-12 else 1.0


def _pixel_dir(field, view, t, p):
    """Unit direction of the field in pixel space at the pixel points p (N, 2), and |F| there."""
    u, v = field(*view.to_world(p[:, 0], p[:, 1]), t)
    m = np.hypot(u, v)
    with np.errstate(all="ignore"):
        d = np.stack([u / m, -v / m], axis=1)
    return np.nan_to_num(d), m


def trace(field, view, t, seeds, vref, step=3.0, nsteps=250):
    """Follows the field from the pixel points `seeds` (N, 2) forward and backward in steps of `step` px (RK2).
    Returns [(points, speeds)] for both directions: (nsteps, N, 2) and (nsteps, N), NaN after a line stops."""
    n = len(seeds)
    tiny = 1e-6 * vref
    out = []
    for sign in (1.0, -1.0):
        pts = np.full((nsteps, n, 2), np.nan)
        spd = np.full((nsteps, n), np.nan)
        pos = seeds.astype(float)
        prev = np.zeros((n, 2))
        idx = np.arange(n)
        for i in range(nsteps):
            if idx.size == 0:
                break
            p = pos[idx]
            d1, m = _pixel_dir(field, view, t, p)
            d2, _ = _pixel_dir(field, view, t, p + sign * 0.5 * step * d1)
            q = p + sign * step * d2
            ok = ((m > tiny) & ((d2 * d2).sum(1) > 0.5)
                  & (q[:, 0] >= 0) & (q[:, 0] < view.w) & (q[:, 1] >= 0) & (q[:, 1] < view.h)
                  & ((d2 * prev[idx]).sum(1) > -0.5))   # a sharp turn back means a stagnation point
            idx, q, d2, m = idx[ok], q[ok], d2[ok], m[ok]
            pts[i, idx] = q
            spd[i, idx] = m
            pos[idx] = q
            prev[idx] = d2
        out.append((pts, spd))
    return out


def _leading(pts, spd):
    """The part of one traced half before its first NaN."""
    valid = ~np.isnan(pts[:, 0])
    n = len(valid) if valid.all() else int(np.argmin(valid))
    return pts[:n], spd[:n]


def streamlines(field, view, t, vref, sep, step=3.0):
    """Evenly spaced streamlines: seeds on a jittered grid, each line stops where it meets an earlier one.
    Returns [(points (N, 2) px, speeds (N,), index of the seed point)]. Lines run in the field direction."""
    rng = np.random.default_rng(7)
    gx, gy = np.meshgrid(np.arange(sep / 2, view.w, sep), np.arange(sep / 2, view.h, sep))
    seeds = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-sep / 3, sep / 3, (gx.size, 2))
    seeds = np.clip(seeds, 0, [view.w - 1, view.h - 1])
    nsteps = int(min(400, 1.5 * max(view.w, view.h) / step))
    (fp, fs), (bp, bs) = trace(field, view, t, seeds, vref, step, nsteps)
    su, sv = field(*view.to_world(seeds[:, 0], seeds[:, 1]), t)
    seed_speed = np.hypot(su, sv)

    cell = sep * 0.5
    occ = np.zeros((int(view.h / cell) + 1, int(view.w / cell) + 1), bool)
    lines = []
    for k in rng.permutation(len(seeds)):
        cx, cy = (seeds[k] // cell).astype(int)
        if occ[cy, cx]:
            continue
        halves = []
        for pts, spd in ((fp[:, k], fs[:, k]), (bp[:, k], bs[:, k])):
            pts, spd = _leading(pts, spd)
            c = (pts // cell).astype(int)
            hit = occ[c[:, 1], c[:, 0]]
            if hit.any():
                n = int(np.argmax(hit))
                pts, spd = pts[:n], spd[:n]
            halves.append((pts, spd))
        (f, fsp), (b, bsp) = halves
        line = np.vstack([b[::-1], seeds[k][None], f])
        if len(line) < 6:
            continue
        c = (line // cell).astype(int)
        occ[c[:, 1], c[:, 0]] = True
        lines.append((line, np.concatenate([bsp[::-1], seed_speed[k:k + 1], fsp]), len(b)))
    return lines


def trace_through(field, view, t, vref, points_px, step=2.0):
    """Full streamlines through the given pixel points, in the format of streamlines()."""
    seeds = np.asarray(points_px, float).reshape(-1, 2)
    nsteps = int(min(800, 3 * max(view.w, view.h) / step))
    (fp, fs), (bp, bs) = trace(field, view, t, seeds, vref, step, nsteps)
    su, sv = field(*view.to_world(seeds[:, 0], seeds[:, 1]), t)
    lines = []
    for k in range(len(seeds)):
        f, fsp = _leading(fp[:, k], fs[:, k])
        b, bsp = _leading(bp[:, k], bs[:, k])
        speed = np.concatenate([bsp[::-1], [np.hypot(su[k], sv[k])], fsp])
        lines.append((np.vstack([b[::-1], seeds[k][None], f]), speed, len(b)))
    return lines


class Particles:
    """Massless tracers carried by the field. Positions are in world coordinates."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.pos = None

    def _spawn(self, n, view):
        px = self.rng.uniform(0, view.w, n)
        py = self.rng.uniform(0, view.h, n)
        return np.stack(view.to_world(px, py), axis=1)

    def reset(self, n, view):
        self.pos = self._spawn(n, view)
        self.life = self.rng.uniform(40, 140, n)
        self.age = self.rng.uniform(0, self.life)

    def step(self, field, view, t, dt):
        """Moves every particle by one RK2 step and respawns the ones that left the view or grew old.
        Returns (previous positions, speeds, respawned mask)."""
        old = self.pos.copy()
        x, y = old.T
        u, v = field(x, y, t)
        u2, v2 = field(x + 0.5 * dt * u, y + 0.5 * dt * v, t + 0.5 * dt)
        self.pos = old + dt * np.stack([u2, v2], axis=1)
        self.age += 1
        px, py = view.to_px(*self.pos.T)
        dead = ((self.age > self.life) | ~np.isfinite(px) | ~np.isfinite(py)
                | (px < 0) | (px >= view.w) | (py < 0) | (py >= view.h))
        n = int(dead.sum())
        if n:
            self.pos[dead] = self._spawn(n, view)
            self.age[dead] = 0
            self.life[dead] = self.rng.uniform(40, 140, n)
        return old, np.hypot(u2, v2), dead
