"""Vector fields: presets, safe expression parsing, the view transform, the sampled grid, streamlines and particles."""
import ast
import math

import numpy as np

import kernels

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


_GLOBALS = {"__builtins__": {}, **FUNCTIONS}


class Field:
    """F(x, y, t) = (Fx, Fy) given as two expressions. Raises ValueError on a bad expression."""

    def __init__(self, fx, fy):
        self.fx, self.fy = fx.strip(), fy.strip()
        self._code = (_compile(fx, "Fx"), _compile(fy, "Fy"))
        names = set().union(*(c.co_names for c in self._code))
        self.uses_t = "t" in names
        self._uses_r, self._uses_theta = "r" in names, "theta" in names
        self(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))   # surfaces runtime errors now

    def __call__(self, x, y, t=0.0):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        with np.errstate(all="ignore"):
            ns = {"x": x, "y": y, "t": t}
            if self._uses_r:
                ns["r"] = np.hypot(x, y)
            if self._uses_theta:
                ns["theta"] = np.arctan2(y, x)
            out = []
            for name, code in zip(("Fx", "Fy"), self._code):
                try:
                    val = np.asarray(eval(code, _GLOBALS, ns), float)
                    # A constant, or a bare x or y, is not a fresh array of our own: copy before writing.
                    if val.shape != x.shape or np.may_share_memory(val, x) or np.may_share_memory(val, y):
                        val = np.broadcast_to(val, x.shape).copy()
                except Exception as exc:
                    raise ValueError(f"{name}: {exc}") from None
                bad = ~np.isfinite(val)
                if bad.any():
                    val[bad] = 0.0
                out.append(val)
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


class FieldGrid:
    """F sampled once on the world lattice (i·h, j·h) over the view plus `overscan` of its size on every side.
    The integrators read it with bilinear interpolation. Row j holds y = y0 + j·h, so rows grow upward.
    The lattice is anchored in the world: panning by whole nodes gives the very same samples."""

    def __init__(self, field, view, t=0.0, spacing=2.0, overscan=0.25):
        self.h = h = spacing / view.scale
        self.view, self.t = view, t
        mx, my = overscan * view.w, overscan * view.h
        xa, yb = view.to_world(-mx, -my)
        xb, ya = view.to_world(view.w + mx, view.h + my)
        i0, j0 = math.floor(xa / h), math.floor(ya / h)
        i1, j1 = math.ceil(xb / h), math.ceil(yb / h)
        self.i0, self.j0 = i0, j0
        self.x0, self.y0 = i0 * h, j0 * h
        X, Y = np.meshgrid(np.arange(i0, i1 + 1) * h, np.arange(j0, j1 + 1) * h)
        U, V = field(X, Y, t)
        self.U, self.V = U.astype(np.float32), V.astype(np.float32)

    def to_px(self, gx, gy, view):
        """Grid positions -> pixels of `view`."""
        return view.to_px(self.x0 + gx * self.h, self.y0 + gy * self.h)

    def to_grid(self, px, py, view):
        x, y = view.to_world(px, py)
        return (x - self.x0) / self.h, (y - self.y0) / self.h


def _lines_px(grid, view, x, y, m, starts, step, gap):
    """Kernel output -> the parts of the lines that lie in the view, as
    [(points (N, 2) px, speeds (N,), indices of the points that get a direction arrow)].
    Arrows are `gap` px apart along every line, each line starting at its own (golden-ratio) phase,
    so they scatter evenly instead of lining up across neighbouring lines."""
    px, py = grid.to_px(x, y, view)
    pts = np.stack([px, py], axis=1)
    inside = (px > -8) & (px < view.w + 8) & (py > -8) & (py < view.h + 8)
    every = max(2, round(gap / step))   # the points of a line are `step` px apart
    lines = []
    for n, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        vis = inside[a:b]
        if b - a < 2 or not vis.any():
            continue
        heads = np.arange(int((n * 0.6180339887) % 1 * every), b - a, every)
        runs = np.flatnonzero(np.diff(np.concatenate([[0], vis.view(np.int8), [0]])))
        for r0, r1 in runs.reshape(-1, 2):
            r0, r1 = max(r0 - 1, 0), min(r1 + 1, b - a)   # one point past each end reaches the edge
            if r1 - r0 >= 2:
                h = heads[(heads >= r0 + 3) & (heads < r1 - 3)] - r0
                lines.append((pts[a + r0:a + r1], m[a + r0:a + r1], h))
    return lines


def streamlines(grid, view, vref, sep, step=1.5):
    """Evenly spaced streamlines over the grid, `sep` px apart, traced in RK4 steps of `step` px.
    Returns [(points (N, 2) px, speeds (N,), arrow indices)]; lines run in the field direction."""
    unit = grid.h * view.scale   # px per grid unit
    d_sep = sep / unit
    k = max(1, round(d_sep))
    lattice = (k, (-grid.i0) % k, (-grid.j0) % k)
    x, y, m, starts, _ = kernels.evenly_spaced(grid.U, grid.V, d_sep, step / unit, 1e-6 * vref, lattice)
    return _lines_px(grid, view, x, y, m, starts, step, 12 * sep)


def trace_through(grid, view, vref, points_px, step=1.0):
    """Full streamlines through the given pixel points, in the format of streamlines()."""
    pts = np.asarray(points_px, float).reshape(-1, 2)
    gx, gy = grid.to_grid(pts[:, 0], pts[:, 1], view)
    unit = grid.h * view.scale
    x, y, m, starts, _ = kernels.trace_through(grid.U, grid.V, gx, gy, step / unit, 1e-6 * vref, d_self=2.0 / unit)
    return _lines_px(grid, view, x, y, m, starts, step, 240)


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

    def step(self, grid, view, dt):
        """Moves every particle by one RK4 step through the grid and respawns the ones that left the view
        or grew old. Returns (previous positions, speeds, respawned mask)."""
        old = self.pos.copy()
        x, y = old[:, 0].copy(), old[:, 1].copy()
        speed = np.empty(len(x))
        alive = np.empty(len(x), bool)
        kernels.advect(grid.U, grid.V, grid.x0, grid.y0, grid.h, x, y, dt, speed, alive)
        self.pos = np.stack([x, y], axis=1)
        self.age += 1
        px, py = view.to_px(x, y)
        dead = ((self.age > self.life) | ~alive | ~np.isfinite(px) | ~np.isfinite(py)
                | (px < 0) | (px >= view.w) | (py < 0) | (py >= view.h))
        n = int(dead.sum())
        if n:
            self.pos[dead] = self._spawn(n, view)
            self.age[dead] = 0
            self.life[dead] = self.rng.uniform(40, 140, n)
        return old, speed, dead
