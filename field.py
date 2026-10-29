"""Vector fields: presets, safe expression parsing, the view transform, the sampled grid and streamlines."""
import ast
import math
from fractions import Fraction

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
    """Compiles one component. Only the variables and FUNCTIONS may appear: no attributes, no indexing.
    Returns (code, syntax tree)."""
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
    return compile(tree, name, "eval"), tree


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _constant(node):
    """The value of a subtree made of numbers, pi, e and functions only (no variables), else None."""
    if not _names(node) <= set(FUNCTIONS):
        return None
    try:
        v = float(eval(compile(ast.Expression(node), "c", "eval"), _GLOBALS))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _omega(node):
    """ω if the subtree is ω·t + c(x, y) with a constant ω (0 when it does not involve t), else None."""
    if "t" not in _names(node):
        return 0.0
    if isinstance(node, ast.Name):
        return 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        w = _omega(node.operand)
        return None if w is None else (-w if isinstance(node.op, ast.USub) else w)
    if isinstance(node, ast.BinOp):
        a, b = _omega(node.left), _omega(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return a + b if isinstance(node.op, ast.Add) else a - b
        if isinstance(node.op, ast.Mult):
            if a == 0 and (k := _constant(node.left)) is not None:
                return k * b
            if b == 0 and (k := _constant(node.right)) is not None:
                return a * k
        if isinstance(node.op, ast.Div) and b == 0 and (k := _constant(node.right)):
            return a / k
    return None


def _period(trees):
    """The period in t of a field whose expressions use t only as sin or cos(ω·t + c), the common period
    when the ω are rational multiples of each other; None if there is none (or t is not used)."""
    omegas = []

    def visit(node):
        if (isinstance(node, ast.Call) and node.func.id in ("sin", "cos") and len(node.args) == 1
                and (w := _omega(node.args[0])) is not None):
            if w:
                omegas.append(abs(w))
            return True
        if isinstance(node, ast.Name) and node.id == "t":
            return False
        return all(visit(c) for c in ast.iter_child_nodes(node))

    if not all(visit(t) for t in trees) or not omegas:
        return None
    base = omegas[0]
    ratios = []
    for w in omegas:
        r = Fraction(w / base).limit_denominator(64)
        if abs(float(r) - w / base) > 1e-9 * (w / base):
            return None   # incommensurate frequencies: never repeats exactly
        ratios.append(r)
    num = math.gcd(*(r.numerator for r in ratios))
    den = math.lcm(*(r.denominator for r in ratios))
    return 2 * math.pi / (base * num / den)


_GLOBALS = {"__builtins__": {}, **FUNCTIONS}


class Field:
    """F(x, y, t) = (Fx, Fy) given as two expressions. Raises ValueError on a bad expression."""

    def __init__(self, fx, fy):
        self.fx, self.fy = fx.strip(), fy.strip()
        (cx, tx), (cy, ty) = _compile(fx, "Fx"), _compile(fy, "Fy")
        self._code = (cx, cy)
        names = set().union(*(c.co_names for c in self._code))
        self.uses_t = "t" in names
        self.period = _period((tx, ty)) if self.uses_t else None   # in t, for seamless loops
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


def _whole(v):
    """v, or the whole number it is up to float noise."""
    r = round(v)
    return float(r) if abs(v - r) < 1e-6 else v


class View:
    """Maps world coordinates to stage pixels. y grows upward in the world and downward on screen.

    At a given zoom the world is tiled by a lattice of pixels: x·scale and −y·scale are lattice coordinates,
    and the view starts at origin(). A view whose origin is a whole lattice pixel (see viewport.snap) maps
    lattice-anchored points to the same pixels, to the last bit, as any other such view."""

    def __init__(self, cx, cy, half_h, w, h, scale=None):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.scale = scale or h / (2 * half_h)   # pixels per world unit
        self.ox, self.oy = _whole(cx * self.scale - w / 2), _whole(-cy * self.scale - h / 2)

    @classmethod
    def at(cls, cx, cy, scale, w, h):
        """A view at exactly this scale (a half-height would round it)."""
        return cls(cx, cy, None, w, h, scale)

    def origin(self):
        """The lattice coordinates of the view's top-left corner."""
        return self.ox, self.oy

    def to_px(self, x, y):
        return x * self.scale - self.ox, -y * self.scale - self.oy

    def to_world(self, px, py):
        return (px + self.ox) / self.scale, -(py + self.oy) / self.scale

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
        self.spacing = spacing   # px of `view` between nodes
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

    def unit(self, view):
        """Pixels of `view` per grid unit (exactly `spacing` at the grid's own zoom)."""
        s = self.h * view.scale
        return self.spacing if abs(s - self.spacing) < 1e-9 * s else s

    def to_px(self, gx, gy, view):
        """Grid positions -> pixels of `view`. Node (i, j) sits at lattice pixel ((i0 + i)·s, −(j0 + j)·s)."""
        s = self.unit(view)
        return (self.i0 + gx) * s - view.ox, -(self.j0 + gy) * s - view.oy

    def to_grid(self, px, py, view):
        x, y = view.to_world(px, py)
        return (x - self.x0) / self.h, (y - self.y0) / self.h


def _lines_px(grid, view, x, y, m, starts, step, gap):
    """Kernel output -> the parts of the lines that lie in the view, as [(points (N, 2) px, speeds (N,),
    indices of the points that get a direction arrow, index of the first point in its whole line)].
    Arrows are `gap` px apart along every line, each line starting at its own (golden-ratio) phase,
    so they scatter evenly instead of lining up across neighbouring lines."""
    px, py = grid.to_px(x, y, view)
    pts = np.stack([px, py], axis=1)
    mg = 6 * step   # beyond what downscaling and the drawing margin of render.compose can reach
    inside = (px > -mg) & (px < view.w + mg) & (py > -mg) & (py < view.h + mg)
    every = max(2, round(gap / step))   # the points of a line are `step` px apart
    lines = []
    for n, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        vis = inside[a:b]
        if b - a < 2 or not vis.any():
            continue
        heads = np.arange(int((n * 0.6180339887) % 1 * every), b - a, every)
        runs = np.flatnonzero(np.diff(np.concatenate([[0], vis.view(np.int8), [0]])))
        for r0, r1 in runs.reshape(-1, 2):
            # Past each end to a whole piece of 8 points (render.draw_lines colours by pieces), so a line
            # cut by a neighbouring view is drawn the same where both show it.
            r0, r1 = max((r0 - 1) // 8 * 8, 0), min((r1 + 8) // 8 * 8 + 1, b - a)
            if r1 - r0 >= 2:
                h = heads[(heads >= r0 + 3) & (heads < r1 - 3)] - r0
                lines.append((pts[a + r0:a + r1], m[a + r0:a + r1], h, r0))
    return lines


def layout_streamlines(grid, vref, sep, step=1.5):
    """Evenly spaced streamlines over the whole grid, `sep` px apart (px of the grid's view), traced in RK4
    steps of `step` px. Returns the kernel output (grid units), which lines_in() turns into pixels."""
    unit = grid.spacing   # px per grid unit; sep / unit is the same at any px_per_pt
    d_sep = sep / unit
    k = max(1, round(d_sep))
    lattice = (k, (-grid.i0) % k, (-grid.j0) % k)
    x, y, m, starts, _ = kernels.evenly_spaced(grid.U, grid.V, d_sep, step / unit, 1e-6 * vref, lattice)
    return x, y, m, starts


def lines_in(grid, view, layout, step, sep):
    """The parts of a layout_streamlines() layout that `view` shows, in its pixels; step and sep in its px."""
    return _lines_px(grid, view, *layout, step, 12 * sep)


def streamlines(grid, view, vref, sep, step=1.5):
    """Evenly spaced streamlines over the grid, `sep` px apart, traced in RK4 steps of `step` px.
    Returns [(points (N, 2) px, speeds (N,), arrow indices)]; lines run in the field direction."""
    return lines_in(grid, view, layout_streamlines(grid, vref, sep, step), step, sep)


def trace_through(grid, view, vref, points, step=1.0):
    """Full streamlines through the given world points, in the format of streamlines() (pixels of `view`;
    `step` in them). Seeded in grid units straight from the world, so every view traces the same lines."""
    pts = np.asarray(points, float).reshape(-1, 2)
    gx, gy = (pts[:, 0] - grid.x0) / grid.h, (pts[:, 1] - grid.y0) / grid.h
    unit = grid.unit(view)
    x, y, m, starts, _ = kernels.trace_through(grid.U, grid.V, gx, gy, step / unit, 1e-6 * vref,
                                               d_self=2 * step / unit)
    return _lines_px(grid, view, x, y, m, starts, step, 240 * step)
