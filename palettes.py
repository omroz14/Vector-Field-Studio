"""Colour palettes: the built-in ones and the user's own, which are kept in a JSON file between sessions."""
import json
import os
import re
import sys

import numpy as np

from theme import SURFACE, ACCENT, WARM, hex_to_rgb

MAX_STOPS = 7
USER = "user:"   # prefix of the keys of the user's palettes


def _ramp(stops, pos):
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    rgb = [hex_to_rgb(s) for s in stops]
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [s[c] for s in rgb])
    return lut


def _mix(a, b, f):
    """The colour a fraction f of the way from hex colour a to hex colour b."""
    ra, rb = hex_to_rgb(a), hex_to_rgb(b)
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(ra, rb))


class Palette:
    """A color scheme: `seq` for the magnitude (0 -> fast), `div` for signed quantities (negative -> positive),
    and `line` for vectors on the plain background: the bright part of `seq`, so slow vectors stay visible.
    The stops are kept so the palette can serve as the start of a new one."""

    def __init__(self, seq, div, seq_pos=None):
        self.seq_stops, self.div_stops = list(seq), list(div)
        self.seq = _ramp(seq, seq_pos or np.linspace(0, 1, len(seq)))
        self.div = _ramp(div, np.linspace(0, 1, len(div)))
        self.line = self.seq[np.linspace(90, 255, 256).astype(int)]

    @classmethod
    def simple(cls, seq, below, above):
        """A palette from its slow-to-fast colours and the two colours of negative and positive values; the
        diverging map runs from a light `below` through `below`, the stage colour (zero) and `above`."""
        return cls(seq, [_mix(below, "#ffffff", 0.55), below, SURFACE, above, _mix(above, "#ffffff", 0.55)])

    @property
    def below(self):
        return self.div_stops[1]

    @property
    def above(self):
        return self.div_stops[-2]


# Sequential stops run from slow to fast; diverging stops have the stage color in the middle, at zero.
PALETTES = {
    "studio": Palette([SURFACE, ACCENT, WARM, "#ffffff"],     # the motion map palette of Pixel Sort Studio
                      ["#d6ccff", ACCENT, SURFACE, WARM, "#ffe2b8"], [0.0, 0.4, 0.75, 1.0]),
    "viridis": Palette(["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
                       ["#b4c8ff", "#3b6fd8", SURFACE, "#d8503b", "#ffc4a8"]),
    "magma": Palette(["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
                     ["#a8f0e0", "#2a9d8f", SURFACE, "#de4968", "#fcc5c0"]),
    "inferno": Palette(["#000004", "#320a5e", "#781c6d", "#bc3754", "#ed6925", "#fbb61a", "#fcffa4"],
                       ["#9ecae1", "#3182bd", SURFACE, "#ed6925", "#fbb61a"]),
    "plasma": Palette(["#0d0887", "#5c01a6", "#9c179e", "#cc4778", "#ed7953", "#fdb42f", "#f0f921"],
                      ["#9fd8e8", "#2c7fb8", SURFACE, "#cc4778", "#fdb42f"]),
    "cividis": Palette(["#00224e", "#35456c", "#666970", "#948e77", "#c8b866", "#fee838"],
                       ["#a6bddb", "#35456c", SURFACE, "#c8b866", "#fee838"]),
    "ice": Palette([SURFACE, "#16425b", "#2a87b0", "#7fd3e6", "#f2fbff"],
                   ["#7fd3e6", "#2a87b0", SURFACE, "#c98a2e", "#ffe2b8"]),
    "fire": Palette([SURFACE, "#5c1a1a", "#c0392b", "#f39c12", "#fff3b0"],
                    ["#7fd3e6", "#2a87b0", SURFACE, "#c0392b", "#f39c12"]),
    "turbo": Palette(["#30123b", "#4669db", "#26bce1", "#5ffc73", "#d0e935", "#fb8022", "#7a0403"],
                     ["#26bce1", "#4669db", SURFACE, "#fb8022", "#d0e935"]),
    "aurora": Palette([SURFACE, "#123d4a", "#1f8a8a", "#57d3a0", "#b8f5c8", "#f4fff6"],
                      ["#d9c2ff", "#8b6cff", SURFACE, "#1f8a8a", "#8fe3c0"]),
    "sunset": Palette(["#1a1030", "#4a1a5c", "#9b2f6e", "#e2566a", "#f99a5b", "#ffd88a"],
                      ["#a5c8ff", "#4a6fd8", SURFACE, "#e2566a", "#ffd88a"]),
    "ocean": Palette([SURFACE, "#0c2a4a", "#10558a", "#1d93b8", "#6fd3d0", "#e6fff8"],
                     ["#ffd2a8", "#e07b39", SURFACE, "#1d93b8", "#9ee6e0"]),
    "mono": Palette([SURFACE, "#4a4f5e", "#9aa0b0", "#ffffff"],
                    ["#d3dbf5", "#6b7aa6", SURFACE, "#a6876b", "#f5e1cf"]),
}
BUILT_IN = [("studio", "Studio"), ("viridis", "Viridis"), ("magma", "Magma"), ("inferno", "Inferno"),
            ("plasma", "Plasma"), ("cividis", "Cividis"), ("ice", "Ice"), ("fire", "Fire"), ("turbo", "Turbo"),
            ("aurora", "Aurora"), ("sunset", "Sunset"), ("ocean", "Ocean"), ("mono", "Mono")]
PALETTE_NAMES = list(BUILT_IN)   # (key, label) of every palette, the user's after the built-in ones


def is_user(key):
    return key.startswith(USER)


def config_path():
    """Where the user's palettes are kept: the usual per-user settings folder of the platform."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        folder = os.path.join(base, "VectorFieldStudio")
    elif sys.platform == "darwin":
        folder = os.path.expanduser("~/Library/Application Support/VectorFieldStudio")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        folder = os.path.join(base, "vector-field-studio")
    return os.path.join(folder, "palettes.json")


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _valid(entry):
    seq = entry.get("seq")
    return (isinstance(entry.get("name"), str) and entry["name"].strip()
            and isinstance(seq, list) and 2 <= len(seq) <= MAX_STOPS
            and all(isinstance(c, str) and _HEX.match(c) for c in seq + [entry.get("below"), entry.get("above")]))


def _user_entries():
    return [dict(name=label, seq=PALETTES[key].seq_stops, below=PALETTES[key].below, above=PALETTES[key].above)
            for key, label in PALETTE_NAMES if is_user(key)]


def load_user(path=None):
    """Adds the user's saved palettes. A missing file is fine; a broken one is skipped. Returns an error
    message or None."""
    path = path or config_path()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return f"Could not read your palettes ({exc})."
    for entry in data.get("palettes", []) if isinstance(data, dict) else []:
        if isinstance(entry, dict) and _valid(entry):
            add_user(entry["name"], entry["seq"], entry["below"], entry["above"], save=False)
    return None


def save_user(path=None):
    path = path or config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"palettes": _user_entries()}, fh, indent=2)
    os.replace(tmp, path)   # never leaves a half-written file behind


def add_user(name, seq, below, above, save=True, replace=None):
    """Adds (or, with `replace` = its key, updates) a user palette and returns its key."""
    name = name.strip()
    if replace is not None:
        remove_user(replace, save=False)
    key = USER + name
    PALETTES[key] = Palette.simple(seq, below, above)
    if (key, name) not in PALETTE_NAMES:
        PALETTE_NAMES.append((key, name))
    if save:
        save_user()
    return key


def remove_user(key, save=True):
    PALETTE_NAMES[:] = [(k, lbl) for k, lbl in PALETTE_NAMES if k != key]
    PALETTES.pop(key, None)
    if save:
        save_user()
