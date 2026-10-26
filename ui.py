"""Application window: field equations, display settings, the stage and the particle animation."""
import math
import os
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import field as fieldmod
import render
from theme import *

TICK_MS = 33       # ~30 frames per second
PANEL_W = 290      # width of each side panel (px)
QUICK_MS = 220     # quiet time after an interaction before the full-quality redraw
GIF_FRAMES = 120
GIF_MAX = 560      # longest side of an exported GIF (px)
HALF_MIN, HALF_MAX = 0.25, 79.0   # zoom limits: half of the view height in world units

ARROWS, STREAMLINES, PARTICLES = "Arrows", "Streamlines", "Particles"
MODES = (ARROWS, STREAMLINES, PARTICLES)
BACKGROUNDS = (("none", "None"), ("magnitude", "|F|"), ("divergence", "div F"), ("curl", "curl F"))
BG_NOTES = {
    "none": "Plain background. Vectors are coloured by their length.",
    "magnitude": "Brightness shows the speed |F|.",
    "divergence": "Where the field spreads out (source) or converges (sink).",
    "curl": "Which way the field rotates: clockwise or counter-clockwise.",
}
BG_SCALE = {   # colorbar labels: left, middle, right
    "none": ("slow", "", "fast"),
    "magnitude": ("0", "", "fast"),
    "divergence": ("sink", "0", "source"),
    "curl": ("clockwise", "0", "counter-cw"),
}
PROBE_ROWS = ("x, y", "F", "|F|", "Angle", "div F", "curl F")


def _label(options, key):
    return dict(options)[key]


def _key(options, label):
    return next(k for k, v in options if v == label)


def _half_from(v):
    """Zoom slider (0..100, logarithmic) -> half of the view height."""
    return HALF_MIN * 10 ** (v / 40)


def _slider_from(half):
    return float(np.clip(40 * math.log10(half / HALF_MIN), 0, 100))


def _short(expr, n=34):
    return expr if len(expr) <= n else expr[:n - 1] + "…"


def _num(v):
    return f"{v:.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"


class App(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        super().__init__(fg_color=BG)
        self.title("Vector Field Studio")
        self.geometry("1400x860")
        self.minsize(1120, 680)

        self.field = None
        self.preset = "vortex"
        self.home = ((0.0, 0.0), 3.0)   # view of the current preset, for Reset view
        self.center = (0.0, 0.0)
        self.half = 3.0                 # half of the view height, world units
        self.t = 0.0
        self.mode = ARROWS
        self.bg_kind = "none"
        self.palette = "studio"
        self.colored = tk.BooleanVar(value=True)
        self.arrow_spacing = 36
        self.arrow_len = 100            # % of the arrow cell
        self.equal_len = tk.BooleanVar(value=False)
        self.overlay = tk.BooleanVar(value=False)    # arrows in every view
        self.stream_sep = 22
        self.stream_heads = tk.BooleanVar(value=True)
        self.n_particles = 2000
        self.p_speed = 3                # px per frame at the typical speed
        self.trail_len = 20             # frames
        self.pins = []                  # world points with a traced streamline

        self.parts = fieldmod.Particles()
        self.trail = None               # particle trails, float32 (h, w, 3)
        self._static = None             # background + vectors, uint8 (h, w, 3)
        self._frame = None              # what the stage shows now
        self._photo = None
        self._quick = False
        self.running = False
        self._after_id = None
        self._redraw_id = None
        self._full_id = None
        self._press = None
        self._dragging = False

        self._build_ui()
        self._bind_keys()
        self.set_preset(self.preset)
        self.set_mode(ARROWS)
        self._status("Pick a preset or type your own Fx and Fy. Space animates the flow.")

    # ---------- building the interface ----------

    def _make_panel(self, column):
        """A scrollable side panel in grid `column`. Returns the frame to put cards in."""
        panel = ctk.CTkScrollableFrame(self, width=PANEL_W, fg_color=BG, corner_radius=0,
                                       scrollbar_button_color=BG, scrollbar_button_hover_color=BORDER)
        panel.grid(row=0, column=column, sticky="ns", padx=(12, 0) if column == 0 else (0, 12), pady=12)
        return panel

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        left = self._make_panel(0)
        right = self._make_panel(2)
        self._build_left(left)
        self._build_right(right)
        self._build_main()

    def _build_left(self, left):
        """What to show: the field and the region of the plane."""
        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill="x", pady=(4, 14), padx=4)
        ctk.CTkLabel(head, text="Vector Field Studio", font=FONT_TITLE, text_color=TEXT, height=26).pack(anchor="w")
        ctk.CTkLabel(head, text="explore 2D vector fields", font=FONT_SMALL,
                     text_color=MUTED, height=16).pack(anchor="w")

        # PRESETS
        c = card(left, "PRESETS")
        self.preset_grid = ChoiceGrid(c, [(k, lbl) for k, lbl, *_ in fieldmod.PRESETS],
                                      self.preset, self.set_preset, cols=2)
        ctk.CTkFrame(c, fg_color="transparent", height=PAD - 4).pack()

        # EQUATIONS
        c = card(left, "EQUATIONS", "Enter applies")
        self.ent = {}
        for name in ("Fx", "Fy"):
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=PAD, pady=(0, 6))
            ctk.CTkLabel(row, text=f"{name} =", font=FONT_BOLD, text_color=TEXT, width=38,
                         anchor="w").pack(side="left")
            e = entry(row)
            e.pack(side="left", fill="x", expand=True)
            e.bind("<Return>", lambda ev: self.apply_equations())
            e.bind("<Escape>", lambda ev: self.stage.focus_set())
            self.ent[name] = e
        button(c, "Apply", self.apply_equations, accent=True).pack(fill="x", padx=PAD, pady=(2, 0))
        self.lbl_err = note(c, "", pady=(6, 0))
        self.lbl_err.configure(text_color=ERROR)
        note(c, "Variables: x, y, t, r, theta. Functions: sin cos tan exp log sqrt abs atan2 "
                "sinh cosh tanh min max … Constants: pi, e. ^ means power.", pady=(4, PAD))

        # VIEW
        c = card(left, "VIEW", "key R")
        self.row_zoom = SliderRow(c, "Zoom", 0, 100, round(_slider_from(self.half)), self._on_zoom,
                                  fmt=lambda v: f"±{_half_from(v):.3g}")
        vb = ctk.CTkFrame(c, fg_color="transparent")
        vb.pack(fill="x", padx=PAD, pady=(6, 0))
        button(vb, "Reset view", self.reset_view, width=10).pack(side="left", fill="x", expand=True)
        button(vb, "Clear traces", self.clear_pins, width=10).pack(side="left", fill="x", expand=True, padx=(6, 0))
        note(c, "Scroll to zoom, drag to pan. Click to trace the streamline through a point; "
                "right-click clears the traces.", pady=(8, PAD))

    def _build_right(self, right):
        """How to show it: colors, arrows, streamlines, particles, the probe and export."""
        # COLORS
        c = card(right, "COLOURS", "keys B, P")
        self.bg_tabs = segmented(c, [lbl for _, lbl in BACKGROUNDS],
                                 lambda v: self.set_background(_key(BACKGROUNDS, v)))
        self.bg_tabs.set(_label(BACKGROUNDS, self.bg_kind))
        self.bg_tabs.pack(fill="x", padx=PAD, pady=(0, 8))
        ctk.CTkLabel(c, text="Palette", font=FONT, text_color=TEXT, height=20, anchor="w").pack(
            fill="x", padx=PAD, pady=(0, 2))
        self.palette_grid = ChoiceGrid(c, render.PALETTE_NAMES, self.palette, self.set_palette, cols=3)
        ctk.CTkFrame(c, fg_color="transparent", height=8).pack()
        self.colorbar = tk.Canvas(c, height=10, bg=SURFACE, highlightthickness=0)
        self.colorbar.pack(fill="x", padx=PAD)
        self.colorbar.bind("<Configure>", lambda e: self._draw_colorbar())
        scale = ctk.CTkFrame(c, fg_color="transparent")
        scale.pack(fill="x", padx=PAD)
        self.lbl_scale = [ctk.CTkLabel(scale, text="", font=FONT_SMALL, text_color=MUTED, height=16)
                          for _ in range(3)]
        self.lbl_scale[0].pack(side="left")
        self.lbl_scale[2].pack(side="right")
        self.lbl_scale[1].place(relx=0.5, rely=0, anchor="n")
        self.lbl_bg = note(c, BG_NOTES[self.bg_kind], pady=(4, 8))
        switch(c, "Colour vectors by speed", self.colored, self._invalidate).pack(anchor="w", padx=PAD)
        note(c, "On a background map, vectors are drawn in white.", pady=(4, PAD))

        # ARROWS
        c = card(right, "ARROWS")
        self.row_spacing = SliderRow(c, "Spacing", 16, 90, self.arrow_spacing, self._on_arrow_spacing,
                                     fmt=lambda v: f"{v} px")
        self.row_len = SliderRow(c, "Length", 20, 200, self.arrow_len, self._on_arrow_len, fmt=lambda v: f"{v}%")
        switch(c, "Equal length", self.equal_len, self._invalidate).pack(anchor="w", padx=PAD, pady=(8, 4))
        switch(c, "Show arrows in every view", self.overlay, self._invalidate).pack(
            anchor="w", padx=PAD, pady=(4, PAD))

        # STREAMLINES
        c = card(right, "STREAMLINES")
        self.row_sep = SliderRow(c, "Spacing", 10, 60, self.stream_sep, self._on_stream_sep,
                                 fmt=lambda v: f"{v} px")
        switch(c, "Direction arrows", self.stream_heads, self._invalidate).pack(anchor="w", padx=PAD, pady=(8, PAD))

        # PARTICLES
        c = card(right, "PARTICLES", "Space")
        self.row_count = SliderRow(c, "Count", 100, 6000, self.n_particles, self._on_count)
        self.row_speed = SliderRow(c, "Speed", 1, 12, self.p_speed, self._on_speed, fmt=lambda v: f"{v}×")
        self.row_trail = SliderRow(c, "Trail", 1, 60, self.trail_len, self._on_trail,
                                   fmt=lambda v: "none" if v <= 1 else f"{v} frames")
        ctk.CTkFrame(c, fg_color="transparent", height=PAD - 4).pack()

        # PROBE
        c = card(right, "PROBE", "hover the stage")
        self.probe = {}
        for name in PROBE_ROWS:
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=PAD)
            ctk.CTkLabel(row, text=name, font=FONT, text_color=MUTED, height=20).pack(side="left")
            self.probe[name] = ctk.CTkLabel(row, text="—", font=FONT_BOLD, text_color=TEXT, height=20)
            self.probe[name].pack(side="right")
        ctk.CTkFrame(c, fg_color="transparent", height=PAD - 4).pack()

        # EXPORT
        c = card(right, "EXPORT")
        ex = ctk.CTkFrame(c, fg_color="transparent")
        ex.pack(fill="x", padx=PAD, pady=(0, PAD))
        button(ex, "Save PNG", self.save_png, width=10).pack(side="left", fill="x", expand=True)
        button(ex, "Save GIF", self.save_gif, width=10).pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _build_main(self):
        """The center: view tabs, the stage and the play bar."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        self.tabs = segmented(main, MODES, self.set_mode)
        self.tabs.grid(row=0, column=0, sticky="w", pady=(4, 10))
        ctk.CTkLabel(main, text="Space play · 1–3 views · B background · P palette · R reset view · C clear traces",
                     font=FONT_SMALL, text_color=MUTED).grid(row=0, column=0, sticky="e", pady=(4, 10))

        frame = ctk.CTkFrame(main, fg_color=SURFACE, corner_radius=12)
        frame.grid(row=1, column=0, sticky="nsew")
        self.stage = tk.Canvas(frame, bg=SURFACE, highlightthickness=0)
        self.stage.pack(fill="both", expand=True, padx=8, pady=8)
        self._bind_stage()

        bar = ctk.CTkFrame(main, fg_color=SURFACE, corner_radius=12)
        bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bar.grid_columnconfigure(1, weight=1)
        self.btn_play = button(bar, "▶  Play", self.toggle_play, accent=True, width=130, height=38)
        self.btn_play.grid(row=0, column=0, rowspan=2, padx=PAD, pady=PAD - 2)
        self.lbl_t = ctk.CTkLabel(bar, text="t = 0.00", font=FONT_BOLD, text_color=TEXT, anchor="w", height=18)
        self.lbl_t.grid(row=0, column=1, sticky="ew", padx=(0, PAD), pady=(PAD - 2, 0))
        self.lbl_status = ctk.CTkLabel(bar, text="", font=FONT_SMALL, text_color=MUTED, anchor="w", height=18)
        self.lbl_status.grid(row=1, column=1, sticky="ew", padx=(0, PAD), pady=(2, PAD - 4))
        button(bar, "↺  Reset time", self.reset_time, width=120).grid(row=0, column=2, rowspan=2, padx=(0, PAD))

    def _bind_stage(self):
        s = self.stage
        s.bind("<Configure>", lambda e: self._view_changed())
        s.bind("<ButtonPress-1>", self._on_press)
        s.bind("<B1-Motion>", self._on_drag)
        s.bind("<ButtonRelease-1>", self._on_release)
        for b in ("2", "3"):   # right button: 2 on macOS, 3 on Windows and Linux
            s.bind(f"<ButtonPress-{b}>", lambda e: self.clear_pins())
        s.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15))   # wheel on Linux
        s.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        s.bind("<MouseWheel>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15 if e.delta > 0 else 1.15))
        s.bind("<Motion>", lambda e: self._probe(e.x, e.y))
        s.bind("<Leave>", lambda e: self._probe(None, None))

    @staticmethod
    def _unless_typing(fn):
        """Wraps a shortcut so it does nothing while an equation field has the focus."""
        return lambda e: None if isinstance(e.widget, tk.Entry) else fn()

    def _bind_keys(self):
        keys = {"<space>": self.toggle_play, "<r>": self.reset_view, "<c>": self.clear_pins,
                "<b>": self._cycle_background, "<p>": self._cycle_palette, "<t>": self.reset_time}
        for seq, fn in keys.items():
            self.bind(seq, self._unless_typing(fn))
        self.bind("<Control-s>", lambda e: self.save_png())
        for i, m in enumerate(MODES, 1):
            self.bind(f"<Key-{i}>", self._unless_typing(lambda m=m: self.set_mode(m)))

    # ---------- field and settings ----------

    def set_preset(self, key):
        _, _, fx, fy, center, half = next(p for p in fieldmod.PRESETS if p[0] == key)
        for name, expr in (("Fx", fx), ("Fy", fy)):
            self.ent[name].delete(0, "end")
            self.ent[name].insert(0, expr)
        self.home = (tuple(map(float, center)), float(half))
        self.center, self.half = self.home
        self._sync_zoom()
        self.t = 0.0
        self.pins = []
        self.apply_equations()

    def apply_equations(self):
        try:
            f = fieldmod.Field(self.ent["Fx"].get(), self.ent["Fy"].get())
        except ValueError as exc:
            self.lbl_err.configure(text=str(exc))
            return
        self.lbl_err.configure(text="")
        self.field = f
        self.preset = next((p[0] for p in fieldmod.PRESETS
                            if (p[2], p[3]) == (f.fx, f.fy)), None)
        self.preset_grid.select(self.preset)
        self.trail = None
        self._invalidate()
        self._status("Field applied." + (" It depends on t: press Space to let time run." if f.uses_t else ""))

    def set_mode(self, mode):
        self.mode = mode
        self.tabs.set(mode)
        self._invalidate()

    def set_background(self, kind):
        self.bg_kind = kind
        self.bg_tabs.set(_label(BACKGROUNDS, kind))
        self.lbl_bg.configure(text=BG_NOTES[kind])
        self._draw_colorbar()
        self._invalidate()

    def _cycle_background(self):
        keys = [k for k, _ in BACKGROUNDS]
        self.set_background(keys[(keys.index(self.bg_kind) + 1) % len(keys)])

    def set_palette(self, key):
        self.palette = key
        self.palette_grid.select(key)
        self._draw_colorbar()
        if self.trail is not None:
            self.trail[:] = 0             # old trails keep the colors of the previous palette
        self._invalidate()

    def _cycle_palette(self):
        keys = [k for k, _ in render.PALETTE_NAMES]
        self.set_palette(keys[(keys.index(self.palette) + 1) % len(keys)])

    def _draw_colorbar(self):
        c = self.colorbar
        c.delete("all")
        w = max(c.winfo_width(), 10)
        pal = render.PALETTES[self.palette]
        if self.bg_kind == "none":
            lut = pal.line
        elif self.bg_kind == "magnitude":
            lut = pal.seq[:render.MAG_TOP + 1]
        else:
            lut = pal.div
        for x in range(0, w, 2):
            r, g, b = lut[int(x / w * (len(lut) - 1))]
            c.create_rectangle(x, 0, x + 2, 10, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
        for lbl, text in zip(self.lbl_scale, BG_SCALE[self.bg_kind]):
            lbl.configure(text=text)

    def _on_zoom(self, v):
        self.half = _half_from(v)
        self._view_changed()

    def _sync_zoom(self):
        """Moves the zoom slider to the current half-height and shows the exact value."""
        self.row_zoom.set(round(_slider_from(self.half)))
        self.row_zoom.value.configure(text=f"±{self.half:.3g}")

    def _on_arrow_spacing(self, v):
        self.arrow_spacing = v
        self._invalidate(quick=True)

    def _on_arrow_len(self, v):
        self.arrow_len = v
        self._invalidate(quick=True)

    def _on_stream_sep(self, v):
        self.stream_sep = v
        self._invalidate(quick=True)

    def _on_count(self, v):
        self.n_particles = v
        self.trail = None
        self._invalidate(static=False)

    def _on_speed(self, v):
        self.p_speed = v

    def _on_trail(self, v):
        self.trail_len = v

    # ---------- view: pan, zoom, traces, probe ----------

    def _view(self):
        w, h = self._stage_size()
        return fieldmod.View(self.center[0], self.center[1], self.half, w, h)

    def _stage_size(self):
        return max(self.stage.winfo_width(), 50), max(self.stage.winfo_height(), 50)

    def _view_changed(self):
        self._invalidate(quick=True)
        if self.trail is not None:
            self.trail[:] = 0             # trails are in screen space; particles stay in the world

    def reset_view(self):
        self.center, self.half = self.home
        self._sync_zoom()
        self._view_changed()

    def _zoom_at(self, px, py, factor):
        """Zooms keeping the world point under the cursor in place."""
        x, y = self._view().to_world(px, py)
        new = float(np.clip(self.half * factor, HALF_MIN, HALF_MAX))
        f = new / self.half
        self.center = (x + (self.center[0] - x) * f, y + (self.center[1] - y) * f)
        self.half = new
        self._sync_zoom()
        self._view_changed()

    def _on_press(self, e):
        self.stage.focus_set()            # takes the focus away from the equation fields
        self._press = (e.x, e.y, self.center)
        self._dragging = False

    def _on_drag(self, e):
        if self._press is None:
            return
        x0, y0, c = self._press
        if not self._dragging and abs(e.x - x0) + abs(e.y - y0) < 5:
            return
        self._dragging = True
        s = self._view().scale
        self.center = (c[0] - (e.x - x0) / s, c[1] + (e.y - y0) / s)
        self._view_changed()

    def _on_release(self, e):
        if self._press is not None and not self._dragging:
            x, y = self._view().to_world(e.x, e.y)
            self.pins.append((float(x), float(y)))
            self._invalidate()
            self._status(f"Traced the streamline through ({_num(x)}, {_num(y)}). Right-click clears traces.")
        self._press = None
        self._dragging = False

    def clear_pins(self):
        if self.pins:
            self.pins = []
            self._invalidate()
            self._status("Traces cleared.")

    def _probe(self, px, py):
        if px is None or self.field is None:
            for lbl in self.probe.values():
                lbl.configure(text="—")
            return
        x, y = self._view().to_world(px, py)
        h = 1e-4 * self.half
        u, v = self.field(np.array([x, x + h, x - h, x, x]), np.array([y, y, y, y + h, y - h]), self.t)
        div = (u[1] - u[2] + v[3] - v[4]) / (2 * h)
        curl = (v[1] - v[2] - u[3] + u[4]) / (2 * h)
        vals = {
            "x, y": f"{_num(x)}, {_num(y)}",
            "F": f"({_num(u[0])}, {_num(v[0])})",
            "|F|": _num(math.hypot(u[0], v[0])),
            "Angle": f"{math.degrees(math.atan2(v[0], u[0])) % 360:.0f}°" if u[0] or v[0] else "—",
            "div F": _num(div),
            "curl F": _num(curl),
        }
        for name, text in vals.items():
            self.probe[name].configure(text=text)

    # ---------- rendering ----------

    def _invalidate(self, quick=False, static=True):
        """Schedules a redraw. `quick` draws at lower quality now and at full quality once things settle."""
        if static:
            self._static = None
        if quick:
            self._quick = True
            if self._full_id is not None:
                self.after_cancel(self._full_id)
            self._full_id = self.after(QUICK_MS, self._full_redraw)
        if self._redraw_id is None:
            self._redraw_id = self.after_idle(self._redraw)

    def _full_redraw(self):
        self._full_id = None
        self._quick = False
        self._static = None
        self._redraw()

    def _redraw(self):
        self._redraw_id = None
        if not self.running and self.field is not None:   # while playing, every tick redraws
            self._show(self._view(), self._quick)

    def _build_static(self, view, quick):
        t = self.t
        try:
            vref = fieldmod.reference_speed(self.field, view, t)
            lines = None
            if self.mode == STREAMLINES:
                lines = fieldmod.streamlines(self.field, view, t, vref, self.stream_sep, step=6 if quick else 3)
            traces = ()
            if self.pins:
                pts = np.array([view.to_px(x, y) for x, y in self.pins])
                traces = fieldmod.trace_through(self.field, view, t, vref, pts, step=4 if quick else 2)
            arrows = None
            if self.mode == ARROWS or self.overlay.get():
                arrows = dict(spacing=self.arrow_spacing, length=self.arrow_len / 100, equal=self.equal_len.get())
            self._static = render.compose(self.field, view, t, vref, self.bg_kind, arrows, lines, traces,
                                          colored=self.colored.get(), heads=self.stream_heads.get(),
                                          palette=self.palette,
                                          ss=1 if quick else 2)
        except ValueError as exc:
            self._status(f"Could not evaluate the field: {exc}")
            self._static = np.full((view.h, view.w, 3), hex_to_rgb(SURFACE), np.uint8)

    def _compose(self, view, quick=False):
        """The frame to show: the static layer, plus particle trails in the Particles view."""
        if self._static is None or self._static.shape[:2] != (view.h, view.w):
            self._build_static(view, quick)
        if self.mode != PARTICLES:
            return self._static
        if self.trail is None or self.trail.shape[:2] != (view.h, view.w):
            self._reset_particles(view)
        return np.maximum(self._static, self.trail.astype(np.uint8))

    def _show(self, view, quick=False):
        self._frame = self._compose(view, quick)
        self._photo = ImageTk.PhotoImage(Image.fromarray(self._frame))
        s = self.stage
        s.delete("all")
        s.create_image(0, 0, image=self._photo, anchor="nw")
        self._draw_ticks(view)
        s.create_text(10, 6, anchor="nw", fill=MUTED, font=FONT_SECTION, text=self._caption())

    def _draw_ticks(self, view):
        """Coordinate labels along the bottom and left edges."""
        step, xs, ys = render.ticks(view)
        dec = max(0, -math.floor(math.log10(step)))
        for x in xs:
            px = view.to_px(x, 0)[0]
            if 20 < px < view.w - 20:
                self.stage.create_text(px, view.h - 4, anchor="s", fill=MUTED, font=FONT_SMALL, text=f"{x:.{dec}f}")
        for y in ys:
            py = view.to_px(0, y)[1]
            if 30 < py < view.h - 24:
                self.stage.create_text(6, py, anchor="w", fill=MUTED, font=FONT_SMALL, text=f"{y:.{dec}f}")

    def _caption(self):
        f = self.field
        text = f"{self.mode.upper()} · F = ({_short(f.fx)}, {_short(f.fy)})"
        if self.bg_kind != "none":
            text += f" · {_label(BACKGROUNDS, self.bg_kind)}"
        if f.uses_t:
            text += f" · t = {self.t:.2f}"
        return text

    # ---------- particles and time ----------

    def _reset_particles(self, view):
        self.parts.reset(self.n_particles, view)
        self.trail = np.zeros((view.h, view.w, 3), np.float32)
        x, y = self.parts.pos.T
        u, v = self.field(x, y, self.t)
        vref = fieldmod.reference_speed(self.field, view, self.t)
        px, py = view.to_px(x, y)
        self._dot(px.astype(int), py.astype(int), self._particle_colors(np.hypot(u, v) / vref))

    def _particle_colors(self, rel):
        return render.vector_colors(rel, self.colored.get() and self.bg_kind == "none",
                                    self.palette).astype(np.float32)

    def _dot(self, px, py, col):
        """Paints 2×2 dots into the trail buffer."""
        h, w = self.trail.shape[:2]
        ok = (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
        px, py, col = px[ok], py[ok], col[ok]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            self.trail[py + dy, px + dx] = np.maximum(self.trail[py + dy, px + dx], col)

    def _step(self, view):
        """Advances time by one frame, and the particles with it in the Particles view."""
        vref = fieldmod.reference_speed(self.field, view, self.t)
        dt = self.p_speed / (vref * view.scale)    # a particle at the typical speed moves p_speed px
        if self.mode == PARTICLES:
            if self.trail is None or self.trail.shape[:2] != (view.h, view.w):
                self._reset_particles(view)
            self.trail *= 1 - 1 / self.trail_len
            old, speed, dead = self.parts.step(self.field, view, self.t, dt)
            keep = ~dead
            ox, oy = view.to_px(*old[keep].T)
            nx, ny = view.to_px(*self.parts.pos[keep].T)
            col = self._particle_colors(speed[keep] / vref)
            # Fill the gap between the old and new position, so fast particles leave lines, not dots.
            dist = np.hypot(nx - ox, ny - oy)
            nsub = int(np.clip(np.ceil(dist.max(initial=0) / 1.2), 1, 12))
            for k in range(1, nsub + 1):
                a = k / nsub
                self._dot((ox + a * (nx - ox)).astype(int), (oy + a * (ny - oy)).astype(int), col)
        self.t += dt
        if self.field.uses_t:
            self._static = None

    def toggle_play(self):
        if self.running:
            self._stop()
            self._status("Paused.")
            return
        if self.mode != PARTICLES and not self.field.uses_t:
            self.set_mode(PARTICLES)      # a steady field only moves through its particles
        self.running = True
        self.btn_play.configure(text="❚❚  Pause")
        self._status("Playing. Particles follow the field; their colour shows the speed.")
        self._tick()

    def _stop(self):
        self.running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.btn_play.configure(text="▶  Play")

    def _tick(self):
        self._after_id = None
        if not self.running:
            return
        view = self._view()
        self._step(view)
        self._show(view, quick=self._quick or self.field.uses_t)
        self.lbl_t.configure(text=f"t = {self.t:.2f}")
        self._after_id = self.after(TICK_MS, self._tick)

    def reset_time(self):
        self.t = 0.0
        self.trail = None
        self.lbl_t.configure(text="t = 0.00")
        self._invalidate()

    # ---------- files ----------

    def save_png(self):
        if self._frame is None:
            return
        path = filedialog.asksaveasfilename(title="Save image", defaultextension=".png",
                                            initialfile="vector_field.png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        Image.fromarray(self._frame).save(path)
        self._status(f"Saved: {os.path.basename(path)}")

    def save_gif(self):
        if self.field is None:
            return
        path = filedialog.asksaveasfilename(title="Save animation", defaultextension=".gif",
                                            initialfile="vector_field.gif", filetypes=[("GIF", "*.gif")])
        if not path:
            return
        self._stop()
        if self.mode != PARTICLES and not self.field.uses_t:
            self.set_mode(PARTICLES)
        view = self._view()
        frames = []
        for i in range(GIF_FRAMES):
            self._step(view)
            img = Image.fromarray(self._compose(view, quick=self.field.uses_t))
            img.thumbnail((GIF_MAX, GIF_MAX), Image.LANCZOS)
            frames.append(img)
            if i % 10 == 0:
                self._status(f"Recording frame {i + 1} / {GIF_FRAMES}…")
                self.update_idletasks()
        # One shared palette and no dithering keep the file small: noisy dithered trails compress badly.
        pal = frames[-1].quantize(colors=64, dither=Image.Dither.NONE)
        frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=TICK_MS, loop=0)
        self.lbl_t.configure(text=f"t = {self.t:.2f}")
        self._invalidate()
        self._status(f"Saved animation: {os.path.basename(path)}")

    def _status(self, text):
        self.lbl_status.configure(text=text)
