"""Application window: field equations, display settings, the stage and the particle animation."""
import math
import os
import time
import tkinter as tk
import traceback
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import export
import field as fieldmod
import particles
import render
import worker
from theme import *

FPS = 30           # frames per second of the animation (and of the GIF)
PANEL_W = 290      # width of each side panel (px)
QUICK_MS = 220     # quiet time after an interaction before the full-quality redraw
POLL_MS = 8        # how often the Tk thread looks for finished renders
EXPORT_POLL_MS = 100
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


class Static:
    """A finished static layer: the image for `view`, and the sampled field grid (None if nothing needed it)."""

    def __init__(self, gen, view, img=None, grid=None, error=None):
        self.gen, self.view, self.img, self.grid, self.error = gen, view, img, grid, error


def _label(options, key):
    return dict(options)[key]


def _key(options, label):
    return next(k for k, v in options if v == label)


def _half_from(v):
    """Zoom slider (0..100, logarithmic) -> half of the view height."""
    return HALF_MIN * 10 ** (v / 40)


def _slider_from(half):
    return float(np.clip(40 * math.log10(half / HALF_MIN), 0, 100))


def _rate_from(v):
    """Time-rate slider (0..100, logarithmic) -> units of t per second, 0.05 to 5."""
    return 0.05 * 10 ** (v / 50)


def _rate_slider(rate):
    return float(np.clip(50 * math.log10(rate / 0.05), 0, 100))


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
        self.density = 25               # particles per 100 × 100 points, on the stage and in exports
        self.p_speed = 3                # × 30 points per second at the typical speed (steady fields)
        self.trail_s = 0.65             # seconds: how long trails take to fade
        self.t_rate = 0.5               # units of t per second (fields that use t)
        self.pins = []                  # world points with a traced streamline

        self.flow = None                # particles and their trails (particles.Flow)
        self.ex_size = "4:5"            # export: key of export.SIZES
        self.ex_format = "mp4"
        self.ex_fps = 30
        self.ex_seconds = 10
        self.ex_ss = 2                  # supersampling of the static layer
        self.ex_grid = tk.BooleanVar(value=False)
        self._export = None             # the running export.Export
        self._worker = worker.Worker()  # builds static layers off the Tk thread
        self._gen = 0                   # number of the newest static layer asked for
        self._static = None             # newest finished Static: background + vectors, maybe for an older view
        self._fitted = None             # (Static, view key, its image fitted to that view)
        self._poll_id = None
        self._compiling = False         # the first run compiles the numba kernels: a note says so
        self._frame = None              # what the stage shows now
        self._photo = None
        self.running = False
        self._after_id = None
        self._deadline = 0.0            # when the next animation frame is due (perf_counter seconds)
        self._redraw_id = None
        self._full_id = None
        self._press = None
        self._dragging = False

        self._build_ui()
        self._bind_keys()
        self.set_preset(self.preset)
        self.set_mode(ARROWS)
        self._status("Pick a preset or type your own Fx and Fy. Space animates the flow.")
        self.after(800, self._compiling_note)

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
        self.row_density = SliderRow(c, "Density", 2, 80, self.density, self._on_density,
                                     fmt=lambda v: f"{v} per 100² pt")
        self.row_speed = SliderRow(c, "Speed", 1, 12, self.p_speed, self._on_speed, fmt=lambda v: f"{v}×")
        self.row_trail = SliderRow(c, "Trail", 0, 300, round(self.trail_s * 100), self._on_trail,
                                   fmt=lambda v: "none" if v == 0 else f"{v / 100:.2f} s")
        note(c, "For a field that uses t, the time rate under the stage sets the speed.", pady=(4, PAD))

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
        c = card(right, "EXPORT", "Ctrl+E")
        self.ex_size_tabs = segmented(c, [k for k, _, _ in export.SIZES], self._on_ex_size)
        self.ex_size_tabs.set(self.ex_size)
        self.ex_size_tabs.pack(fill="x", padx=PAD, pady=(0, 6))
        self.ex_format_tabs = segmented(c, [lbl for _, lbl in export.FORMATS],
                                        lambda v: self._on_ex_format(_key(export.FORMATS, v)))
        self.ex_format_tabs.set(_label(export.FORMATS, self.ex_format))
        self.ex_format_tabs.pack(fill="x", padx=PAD, pady=(0, 6))
        row = ctk.CTkFrame(c, fg_color="transparent")
        row.pack(fill="x", padx=PAD, pady=(0, 2))
        self.ex_fps_tabs = segmented(row, ("30 fps", "60 fps"), self._on_ex_fps)
        self.ex_fps_tabs.set(f"{self.ex_fps} fps")
        self.ex_fps_tabs.pack(side="left", fill="x", expand=True)
        self.ex_ss_tabs = segmented(row, ("2×", "3×"), self._on_ex_ss)
        self.ex_ss_tabs.set(f"{self.ex_ss}×")
        self.ex_ss_tabs.pack(side="left", padx=(6, 0))
        self.row_seconds = SliderRow(c, "Duration", 1, 30, self.ex_seconds, self._on_ex_seconds,
                                     fmt=lambda v: f"{v} s")
        switch(c, "Include grid and axes", self.ex_grid, lambda: None).pack(anchor="w", padx=PAD, pady=(8, 0))
        self.lbl_ex = note(c, "", pady=(6, 8))
        ex = ctk.CTkFrame(c, fg_color="transparent")
        ex.pack(fill="x", padx=PAD, pady=(0, PAD))
        self.btn_export = button(ex, "Export video", self.export_video, accent=True, width=10)
        self.btn_export.pack(side="left", fill="x", expand=True)
        button(ex, "Save PNG", self.save_png, width=10).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.ex_progress = ctk.CTkProgressBar(c, height=6, fg_color=SURFACE_2, progress_color=ACCENT)
        self._ex_note()

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
        rate = ctk.CTkFrame(bar, fg_color="transparent", width=230)
        rate.grid(row=0, column=2, rowspan=2, sticky="ew", padx=(0, 4))
        self.row_rate = SliderRow(rate, "Time rate", 0, 100, round(_rate_slider(self.t_rate)), self._on_rate,
                                  fmt=lambda v: f"{_rate_from(v):.2g} t/s")
        button(bar, "↺  Reset time", self.reset_time, width=120).grid(row=0, column=3, rowspan=2, padx=(0, PAD))

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
        self.bind("<Control-e>", lambda e: self.export_video())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
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
        self.flow = None
        self.row_rate.enable(f.uses_t)
        self.row_speed.enable(not f.uses_t)
        self._invalidate()
        self._status("Field applied." + (" It uses t: Space runs time." if f.uses_t else ""))

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
        if self.flow is not None:
            self.flow.clear()             # old trails keep the colors of the previous palette
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

    def _on_density(self, v):
        self.density = v
        self.flow = None
        self._invalidate(static=False)

    def _on_ex_size(self, key):
        self.ex_size = key
        self._ex_note()
        self._invalidate(static=False)   # moves the export frame on the stage

    def _on_ex_format(self, key):
        self.ex_format = key
        self._ex_note()

    def _on_ex_fps(self, label):
        self.ex_fps = int(label.split()[0])
        self._ex_note()

    def _on_ex_ss(self, label):
        self.ex_ss = int(label.rstrip("×"))

    def _on_ex_seconds(self, v):
        self.ex_seconds = v
        self._ex_note()

    def _ex_dims(self):
        return next((w, h) for k, w, h in export.SIZES if k == self.ex_size)

    def _ex_note(self):
        w, h = self._ex_dims()
        text = f"{w} × {h} px, {self.ex_fps} fps, {self.ex_seconds} s: the dashed frame on the stage."
        if self.ex_format == "gif":
            text = (f"GIF preview: {min(w, export.GIF_WIDTH)} px wide, {50 if self.ex_fps > 30 else 25} fps "
                    f"(GIF counts time in 1/100 s), {self.ex_seconds} s.")
        elif self.ex_format == "png":
            text += " One PNG per frame, numbered."
        self.lbl_ex.configure(text=text)

    def _on_speed(self, v):
        self.p_speed = v

    def _on_trail(self, v):
        self.trail_s = v / 100

    def _on_rate(self, v):
        self.t_rate = _rate_from(v)

    def _style(self, px_per_pt=1.0):
        """The current look as a render.Style, for a render at px_per_pt pixels per stage pixel."""
        return render.Style(
            px_per_pt=px_per_pt, palette=self.palette, background=self.bg_kind, colored=self.colored.get(),
            arrow_spacing=self.arrow_spacing, arrow_length=self.arrow_len / 100,
            equal_length=self.equal_len.get(), stream_sep=self.stream_sep, stream_heads=self.stream_heads.get(),
            particle_speed=30.0 * self.p_speed, trail=self.trail_s)

    # ---------- view: pan, zoom, traces, probe ----------

    def _view(self):
        w, h = self._stage_size()
        return fieldmod.View(self.center[0], self.center[1], self.half, w, h)

    def _stage_size(self):
        return max(self.stage.winfo_width(), 50), max(self.stage.winfo_height(), 50)

    def _view_changed(self):
        self._invalidate(quick=True)
        if self.flow is not None:
            self.flow.clear()             # trails are in screen space; particles stay in the world

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
        """Schedules a redraw; with `static`, also a new static layer. A `quick` one is drawn at lower
        quality now and at full quality once things settle."""
        if static:
            self._request(quick)
        if quick:
            if self._full_id is not None:
                self.after_cancel(self._full_id)
            self._full_id = self.after(QUICK_MS, self._full_redraw)
        if self._redraw_id is None:
            self._redraw_id = self.after_idle(self._redraw)

    def _full_redraw(self):
        self._full_id = None
        self._request(quick=False)

    def _redraw(self):
        self._redraw_id = None
        if not self.running and self.field is not None:   # while playing, every tick redraws
            self._show(self._view())

    def _static_job(self, view, quick, guides=True):
        """A snapshot of everything the static layer depends on, as a job for the render thread.
        Without `guides` it is the art alone, as exports show it."""
        gen, field, t, mode, style, pins = self._gen, self.field, self.t, self.mode, self._style(), list(self.pins)
        arrows = mode == ARROWS or self.overlay.get()

        def build():
            try:
                img, grid, _ = render.build_static(field, view, t, style, arrows, mode == STREAMLINES,
                                                   mode == PARTICLES, pins, 1 if quick else 2, guides)
                return Static(gen, view, img, grid)
            except ValueError as exc:
                return Static(gen, view, error=f"Could not evaluate the field: {exc}")
            except Exception as exc:
                traceback.print_exc()
                return Static(gen, view, error=f"Rendering failed: {exc}")
        return build

    def _request(self, quick=False):
        """Asks the render thread for a static layer of the current view and settings."""
        if self.field is None:
            return
        self._gen += 1
        self._worker.submit(self._static_job(self._view(), quick))
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll)

    def _poll(self):
        self._poll_id = None
        got = None
        while not self._worker.results.empty():
            got = self._worker.results.get()
        if got is not None:
            self._accept(got)
            if not self.running:
                self._redraw()
        if self._static is None or self._static.gen < self._gen:
            self._poll_id = self.after(POLL_MS, self._poll)

    def _accept(self, static):
        if isinstance(static, Exception):
            self._status(f"Rendering failed: {static}")
            return
        if self._static is not None and static.gen < self._static.gen:
            return
        if static.error:
            self._status(static.error)
            static.img = np.full((static.view.h, static.view.w, 3), hex_to_rgb(SURFACE), np.uint8)
        if self._static is None and self._compiling:
            self._compiling = False
            self._status("Ready. Pick a preset or type your own Fx and Fy. Space animates the flow.")
        self._static = static

    def _compiling_note(self):
        self._compiling = self._static is None
        if self._compiling:
            self._status("Compiling the numerical kernels. This happens only on the first run and takes a few seconds…")

    def _fit_static(self, view):
        """The newest static layer as seen from `view`. One built for an older view (the user is panning or
        zooming and the new one is not ready yet) is moved and scaled to match, so it never lags behind."""
        st = self._static
        if st is None:
            return np.full((view.h, view.w, 3), hex_to_rgb(SURFACE), np.uint8)
        key = (view.cx, view.cy, view.scale, view.w, view.h)
        if self._fitted is not None and self._fitted[0] is st and self._fitted[1] == key:
            return self._fitted[2]
        old = st.view
        if (old.cx, old.cy, old.scale, old.w, old.h) == key:
            img = st.img
        else:
            k = old.scale / view.scale
            c = (view.cx - view.w / (2 * view.scale) - old.cx) * old.scale + old.w / 2
            f = old.h / 2 - (view.cy - old.cy) * old.scale - view.h / 2 * k
            img = np.asarray(Image.fromarray(st.img).transform(
                (view.w, view.h), Image.AFFINE, (k, 0, c, 0, k, f), Image.BILINEAR, fillcolor=SURFACE))
        self._fitted = (st, key, img)
        return img

    def _compose(self, view, static=None):
        """The frame to show: the static layer (the newest one, unless given), plus the particle trails
        screen-blended on top in the Particles view."""
        if static is None:
            static = self._fit_static(view)
        if self.mode != PARTICLES:
            return static
        self._ensure_flow(view)
        return render.screen(static, self.flow.image())

    def _show(self, view):
        self._frame = self._compose(view)
        self._photo = ImageTk.PhotoImage(Image.fromarray(self._frame))
        s = self.stage
        s.delete("all")
        s.create_image(0, 0, image=self._photo, anchor="nw")
        w, h = self._ex_dims()
        x, y, fw, fh = export.frame_rect(view, w, h)
        s.create_rectangle(x, y, x + fw, y + fh, outline=MUTED, dash=(6, 4))
        s.create_text(x + fw - 6, y + 6, anchor="ne", fill=MUTED, font=FONT_SMALL, text=f"{self.ex_size} · {w} × {h}")
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

    def _ensure_flow(self, view):
        """Particles for `view`: new ones if there are none or the stage changed size."""
        if self.flow is not None and (self.flow.view.w, self.flow.view.h) == (view.w, view.h):
            self.flow.view = view
            return
        self.flow = particles.Flow(round(self.density * view.w * view.h / 1e4), view, FPS)
        self.flow.paint(self.field, self.t, fieldmod.reference_speed(self.field, view, self.t), self._style())

    def _step(self, view, static=None):
        """Advances time by one frame, and the particles with it in the Particles view. They move through the
        grid of `static`, by default the newest static layer."""
        style = self._style()
        vref = fieldmod.reference_speed(self.field, view, self.t)
        dt = particles.frame_dt(self.field, view, style, vref, FPS, self.t_rate)
        static = static or self._static
        grid = static.grid if static is not None else None
        if self.mode == PARTICLES and grid is not None:   # no grid yet: the first render is still running
            self._ensure_flow(view)
            self.flow.step(grid, dt, vref, style)
        self.t += dt

    def toggle_play(self):
        if self.running:
            self._stop()
            self._status("Paused.")
            if self.field.uses_t:
                self._invalidate()        # frames while playing were quick ones
            return
        if self.mode != PARTICLES and not self.field.uses_t:
            self.set_mode(PARTICLES)      # a steady field only moves through its particles
        self.running = True
        self.btn_play.configure(text="❚❚  Pause")
        self._status("Playing. Particles follow the field; their colour shows the speed.")
        self._deadline = time.perf_counter()
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
        if self.field.uses_t:
            self._request(quick=True)     # shown by a later tick; until then the last one stays up
        self._show(view)
        self.lbl_t.configure(text=f"t = {self.t:.2f}")
        # Frames are due every 1/FPS s whatever the work took; after a stall the lost frames are dropped.
        now = time.perf_counter()
        self._deadline = max(self._deadline + 1 / FPS, now)
        self._after_id = self.after(max(1, round((self._deadline - now) * 1000)), self._tick)

    def reset_time(self):
        self.t = 0.0
        self.flow = None
        self.lbl_t.configure(text="t = 0.00")
        self._invalidate()

    # ---------- files ----------

    def _scene(self):
        """What the stage shows now, for an export."""
        return export.Scene(self.field, self._style(), self.t, self.t_rate,
                            arrows=self.mode == ARROWS or self.overlay.get(), streamlines=self.mode == STREAMLINES,
                            density=self.density if self.mode == PARTICLES else 0, pins=tuple(self.pins))

    def _ex_settings(self, fmt):
        w, h = self._ex_dims()
        return export.Settings(w, h, self.ex_fps, self.ex_seconds, fmt, self.ex_ss, self.ex_grid.get())

    def save_png(self):
        """A still of the export frame at the export size."""
        if self.field is None or self._export is not None:
            return
        path = filedialog.asksaveasfilename(title="Save image", defaultextension=".png",
                                            initialfile="vector_field.png", filetypes=[("PNG", "*.png")])
        if path:
            self._start_export(self._ex_settings("still"), path)

    def export_video(self):
        if self._export is not None:
            self._export.cancel.set()
            return
        if self.field is None:
            return
        ext = export.EXTENSIONS[self.ex_format]
        label = _label(export.FORMATS, self.ex_format)
        path = filedialog.asksaveasfilename(title="Export video", defaultextension=ext,
                                            initialfile=f"vector_field{ext}", filetypes=[(label, f"*{ext}")])
        if not path:
            return
        if self.mode != PARTICLES and not self.field.uses_t:
            self.set_mode(PARTICLES)      # a steady field only moves through its particles
        self._start_export(self._ex_settings(self.ex_format), path)

    def _start_export(self, settings, path):
        self._export = export.Export(export.Renderer(self._scene(), self._view(), settings), path)
        if settings.format != "still":
            self.btn_export.configure(text="Cancel")
        self.ex_progress.set(0)
        self.ex_progress.pack(fill="x", padx=PAD, pady=(0, PAD))
        self.after(EXPORT_POLL_MS, self._poll_export)

    def _poll_export(self):
        ex = self._export
        name = os.path.basename(ex.path)
        if not ex.finished:
            self.ex_progress.set(ex.done / ex.total)
            warm = ex.renderer.warm_up
            phase = (f"growing the trails ({ex.done} / {warm})" if ex.done < warm
                     else f"frame {ex.done - warm + 1} / {ex.settings.frames}")
            self._status(f"Exporting {name}: {phase}…")
            self.after(EXPORT_POLL_MS, self._poll_export)
            return
        self._export = None
        self.btn_export.configure(text="Export video")
        self.ex_progress.pack_forget()
        if ex.error:
            self._status(f"Export failed: {ex.error}")
        elif ex.cancelled:
            self._status("Export cancelled.")
        else:
            self._status(f"Saved: {name}")

    def _on_close(self):
        """Cancels a running export first, so ffmpeg is not left behind with half a file."""
        if self._export is not None and not self._export.finished:
            self._export.cancel.set()
            self.after(50, self._on_close)
            return
        self.destroy()

    def _status(self, text):
        self.lbl_status.configure(text=text)
