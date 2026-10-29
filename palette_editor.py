"""The palette editor: a dialog to make a palette of one's own, or to change or delete one made earlier."""
import tkinter as tk
from tkinter import colorchooser

import customtkinter as ctk

import palettes
from theme import *


class PaletteEditor(ctk.CTkToplevel):
    """Edits the colours from slow to fast (2 to 7 of them) and the colours of negative and positive values.
    Starts from the palette `start`. With `editing`, Save replaces that user palette and Delete removes it.
    Calls on_done(key, error): key of the saved palette, or None after Delete; error if it could not be written
    to disk (it still works until the app closes)."""

    def __init__(self, master, start, editing, on_done):
        super().__init__(master, fg_color=BG)
        self.title("Edit palette" if editing else "New palette")
        self.resizable(False, False)
        self.transient(master)
        self.on_done = on_done
        self.editing = start if editing else None
        p = palettes.PALETTES[start]
        self.seq = list(p.seq_stops)[:palettes.MAX_STOPS]
        self.below, self.above = p.below, p.above
        label = dict(palettes.PALETTE_NAMES).get(start, "")

        body = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=12)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", padx=PAD, pady=(PAD, 10))
        ctk.CTkLabel(row, text="Name", font=FONT, text_color=TEXT, width=50, anchor="w").pack(side="left")
        self.ent_name = entry(row, width=230)
        self.ent_name.pack(side="left", fill="x", expand=True)
        self.ent_name.insert(0, label if editing else f"My {label.lower()}" if label else "My palette")

        ctk.CTkLabel(body, text="Colours, from slow to fast", font=FONT, text_color=TEXT, height=20,
                     anchor="w").pack(fill="x", padx=PAD)
        self.row_seq = ctk.CTkFrame(body, fg_color="transparent")
        self.row_seq.pack(fill="x", padx=PAD, pady=(4, 10))

        signs = ctk.CTkFrame(body, fg_color="transparent")
        signs.pack(fill="x", padx=PAD)
        self.btn_below = self._swatch(signs, self.below, lambda: self._pick_sign("below"))
        ctk.CTkLabel(signs, text="below zero", font=FONT, text_color=TEXT).pack(side="left", padx=(6, 18))
        self.btn_above = self._swatch(signs, self.above, lambda: self._pick_sign("above"))
        ctk.CTkLabel(signs, text="above zero", font=FONT, text_color=TEXT).pack(side="left", padx=(6, 0))
        note(body, "These two colour div F and curl F; the stage colour marks zero.", pady=(4, 10))

        self.preview = tk.Canvas(body, width=320, height=34, bg=SURFACE, highlightthickness=0)
        self.preview.pack(padx=PAD)
        self.lbl_err = ctk.CTkLabel(body, text="", font=FONT_SMALL, text_color=ERROR, height=16)
        self.lbl_err.pack(fill="x", padx=PAD, pady=(4, 0))

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.pack(fill="x", padx=PAD, pady=(6, PAD))
        button(buttons, "Save", self._save, accent=True, width=90).pack(side="right")
        button(buttons, "Cancel", self.destroy, width=90).pack(side="right", padx=(0, 6))
        if editing:
            button(buttons, "Delete", self._delete, width=90).pack(side="left")

        self._build_stops()
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(50, self._modal)

    def _modal(self):
        try:
            self.grab_set()
        except tk.TclError:   # not mapped yet
            self.after(50, self._modal)
        self.ent_name.focus_set()

    @staticmethod
    def _swatch(master, colour, command):
        b = ctk.CTkButton(master, text="", width=30, height=30, corner_radius=6, fg_color=colour,
                          hover_color=colour, border_width=1, border_color=BORDER, command=command)
        b.pack(side="left")
        return b

    def _build_stops(self):
        for w in self.row_seq.winfo_children():
            w.destroy()
        for i, colour in enumerate(self.seq):
            b = self._swatch(self.row_seq, colour, lambda i=i: self._pick_stop(i))
            b.pack_configure(padx=(0, 4))
        if len(self.seq) < palettes.MAX_STOPS:
            button(self.row_seq, "+", self._add, width=30, height=30).pack(side="left", padx=(4, 0))
        if len(self.seq) > 2:
            button(self.row_seq, "−", self._remove, width=30, height=30).pack(side="left", padx=(4, 0))
        self._draw_preview()

    def _ask(self, colour, title):
        return colorchooser.askcolor(color=colour, parent=self, title=title)[1]

    def _pick_stop(self, i):
        colour = self._ask(self.seq[i], f"Colour {i + 1} of {len(self.seq)}")
        if colour:
            self.seq[i] = colour
            self._build_stops()

    def _pick_sign(self, which):
        colour = self._ask(getattr(self, which), f"Colour {which} zero")
        if colour:
            setattr(self, which, colour)
            btn = self.btn_below if which == "below" else self.btn_above
            btn.configure(fg_color=colour, hover_color=colour)
            self._draw_preview()

    def _add(self):
        self.seq.append(self.seq[-1])   # a copy of the last one, to change
        self._build_stops()

    def _remove(self):
        self.seq.pop()
        self._build_stops()

    def _draw_preview(self):
        """The magnitude ramp on top and the signed one under it, as they will look on the stage."""
        p = palettes.Palette.simple(self.seq, self.below, self.above)
        c = self.preview
        c.delete("all")
        w = int(c.cget("width"))
        for y0, lut in ((0, p.seq), (18, p.div)):
            for x in range(0, w, 2):
                r, g, b = lut[int(x / w * 255)]
                c.create_rectangle(x, y0, x + 2, y0 + 16, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

    def _save(self):
        name = self.ent_name.get().strip()
        if not name:
            self.lbl_err.configure(text="Give the palette a name.")
            return
        taken = {lbl.lower() for k, lbl in palettes.PALETTE_NAMES if k != self.editing}
        if name.lower() in taken:
            self.lbl_err.configure(text="There is a palette with this name already.")
            return
        key = palettes.add_user(name, self.seq, self.below, self.above, save=False, replace=self.editing)
        self._finish(key)

    def _delete(self):
        palettes.remove_user(self.editing, save=False)
        self._finish(None)

    def _finish(self, key):
        error = None
        try:
            palettes.save_user()
        except OSError as exc:
            error = f"Could not save your palettes ({exc}); they last until the app closes."
        self.destroy()
        self.on_done(key, error)
