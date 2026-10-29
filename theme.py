"""Color palette, fonts and small shared widgets (CustomTkinter, dark style)."""
import customtkinter as ctk

BG = "#0d0f14"          # window background
SURFACE = "#161922"     # cards and the stage
SURFACE_2 = "#1f2331"   # inputs, unselected buttons
BORDER = "#2b3144"
TEXT = "#e7e9f0"
MUTED = "#7d8497"
ACCENT = "#8b6cff"
ACCENT_HOVER = "#a18bff"
WARM = "#ffb454"
ERROR = "#ff6b81"

# Negative sizes are pixels, so tk.Canvas text and CTk widgets match.
FAMILY = "Helvetica"
FONT = (FAMILY, -13)
FONT_BOLD = (FAMILY, -13, "bold")
FONT_SMALL = (FAMILY, -11)
FONT_TITLE = (FAMILY, -20, "bold")
FONT_SECTION = (FAMILY, -11, "bold")
FONT_BIG = (FAMILY, -30, "bold")

PAD = 14   # inner padding of a card

# CustomTkinter bug: the side panels listen to the mouse wheel app-wide, and over widgets Tk builds itself
# (such as the file dialog) event.widget is a plain string, which crashes the scroll check.
_valid_scroll = ctk.CTkScrollableFrame._check_if_valid_scroll
ctk.CTkScrollableFrame._check_if_valid_scroll = (
    lambda self, widget: not isinstance(widget, str) and _valid_scroll(self, widget))


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def card(master, title, hint=None):
    """A rounded panel with a section title (and an optional muted note on the right). Returns it."""
    c = ctk.CTkFrame(master, fg_color=SURFACE, corner_radius=12)
    c.pack(fill="x", pady=(0, 10))
    head = ctk.CTkFrame(c, fg_color="transparent")
    head.pack(fill="x", padx=PAD, pady=(PAD - 2, 8))
    ctk.CTkLabel(head, text=title, font=FONT_SECTION, text_color=MUTED, height=16).pack(side="left")
    if hint:
        ctk.CTkLabel(head, text=hint, font=FONT_SMALL, text_color=MUTED, height=16).pack(side="right")
    return c


def button(master, text, command, accent=False, **kw):
    return ctk.CTkButton(master, text=text, command=command, height=kw.pop("height", 32),
                         corner_radius=8, font=FONT_BOLD if accent else FONT,
                         fg_color=ACCENT if accent else SURFACE_2,
                         hover_color=ACCENT_HOVER if accent else BORDER,
                         text_color=TEXT, **kw)


def note(master, text, **pack):
    """Muted, wrapped helper text."""
    lbl = ctk.CTkLabel(master, text=text, font=FONT_SMALL, text_color=MUTED,
                       justify="left", anchor="w", wraplength=250)
    lbl.pack(fill="x", padx=PAD, **({"pady": (6, 0)} | pack))
    return lbl


def entry(master, **kw):
    return ctk.CTkEntry(master, height=30, corner_radius=8, font=FONT, fg_color=SURFACE_2,
                        border_color=BORDER, border_width=1, text_color=TEXT, **kw)


def segmented(master, values, command, **kw):
    return ctk.CTkSegmentedButton(master, values=list(values), command=command, height=30,
                                  font=FONT, fg_color=SURFACE_2, selected_color=ACCENT,
                                  selected_hover_color=ACCENT_HOVER, unselected_color=SURFACE_2,
                                  unselected_hover_color=BORDER, text_color=TEXT, corner_radius=8, **kw)


def switch(master, text, variable, command):
    return ctk.CTkSwitch(master, text=text, variable=variable, command=command, font=FONT,
                         text_color=TEXT, progress_color=ACCENT, button_color=TEXT,
                         button_hover_color="#ffffff", fg_color=BORDER)


class SliderRow:
    """A label, a live value on the right and a slider under them."""

    def __init__(self, master, label, lo, hi, value, command, fmt=str):
        self._fmt = fmt
        self._command = command
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", padx=PAD, pady=(4, 0))
        ctk.CTkLabel(row, text=label, font=FONT, text_color=TEXT, height=20).pack(side="left")
        self.value = ctk.CTkLabel(row, text=fmt(value), font=FONT_BOLD, text_color=TEXT, height=20)
        self.value.pack(side="right")
        self.slider = ctk.CTkSlider(master, from_=lo, to=hi, number_of_steps=hi - lo, height=16,
                                    fg_color=SURFACE_2, progress_color=ACCENT, button_color=TEXT,
                                    button_hover_color="#ffffff", command=self._on)
        self.slider.set(value)
        self.slider.pack(fill="x", padx=PAD - 4, pady=(2, 4))

    def _on(self, v):
        v = int(round(v))
        self.value.configure(text=self._fmt(v))
        self._command(v)

    def set(self, v):
        """Moves the slider without calling the command."""
        self.slider.set(v)
        self.value.configure(text=self._fmt(int(v)))

    def enable(self, on):
        self.slider.configure(state="normal" if on else "disabled",
                              button_color=TEXT if on else MUTED,
                              progress_color=ACCENT if on else BORDER)
        self.value.configure(text_color=TEXT if on else MUTED)


class ChoiceGrid:
    """Mutually exclusive toggle buttons in a grid. For lists too long for a segmented button."""

    def __init__(self, master, options, current, command, cols=2):
        self.buttons = {}
        self._command, self._cols = command, cols
        self.wrap = ctk.CTkFrame(master, fg_color="transparent")
        self.wrap.pack(fill="x", padx=PAD - 2)
        for c in range(cols):
            self.wrap.columnconfigure(c, weight=1, uniform="col")
        self.set_options(options, current)

    def set_options(self, options, current):
        """Replaces the buttons (when the list of choices changes)."""
        for b in self.buttons.values():
            b.destroy()
        self.buttons = {}
        for i, (key, label) in enumerate(options):
            b = button(self.wrap, label, lambda k=key: self._command(k), height=30)
            b.grid(row=i // self._cols, column=i % self._cols, sticky="ew", padx=2, pady=2)
            self.buttons[key] = b
        self.select(current)

    def select(self, current):
        for key, b in self.buttons.items():
            b.configure(fg_color=ACCENT if key == current else SURFACE_2,
                        hover_color=ACCENT_HOVER if key == current else BORDER)
