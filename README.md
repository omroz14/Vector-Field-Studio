# Vector Field Studio

A desktop app (Python + CustomTkinter) for exploring 2D vector fields F(x, y, t) = (Fx, Fy).
It shares its look with Pixel Sort Studio.

## Features

- **12 presets:** uniform, source, sink, vortex, saddle, spiral sink, shear, dipole, cellular flow,
  pendulum, Van der Pol oscillator and the time-dependent double gyre.
- **Your own equations** for Fx and Fy, using `x, y, t, r, theta`, the usual math functions, `pi`, `e`. `^` means power.
- **Three views:** arrows, evenly spaced streamlines and animated particles with trails.
- **Background maps:** speed |F|, divergence or curl.
- **Interaction:** scroll to zoom, drag to pan, click to trace the streamline through a point.
  The probe shows F, |F|, div F and curl F under the cursor.
- **Time:** fields that use `t` evolve while playing.
- **Export** the view to PNG and the animation to GIF.

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Tkinter is a system package: `sudo apt install python3-tk` on Debian / Ubuntu / Linux Mint.

## Keys

| Key | Action |
| --- | --- |
| Space | play / pause |
| 1–3 | arrows / streamlines / particles |
| B | next background map |
| R | reset view |
| C | clear traced streamlines |
| T | reset time |
| Ctrl+S | save PNG |

## Files

- `main.py`: entry point
- `ui.py`: the window, panels, stage and animation
- `field.py`: expression parsing, presets, the view transform, streamline tracing, particles
- `render.py`: color maps, heatmaps, arrows and streamlines
- `theme.py`: colors, fonts and shared widgets (same as Pixel Sort Studio)
