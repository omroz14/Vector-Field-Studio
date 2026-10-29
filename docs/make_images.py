"""Renders the README's view images and demo GIFs with the exporter:  python docs/make_images.py
(The main screenshot is of the running app and is taken by hand.)"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import export  # noqa: E402
import field as F  # noqa: E402
import render  # noqa: E402
import viewport  # noqa: E402

OUT = os.path.join(HERE, "images")


def make(name, key, fmt, w, h, mode="particles", seconds=4, density=25, **style):
    _, _, fx, fy, c, half = next(p for p in F.PRESETS if p[0] == key)
    field = F.Field(fx, fy)
    m = export.FRAME_MARGIN
    stage = F.View(c[0], c[1], half * (h + 2 * m) / h, w + 2 * m, h + 2 * m)   # the frame is w × h at k = 1
    st = render.Style(**style)
    layout = None
    if mode == "streamlines" and not field.uses_t:
        region = viewport.Region().prepare(field, viewport.cache_for(stage), st)
        region.lines(F.reference_speed(field, stage, 0.0), st)
        layout = region.layout
    scene = export.Scene(field, st, t=0.0, t_rate=0.5, arrows=mode == "arrows", streamlines=mode == "streamlines",
                         density=density if mode == "particles" else 0, layout=layout)
    settings = export.Settings(w, h, 30, seconds, fmt, 3, False, True, field.period)
    path = os.path.join(OUT, name)
    t = time.perf_counter()
    ex = export.Export(export.Renderer(scene, stage, settings, seed=7), path)
    while not ex.finished:
        time.sleep(0.05)
    print(f"{name}: {ex.error or 'ok'}, {os.path.getsize(path) / 1e6:.1f} MB, {time.perf_counter() - t:.0f} s")


if __name__ == "__main__":
    make("view-arrows.png", "dipole", "still", 800, 560, "arrows")
    make("view-streamlines.png", "vanderpol", "still", 800, 560, "streamlines", palette="magma",
         background="magnitude")
    make("view-particles.png", "vanderpol", "still", 800, 560, palette="turbo", glow=True)
    make("vortex_ice.gif", "vortex", "gif", 400, 280, seconds=3, palette="ice", glow=True)
    make("van_der_pol_magma.gif", "vanderpol", "gif", 400, 280, density=45, palette="magma", glow=True)
    make("double_gyre_fire.gif", "gyre", "gif", 400, 280, palette="fire", background="magnitude")
