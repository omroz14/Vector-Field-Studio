"""Timings of the engine, for comparing changes:  python -m tests.benchmark
(Not a test: it asserts nothing. The first run also compiles the kernels.)"""
import os
import tempfile
import time

import numpy as np

from tests.helpers import preset
import export
import field as F
import kernels
import particles
import render
import viewport


def timed(fn, repeat=5):
    """Median wall time of fn() in milliseconds (after one warm-up call)."""
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return 1e3 * float(np.median(times))


def row(name, ms, note=""):
    print(f"  {name:56s} {ms:9.1f} ms  {note}")


def main():
    t = time.perf_counter()
    kernels.warm_up()
    print(f"kernels ready in {time.perf_counter() - t:.1f} s (compiled once, then cached on disk)\n")

    print("Field and grid")
    field, stage = preset("dipole", 1000, 780)
    x, y = np.random.default_rng(0).uniform(-3, 3, (2, 2000))
    row("Field() at 2000 points", timed(lambda: field(x, y), 200) , "(dipole)")
    for key in ("vortex", "dipole", "gyre"):
        f, v = preset(key, 1000, 780)
        row(f"grid of a 1000×780 stage + 25 % overscan ({key})", timed(lambda: F.FieldGrid(f, v, 0.0), 3))

    print("Streamlines")
    for key in ("vanderpol", "cells"):
        f, v = preset(key, 1000, 780)
        style = render.Style()
        region = viewport.Region().prepare(f, viewport.cache_for(v), style)
        vref = F.reference_speed(f, v, 0.0)
        n = region.grid.U.size
        row(f"layout over 3 × 3 stage views ({key})",
            timed(lambda: F.layout_streamlines(region.grid, vref, 22.0, 1.5), 3), f"{n / 1e6:.2f} M samples")

    print("Static layer (cached view: 1000×780 + 25 % overscan, 2× supersampled)")
    for key, mode, bg in (("dipole", "arrows", "none"), ("vanderpol", "streamlines", "magnitude")):
        f, v = preset(key, 1000, 780)
        style = render.Style(background=bg)
        cache = viewport.cache_for(v)
        region = viewport.Region()
        region.prepare(f, cache, style)
        vref = F.reference_speed(f, v, 0.0)
        ms = timed(lambda: render.build_static(f, cache, 0.0, style, mode == "arrows", mode == "streamlines",
                                               False, (), 2, True, vref, region), 3)
        row(f"{mode} on {bg} ({key})", ms)

    print("Particles at 1080×1920 (export size, k = 1.54)")
    f, v = preset("vortex", 1080, 1920)
    style = render.Style(px_per_pt=1.54)
    grid = F.FieldGrid(f, v, 0.0, spacing=style.px(style.sample))
    vref = F.reference_speed(f, v, 0.0)
    for density in (25, 80, 200):
        flow = particles.Flow(density, v, 30, style.px_per_pt, seed=0)
        dt = particles.frame_dt(f, v, style, vref, 30, 0.5)
        for _ in range(30):
            flow.step(grid, dt, vref, style)
        row(f"step + splat + fade + tone map, {len(flow.x)} particles", timed(lambda: flow.step(grid, dt, vref, style), 20))
    flow = particles.Flow(25, v, 30, style.px_per_pt, seed=0)
    row("tone map alone", timed(lambda: flow.restyle(vref, style), 20))
    a = np.random.default_rng(1).integers(0, 255, (1920, 1080, 3), np.uint8)
    row("screen blend", timed(lambda: render.screen(a, a), 20))

    print("Export (1080×1350, 30 fps, 2 s, MP4)")
    with tempfile.TemporaryDirectory() as d:
        for key, mode in (("vortex", "particles"), ("gyre", "particles"), ("vanderpol", "streamlines")):
            f, v = preset(key, 1000, 780)
            scene = export.Scene(f, render.Style(), density=25 if mode == "particles" else 0,
                                 streamlines=mode == "streamlines")
            r = export.Renderer(scene, v, export.Settings(seconds=2, period=f.period), seed=0)
            t = time.perf_counter()
            ex = export.Export(r, os.path.join(d, "b.mp4"))
            while not ex.finished:
                time.sleep(0.01)
            el = time.perf_counter() - t
            row(f"{key} {mode} ({r.kind} loop, {r.warm_up} warm-up steps)", 1e3 * el,
                f"{ex.total / el:.0f} steps/s{'' if ex.error is None else ' ERROR ' + ex.error}")


if __name__ == "__main__":
    main()
