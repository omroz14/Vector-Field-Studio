"""Shared pieces of the tests: presets, small views, frame comparisons and an ffprobe-like reader."""
import re
import subprocess

import imageio_ffmpeg
import numpy as np

import field as F
import render
import viewport


def preset(key, w=640, h=480):
    """(Field, snapped stage View) of a preset, shifted off any round centre."""
    _, _, fx, fy, c, half = next(p for p in F.PRESETS if p[0] == key)
    return F.Field(fx, fy), viewport.snap(F.View(c[0] + 0.0123, c[1] - 0.0456, half, w, h))


def moved(view, dx, dy):
    """The snapped view dx pixels right and dy pixels down."""
    return viewport.snap(F.View.at(view.cx + dx / view.scale, view.cy - dy / view.scale, view.scale, view.w, view.h))


def overlap(a, va, b, vb):
    """The parts of images a (of view va) and b (of view vb) that show the same pixels, as int arrays."""
    x0, y0 = int(vb.ox - va.ox), int(vb.oy - va.oy)
    xa, xb = max(0, x0), min(a.shape[1], x0 + b.shape[1])
    ya, yb = max(0, y0), min(a.shape[0], y0 + b.shape[0])
    return a[ya:yb, xa:xb].astype(int), b[ya - y0:yb - y0, xa - x0:xb - x0].astype(int)


def mean_step(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def probe(path):
    """What ffprobe would say about the first video stream: codec, pixel format, colour tags, size, fps,
    frame count. Read from ffmpeg's own report, since imageio-ffmpeg ships no ffprobe."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    err = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    line = next(l for l in err.splitlines() if "Video:" in l)
    m = re.search(r"Video: (\w+).*?, (\w+)(\(([^)]*)\))?, (\d+)x(\d+)", line)
    fps = re.search(r"([\d.]+) fps", line)
    count = subprocess.run([ff, "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                           capture_output=True, text=True)
    frames = re.findall(r"frame=\s*(\d+)", count.stderr)
    return dict(codec=m.group(1), pix_fmt=m.group(2), tags=[t.strip() for t in (m.group(4) or "").split(",")],
                width=int(m.group(5)), height=int(m.group(6)), fps=float(fps.group(1)) if fps else None,
                line=line.strip())


def decode(path, width, height):
    """Every frame of a video as uint8 arrays."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def static(field, view, style, region=None, **kw):
    """The static layer of a view (a cache view: stage plus overscan), drawn from `region` if given."""
    if region is not None:
        region.prepare(field, view, style)
    vref = kw.pop("vref", None) or F.reference_speed(field, view, 0.0)
    return render.build_static(field, view, 0.0, style, vref=vref, region=region, **kw)[0]
