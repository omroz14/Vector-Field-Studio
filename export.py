"""Export: an offscreen renderer of the export frame and writers that pipe its frames into ffmpeg.

The export frame is the largest rectangle of the chosen aspect centred in the stage. It is rendered at the
output size with px_per_pt = k = output height / frame height on the stage, so the video looks like the
preview, only sharper. The renderer has its own view, static layer and particles; the stage keeps running.
"""
import glob
import math
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, replace

import imageio_ffmpeg
import numpy as np
from PIL import Image

import field as fieldmod
import particles
import render

SIZES = (("4:5", 1080, 1350), ("9:16", 1080, 1920), ("1:1", 1080, 1080), ("16:9", 1920, 1080))
FORMATS = (("mp4", "MP4"), ("gif", "GIF"), ("prores", "ProRes"), ("png", "PNGs"))
EXTENSIONS = {"mp4": ".mp4", "gif": ".gif", "prores": ".mov", "png": ".png", "still": ".png"}
FRAME_MARGIN = 16     # px between the export frame and the edges of the stage
CROSSFADE = 18        # frames blended across the seam of a loop that cannot repeat exactly
GIF_WIDTH = 540       # a GIF is a preview: at most this wide


def frame_rect(view, w, h, margin=FRAME_MARGIN):
    """(x, y, width, height) in stage pixels: the largest w:h rectangle centred in the stage."""
    s = min(max(view.w - 2 * margin, 2) / w, max(view.h - 2 * margin, 2) / h)
    fw, fh = w * s, h * s
    return (view.w - fw) / 2, (view.h - fh) / 2, fw, fh


@dataclass(frozen=True)
class Scene:
    """What the stage shows, frozen when an export starts. `style` is the stage style (px_per_pt 1);
    `density` is particles per 100 × 100 points, 0 for none; `vref` the stage's typical speed (None: measure)."""
    field: object
    style: render.Style
    t: float = 0.0
    t_rate: float = 0.5
    arrows: bool = False
    streamlines: bool = False
    density: float = 0.0
    pins: tuple = ()
    vref: float = None
    layout: tuple = None    # (grid, streamline layout) of the stage, so the export shows the same lines


@dataclass(frozen=True)
class Settings:
    width: int = 1080
    height: int = 1350
    fps: int = 30
    seconds: float = 10.0
    format: str = "mp4"     # mp4, gif, prores, png (a sequence) or still (one PNG)
    ss: int = 2             # supersampling of the static layer
    grid: bool = False      # draw the grid and axes into the export
    loop: bool = True       # a seamless loop: the frame after the last one is the first one
    period: float = None    # period in t of a time-dependent field (None: not periodic, crossfade the seam)

    @property
    def frames(self):
        return 1 if self.format == "still" else max(1, round(self.seconds * self.fps))


class Renderer:
    """Renders the export frame of the stage view `stage` at settings.width × settings.height."""

    def __init__(self, scene, stage, settings, seed=0):
        self.scene, self.settings = scene, settings
        x, y, fw, fh = frame_rect(stage, settings.width, settings.height)
        cx, cy = stage.to_world(x + fw / 2, y + fh / 2)
        self.view = fieldmod.View(cx, cy, fh / (2 * stage.scale), settings.width, settings.height)
        self.style = replace(scene.style, px_per_pt=settings.height / fh)
        # The colour scale and the particle pace of the preview, held for the whole export.
        self.vref = scene.vref or fieldmod.reference_speed(scene.field, stage, scene.t)
        self._steady = None   # (image, grid) of a steady field, built once

        # How the clip loops: "exact" repeats every `frames` frames (steady fields, and periodic ones with
        # whole periods per loop), "crossfade" blends the end into the start, "plain" does not loop.
        n, fps, uses_t = settings.frames, settings.fps, scene.field.uses_t
        self.kind = "plain"
        if settings.loop and settings.format != "still":
            self.kind = "exact" if not uses_t or settings.period else "crossfade"
        self.dt = particles.frame_dt(scene.field, self.view, self.style, self.vref, fps, scene.t_rate)
        self.periods = 0
        if self.kind == "exact" and uses_t:
            # The nearest whole number of periods to what the time rate would cover in the clip.
            self.periods = max(1, round(scene.t_rate * n / fps / settings.period))
            self.dt = self.periods * settings.period / n

        self.flow = None
        if scene.density > 0:
            k = self.style.px_per_pt
            if self.kind == "exact":
                self.flow = particles.LoopFlow(scene.density, self.view, fps, n, k, seed)
            else:
                self.flow = particles.Flow(scene.density, self.view, fps, k, seed)
        # Steps before the first frame comes out: particles growing their trails, and for a crossfade the
        # frames held back for the seam.
        self.fade = min(CROSSFADE, n // 2) if self.kind == "crossfade" else 0
        self._grow = 0
        if self.flow is not None:
            self._grow = (self.flow.warm_up(self.style) if self.kind == "exact"
                          else math.ceil(fps * max(particles.LIFE[1], 5 * self.style.trail)))
        self.warm_up = self._grow + self.fade

    def _static(self, t, record):
        """(art image or None, grid) at time t. A steady field's layer is built once."""
        sc, st = self.scene, self.settings
        if self._steady is not None:
            return self._steady
        if record or not sc.field.uses_t:
            img, grid, _ = render.build_static(sc.field, self.view, t, self.style, sc.arrows, sc.streamlines,
                                               self.flow is not None, sc.pins, st.ss, st.grid, self.vref,
                                               layout=sc.layout)
        else:   # warming up a time-dependent field: the particles need only the grid
            img, grid = None, fieldmod.FieldGrid(sc.field, self.view, t, spacing=self.style.px(self.style.sample))
        if not sc.field.uses_t:
            self._steady = (img, grid)
        return img, grid

    def _frame(self, f, record):
        """Advances to frame f (negative while warming up) and returns its image if `record`."""
        n = self.settings.frames
        # An exact loop takes t from f mod n, so the field repeats bit for bit; otherwise t runs on.
        t = self.scene.t + ((f % n) if self.kind == "exact" else f) * self.dt
        img, grid = self._static(t, record)
        if self.flow is not None:
            if self.kind == "exact":
                self.flow.step(grid, self.dt, self.vref, self.style, f)
            else:
                self.flow.step(grid, self.dt, self.vref, self.style)
        if not record:
            return None
        return render.screen(img, self.flow.image()) if self.flow is not None else img

    def frames(self, extra=0):
        """Yields one item per step: None while warming up (self.warm_up of them), then settings.frames
        frames (uint8, height × width × 3) starting at the scene's t. `extra` more frames continue an exact
        loop past its end (frame n + i should equal frame i: a test of the loop)."""
        n = self.settings.frames
        for f in range(-self._grow, 0):
            yield self._frame(f, False)
        if self.kind != "crossfade":
            for f in range(n + (extra if self.kind == "exact" else 0)):
                yield self._frame(f, True)
            return
        # Crossfade: render n + M frames F; the loop is F_M … F_(n−1), then G_j = (1 − w_j)·F_(n+j) + w_j·F_j
        # for j < M with w_j rising smoothly from 0 to 1. Its last frame leads into F_M and F_(n−1) into its
        # first blended one, so both joins are continuous. Starting at F_M lets frames stream out at once.
        M = self.fade
        held = []
        for f in range(n + M):
            img = self._frame(f, True)
            if f < M:
                held.append(img)
                yield None
            elif f < n:
                yield img
            else:
                j = f - n
                u = (j + 0.5) / M
                w = u * u * (3 - 2 * u)
                yield ((1 - w) * img + w * held[j] + 0.5).astype(np.uint8)


# ---------- writers ----------

def _sequence_pattern(path):
    root, _ = os.path.splitext(path)
    return root + "_%05d.png"


def ffmpeg_args(settings, path):
    """The ffmpeg command line that reads raw RGB frames on stdin and writes `path`."""
    w, h, fps = settings.width, settings.height, settings.fps
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    fmt = settings.format
    if fmt == "mp4":
        # In a loop the last frames are seen as often as the first: mbtree would spend less on them
        # (nothing refers to them), which shows as a jump in quality at the seam.
        loop = ["-x264-params", "mbtree=0"] if settings.loop else []
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264",
                "-preset", "slow", "-crf", "16", "-profile:v", "high", *loop, *tags, "-movflags", "+faststart", path]
    elif fmt == "prores":
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p10le", "-c:v", "prores_ks",
                "-profile:v", "4", "-vendor", "apl0", *tags, path]
    elif fmt == "gif":
        # GIF delays are in 1/100 s, so 25 or 50 fps; one palette from the whole clip, ordered dithering.
        gfps = 50 if fps > 30 else 25
        gw = min(w, GIF_WIDTH)
        cmd += ["-filter_complex", f"fps={gfps},scale={gw}:-1:flags=lanczos,split[a][b];"
                                   "[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop", "0", path]
    elif fmt == "png":
        cmd += ["-c:v", "png", "-start_number", "1", _sequence_pattern(path)]
    else:
        raise ValueError(f"unknown format {fmt}")
    return cmd


class FFmpegWriter:
    def __init__(self, cmd):
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _error(self):
        self._err.seek(0)
        return self._err.read().decode(errors="replace").strip() or f"ffmpeg exited with {self._proc.returncode}"

    def write(self, frame):
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        except (BrokenPipeError, OSError):
            self._proc.wait()
            raise RuntimeError(self._error()) from None

    def close(self):
        self._proc.stdin.close()
        failed = self._proc.wait()
        message = self._error() if failed else None
        self._err.close()
        if failed:
            raise RuntimeError(message)

    def abort(self):
        self._proc.kill()
        self._proc.wait()
        try:
            self._proc.stdin.close()
        except OSError:   # the pipe is broken: ffmpeg is gone
            pass
        self._err.close()


class StillWriter:
    def __init__(self, path):
        self.path = path

    def write(self, frame):
        Image.fromarray(frame).save(self.path)

    def close(self):
        pass

    def abort(self):
        pass


def _remove_output(settings, path):
    paths = glob.glob(glob.escape(_sequence_pattern(path)).replace("%05d", "[0-9]" * 5)) \
        if settings.format == "png" else [path]
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


class Export:
    """Runs one export on its own thread. The Tk thread reads `done`, `total`, `finished` and `error`,
    and may set `cancel`; nothing here touches Tk."""

    def __init__(self, renderer, path):
        self.renderer, self.path = renderer, path
        self.settings = renderer.settings
        self.total = renderer.warm_up + self.settings.frames
        self.done = 0
        self.finished = False
        self.cancelled = False
        self.error = None
        self.cancel = threading.Event()
        threading.Thread(target=self._run, name="export", daemon=True).start()

    def _run(self):
        writer = None
        try:
            if self.settings.format == "still":
                writer = StillWriter(self.path)
            else:
                writer = FFmpegWriter(ffmpeg_args(self.settings, self.path))
            for frame in self.renderer.frames():
                if self.cancel.is_set():
                    writer.abort()
                    _remove_output(self.settings, self.path)
                    self.cancelled = True
                    return
                if frame is not None:
                    writer.write(frame)
                self.done += 1
            writer.close()
        except Exception as exc:
            if writer is not None:
                writer.abort()
            _remove_output(self.settings, self.path)
            self.error = str(exc) or type(exc).__name__
        finally:
            self.finished = True
