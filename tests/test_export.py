"""Exports through ffmpeg: what the files contain (codec, pixel format, colour tags, size, frame rate, frame
count), and that a cancelled export leaves nothing behind."""
import glob
import os
import tempfile
import time
import unittest

from tests.helpers import decode, mean_step, preset, probe
import export
import render


def run(path, settings, key="vortex", mode="particles"):
    field, stage = preset(key, 600, 460)
    scene = export.Scene(field, render.Style(), density=25 if mode == "particles" else 0,
                         streamlines=mode == "streamlines")
    ex = export.Export(export.Renderer(scene, stage, settings, seed=1), path)
    while not ex.finished:
        time.sleep(0.02)
    return ex


class Formats(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_mp4(self):
        s = export.Settings(540, 676, 30, 1)
        ex = run(self.path("a.mp4"), s)
        self.assertIsNone(ex.error)
        info = probe(ex.path)
        self.assertEqual((info["codec"], info["pix_fmt"]), ("h264", "yuv420p"), info["line"])
        self.assertIn("bt709", info["tags"], info["line"])
        self.assertIn("tv", info["tags"], info["line"])
        self.assertEqual((info["width"], info["height"], info["fps"]), (540, 676, 30.0))
        frames = decode(ex.path, 540, 676)
        self.assertEqual(len(frames), 30)
        # A loop: across the seam the picture changes about as much as between any two frames.
        steps = [mean_step(frames[i + 1], frames[i]) for i in range(len(frames) - 1)]
        self.assertLess(mean_step(frames[0], frames[-1]), 3 * max(steps))

    def test_mp4_60fps_vertical(self):
        ex = run(self.path("b.mp4"), export.Settings(540, 960, 60, 0.5))
        info = probe(ex.path)
        self.assertEqual((info["width"], info["height"], info["fps"]), (540, 960, 60.0))
        self.assertEqual(len(decode(ex.path, 540, 960)), 30)

    def test_gif(self):
        ex = run(self.path("c.gif"), export.Settings(1080, 1080, 30, 1, "gif"))
        info = probe(ex.path)
        self.assertEqual((info["codec"], info["width"], info["fps"]), ("gif", export.GIF_WIDTH, 25.0))

    def test_prores(self):
        ex = run(self.path("d.mov"), export.Settings(400, 500, 30, 0.2, "prores"), mode="streamlines")
        info = probe(ex.path)
        self.assertEqual(info["codec"], "prores", info["line"])
        self.assertTrue(info["pix_fmt"].startswith("yuv444p"), info["line"])

    def test_png_sequence_and_still(self):
        run(self.path("e.png"), export.Settings(300, 300, 30, 0.2, "png"))
        self.assertEqual(len(glob.glob(self.path("e_*.png"))), 6)
        ex = run(self.path("f.png"), export.Settings(300, 400, format="still"))
        self.assertIsNone(ex.error)
        self.assertTrue(os.path.getsize(ex.path) > 0)

    def test_cancel_removes_the_file(self):
        field, stage = preset("vortex", 600, 460)
        scene = export.Scene(field, render.Style(), density=25)
        ex = export.Export(export.Renderer(scene, stage, export.Settings(seconds=30)), self.path("g.mp4"))
        time.sleep(0.5)
        ex.cancel.set()
        while not ex.finished:
            time.sleep(0.02)
        self.assertTrue(ex.cancelled)
        self.assertFalse(os.path.exists(self.path("g.mp4")))


class Frame(unittest.TestCase):
    def test_export_frame_fits_the_stage(self):
        _, stage = preset("vortex", 1000, 700)
        for _, w, h in export.SIZES:
            x, y, fw, fh = export.frame_rect(stage, w, h)
            self.assertAlmostEqual(fw / fh, w / h, places=9)
            self.assertTrue(x >= export.FRAME_MARGIN - 1e-9 and y >= export.FRAME_MARGIN - 1e-9)
            self.assertTrue(x + fw <= stage.w and y + fh <= stage.h)


if __name__ == "__main__":
    unittest.main()
