"""Palettes: the user's own ones go to a JSON file and come back."""
import json
import os
import tempfile
import unittest

from tests import helpers  # noqa: F401
import palettes


class UserPalettes(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sub", "palettes.json")

    def tearDown(self):
        for key, _ in list(palettes.PALETTE_NAMES):
            if palettes.is_user(key):
                palettes.remove_user(key, save=False)
        self.dir.cleanup()

    def test_round_trip(self):
        key = palettes.add_user("Neon", ["#120024", "#ff00aa", "#00ffee"], "#00a0ff", "#ff5050", save=False)
        palettes.save_user(self.path)
        palettes.remove_user(key, save=False)
        self.assertNotIn(key, palettes.PALETTES)
        self.assertIsNone(palettes.load_user(self.path))
        p = palettes.PALETTES[key]
        self.assertEqual(p.seq_stops, ["#120024", "#ff00aa", "#00ffee"])
        self.assertEqual((p.below, p.above), ("#00a0ff", "#ff5050"))
        self.assertEqual(p.seq.shape, (256, 3))

    def test_replace_keeps_one_entry(self):
        k = palettes.add_user("A", ["#000000", "#ffffff"], "#0000ff", "#ff0000", save=False)
        palettes.add_user("B", ["#000000", "#ff0000"], "#0000ff", "#ff0000", save=False, replace=k)
        names = [lbl for key, lbl in palettes.PALETTE_NAMES if palettes.is_user(key)]
        self.assertEqual(names, ["B"])

    def test_missing_and_broken_files(self):
        self.assertIsNone(palettes.load_user(self.path))   # no file yet: fine
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as fh:
            fh.write("{broken")
        self.assertIn("Could not read", palettes.load_user(self.path))
        with open(self.path, "w") as fh:   # bad entries are skipped, good ones load
            json.dump({"palettes": [{"name": "x", "seq": ["red"], "below": "#000000", "above": "#ffffff"},
                                    {"name": "ok", "seq": ["#000000", "#ffffff"], "below": "#0000ff",
                                     "above": "#ff0000"}]}, fh)
        palettes.load_user(self.path)
        self.assertIn(palettes.USER + "ok", palettes.PALETTES)
        self.assertNotIn(palettes.USER + "x", palettes.PALETTES)

    def test_built_ins_are_complete(self):
        for key, _ in palettes.BUILT_IN:
            p = palettes.PALETTES[key]
            self.assertEqual((p.seq.shape, p.div.shape, p.line.shape), ((256, 3),) * 3, key)


if __name__ == "__main__":
    unittest.main()
