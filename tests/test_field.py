"""Expressions, the field and its period in t."""
import math
import unittest

import numpy as np

from tests import helpers  # noqa: F401  (puts the project on the path)
import field as F


class Expressions(unittest.TestCase):
    def test_rejects_unsafe_syntax(self):
        for fx in ("__import__('os')", "x.real", "[x][0]", "lambda: 1", "foo(x)", "sin(x=1)", ""):
            with self.assertRaises(ValueError, msg=fx):
                F.Field(fx, "0")

    def test_values_and_polar_names(self):
        f = F.Field("r*cos(theta)", "x^2")
        u, v = f(np.array([3.0]), np.array([4.0]))
        self.assertAlmostEqual(u[0], 3.0)
        self.assertAlmostEqual(v[0], 9.0)

    def test_nan_and_inf_become_zero_without_touching_the_input(self):
        x = np.array([0.0, 1.0, np.nan])
        u, v = F.Field("x", "1/x")(x, x)
        self.assertTrue(np.isnan(x[2]), "the input was changed")
        self.assertEqual(list(u), [0.0, 1.0, 0.0])
        self.assertEqual(list(v), [0.0, 1.0, 0.0])

    def test_constant_broadcasts(self):
        u, v = F.Field("1", "t")(np.zeros(4), np.zeros(4), 2.5)
        self.assertEqual(u.shape, (4,))
        self.assertTrue(np.all(v == 2.5))


class Period(unittest.TestCase):
    CASES = [
        (("sin(t)", "y"), 2 * math.pi),
        (("cos(2*pi*t/3 + x)", "0"), 3.0),
        (("sin(2*t)", "cos(3*t)"), 2 * math.pi),
        (("sin(t)*sin(1.5*t)", "x"), 4 * math.pi),
        (("sin(sin(t))", "x"), 2 * math.pi),
        (("sin(-t/2)", "x"), 4 * math.pi),
        (("sin(t) + sin(sqrt(2)*t)", "0"), None),   # incommensurate
        (("x*t", "y"), None),
        (("sin(t^2)", "y"), None),
        (("exp(-t)*sin(t)", "y"), None),
        (("-y", "x"), None),                        # steady
    ]

    def test_cases(self):
        for (fx, fy), want in self.CASES:
            got = F.Field(fx, fy).period
            if want is None:
                self.assertIsNone(got, fx)
            else:
                self.assertAlmostEqual(got, want, places=9, msg=fx)

    def test_double_gyre(self):
        _, _, fx, fy, _, _ = next(p for p in F.PRESETS if p[0] == "gyre")
        self.assertAlmostEqual(F.Field(fx, fy).period, 2 * math.pi / 0.63, places=12)


if __name__ == "__main__":
    unittest.main()
