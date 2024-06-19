import unittest
from fractions import Fraction

import numpy as np

import neurosym as ns

from .utils import fit_to


class EnumerateFittedDslTest(unittest.TestCase):
    def enumerate(self, *programs):
        _, _, fam, dist = fit_to(programs)
        out = []
        for x, y in fam.enumerate(dist, min_likelihood=-10):
            assert isinstance(y, np.float64)
            out.append(
                (
                    Fraction.from_float(np.exp(y)).limit_denominator(),
                    ns.s_exp_to_python(ns.render_s_expression(x)),
                )
            )
        out = sorted(out, key=lambda x: (-x[0], x[1]))
        print(out)
        return out

    def test_enumerate_fitted_dsl_basic(self):
        self.assertEqual(
            self.enumerate("x = y + 2 + 2"),
            [
                (Fraction(1, 2), "x = y + 2"),
                (Fraction(1, 4), "x = y + 2 + 2"),
                (Fraction(1, 8), "x = y + 2 + 2 + 2"),
                (Fraction(1, 16), "x = y + 2 + 2 + 2 + 2"),
                (Fraction(1, 32), "x = y + 2 + 2 + 2 + 2 + 2"),
                (Fraction(1, 64), "x = y + 2 + 2 + 2 + 2 + 2 + 2"),
                (Fraction(1, 128), "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2"),
                (Fraction(1, 256), "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2"),
                (Fraction(1, 512), "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2"),
                (Fraction(1, 1024), "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2"),
                (
                    Fraction(1, 2048),
                    "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2",
                ),
                (
                    Fraction(1, 4096),
                    "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2",
                ),
                (
                    Fraction(1, 8192),
                    "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2",
                ),
                (
                    Fraction(1, 16384),
                    "x = y + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2",
                ),
            ],
        )

    def test_enumerate_fitted_dsl_cartesian(self):
        self.assertEqual(
            self.enumerate("x = 2", "y = 3"),
            [
                (Fraction(1, 4), "x = 2"),
                (Fraction(1, 4), "x = 3"),
                (Fraction(1, 4), "y = 2"),
                (Fraction(1, 4), "y = 3"),
            ],
        )

    def test_enumerate_def_use_check(self):
        self.assertEqual(
            self.enumerate("x = 2; y = x", "y = 2; x = y"),
            [
                (Fraction(1, 8), "x = 2\nx = 2"),
                (Fraction(1, 8), "x = 2\nx = x"),
                (Fraction(1, 8), "x = 2\ny = 2"),
                (Fraction(1, 8), "x = 2\ny = x"),
                (Fraction(1, 8), "y = 2\nx = 2"),
                (Fraction(1, 8), "y = 2\nx = y"),
                (Fraction(1, 8), "y = 2\ny = 2"),
                (Fraction(1, 8), "y = 2\ny = y"),
            ],
        )

    def test_enumerate_def_use_check_wglobal(self):
        self.assertEqual(
            self.enumerate("x = print; y = x", "y = print; x = y"),
            [
                (Fraction(1, 6), "x = print\nx = print"),
                (Fraction(1, 6), "x = print\ny = print"),
                (Fraction(1, 6), "y = print\nx = print"),
                (Fraction(1, 6), "y = print\ny = print"),
                (Fraction(1, 12), "x = print\nx = x"),
                (Fraction(1, 12), "x = print\ny = x"),
                (Fraction(1, 12), "y = print\nx = y"),
                (Fraction(1, 12), "y = print\ny = y"),
            ],
        )
