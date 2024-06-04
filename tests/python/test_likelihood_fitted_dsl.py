import unittest
from fractions import Fraction

import numpy as np

import neurosym as ns

from .utils import fit_to


class TestLikelihoodFittedDSL(unittest.TestCase):
    def compute_likelihood(self, corpus, program):
        dfa, _, fam, dist = fit_to(corpus)
        program = ns.to_type_annotated_ns_s_exp(
            ns.python_to_python_ast(program), dfa, "M"
        )
        like = fam.compute_likelihood(dist, program)
        like = Fraction.from_float(float(np.exp(like))).limit_denominator()
        results = fam.compute_likelihood_per_node(dist, program)
        results = [
            (
                ns.render_s_expression(x),
                Fraction.from_float(float(np.exp(y))).limit_denominator(),
            )
            for x, y in results
            if y != 0  # remove zero log-likelihoods
        ]
        print(like)
        print(results)
        return like, results

    def test_likelihood(self):
        like, results = self.compute_likelihood(["x = 2", "y = 3", "y = 4"], "y = 4")
        self.assertAlmostEqual(like, Fraction(2, 9))
        self.assertEqual(
            results,
            [
                ("(const-&y:0~Name)", Fraction(2, 3)),
                ("(const-i4~Const)", Fraction(1, 3)),
            ],
        )

    def test_likelihood_def_use_check(self):
        like, results = self.compute_likelihood(
            ["x = 2; y = x", "y = 2; x = y"], "x = 2; y = x"
        )
        self.assertAlmostEqual(like, Fraction(1, 8))
        self.assertEqual(
            results,
            [
                ("(const-&x:0~Name)", Fraction(1, 2)),
                ("(const-&y:0~Name)", Fraction(1, 2)),
                ("(Name~E (const-&x:0~Name) (Load~Ctx))", Fraction(1, 2)),
            ],
        )

    def test_likelihood_zero(self):
        like, results = self.compute_likelihood(
            ["y = x + 2", "y = 2 + 3", "y = 4"], "y = 2 + x"
        )
        self.assertAlmostEqual(like, Fraction(0))
        self.assertEqual(
            results,
            [
                (
                    ns.render_s_expression(
                        ns.parse_s_expression(
                            """
                            (BinOp~E
                                (Constant~E (const-i2~Const) (const-None~ConstKind))
                                (Add~O) (Name~E (const-g_x~Name) (Load~Ctx)))
                            """
                        )
                    ),
                    Fraction(2, 3),
                ),
                (
                    "(Constant~E (const-i2~Const) (const-None~ConstKind))",
                    Fraction(1, 2),
                ),
                ("(const-i2~Const)", Fraction(1, 2)),
                ("(Name~E (const-g_x~Name) (Load~Ctx))", Fraction(0, 1)),
            ],
        )
