import itertools
import unittest
from fractions import Fraction

import numpy as np

import neurosym as ns

from .bigram_test import fam, fam_with_ordering, fam_with_ordering_231, fam_with_vars

arith_dist = ns.TreeDistribution(
    1,
    {
        ((0, 0),): [(1, np.log(1 / 4)), (2, np.log(3 / 4))],
        ((1, 0),): [(1, np.log(1 / 4)), (2, np.log(3 / 4))],
        ((1, 1),): [(1, np.log(1 / 8)), (2, np.log(7 / 8))],
    },
    [("root", 1), ("+", 2), ("1", 0)],
    ns.NoopPreorderMask,
    lambda _: ns.DefaultNodeOrdering(),
)


def enumerated(*args, **kwargs):
    return [
        (ns.render_s_expression(program), likelihood)
        for program, likelihood in itertools.islice(
            ns.enumerate_tree_dist(*args, **kwargs), 100
        )
    ]


class TreeDistributionTest(unittest.TestCase):
    def enumerate_dsl(self, family, dist, min_likelihood=-6):
        result = list(family.enumerate(dist, min_likelihood=min_likelihood))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        result = [
            (
                ns.render_s_expression(prog),
                Fraction(*np.exp(likelihood).as_integer_ratio()).limit_denominator(),
            )
            for prog, likelihood in result
        ]
        print(result)
        return set(result)

    def test_uniqueness(self):
        programs = enumerated(arith_dist)
        assert len(programs) == len(set(programs))

    def test_enumeration(self):
        out = dict(enumerated(arith_dist))
        p_one = 3 / 4
        p_add = 1 / 4
        self.assertAlmostEqual(out["(1)"], np.log(p_one))
        self.assertAlmostEqual(out["(+ (1) (1))"], np.log(p_add * 3 / 4 * 7 / 8))
        self.assertAlmostEqual(
            out["(+ (1) (+ (1) (1)))"],
            np.log(p_add * (3 / 4) * (1 / 8 * 3 / 4 * 7 / 8)),
        )
        self.assertAlmostEqual(
            out["(+ (+ (1) (1)) (1))"],
            np.log(p_add * (1 / 4 * 3 / 4 * 7 / 8) * (7 / 8)),
        )

    def test_enumeration_from_dsl_uniform(self):
        result = self.enumerate_dsl(fam, fam.uniform())

        self.assertEqual(
            result,
            {
                ("(1)", Fraction(1, 3)),
                ("(2)", Fraction(1, 3)),
                ("(+ (1) (1))", Fraction(1, 27)),
                ("(+ (2) (1))", Fraction(1, 27)),
                ("(+ (1) (2))", Fraction(1, 27)),
                ("(+ (2) (2))", Fraction(1, 27)),
                ("(+ (1) (+ (1) (1)))", Fraction(1, 243)),
                ("(+ (2) (+ (1) (1)))", Fraction(1, 243)),
                ("(+ (1) (+ (2) (1)))", Fraction(1, 243)),
                ("(+ (2) (+ (2) (1)))", Fraction(1, 243)),
                ("(+ (1) (+ (1) (2)))", Fraction(1, 243)),
                ("(+ (2) (+ (1) (2)))", Fraction(1, 243)),
                ("(+ (1) (+ (2) (2)))", Fraction(1, 243)),
                ("(+ (2) (+ (2) (2)))", Fraction(1, 243)),
                ("(+ (+ (1) (1)) (1))", Fraction(1, 243)),
                ("(+ (+ (2) (1)) (1))", Fraction(1, 243)),
                ("(+ (+ (1) (2)) (1))", Fraction(1, 243)),
                ("(+ (+ (2) (2)) (1))", Fraction(1, 243)),
                ("(+ (+ (1) (1)) (2))", Fraction(1, 243)),
                ("(+ (+ (2) (1)) (2))", Fraction(1, 243)),
                ("(+ (+ (1) (2)) (2))", Fraction(1, 243)),
                ("(+ (+ (2) (2)) (2))", Fraction(1, 243)),
            },
        )

    def test_enumeration_from_dsl_fitted_to_single(self):
        result = self.enumerate_dsl(
            fam,
            fam.counts_to_distribution(
                fam.count_programs([[ns.parse_s_expression("(+ (1) (2))")]])
            )[0],
        )
        self.assertEqual(
            result,
            {("(+ (1) (2))", Fraction(1, 1))},
        )

    def test_enumeration_from_dsl_fitted_to_two(self):
        result = self.enumerate_dsl(
            fam,
            fam.counts_to_distribution(
                fam.count_programs(
                    [[ns.parse_s_expression(x) for x in ("(+ (1) (2))", "(+ (2) (1))")]]
                )
            )[0],
        )
        self.assertEqual(
            result,
            {
                ("(+ (1) (1))", Fraction(1, 4)),
                ("(+ (1) (2))", Fraction(1, 4)),
                ("(+ (2) (1))", Fraction(1, 4)),
                ("(+ (2) (2))", Fraction(1, 4)),
            },
        )

    def test_enumeration_from_dsl_fitted_to_nested(self):
        result = self.enumerate_dsl(
            fam,
            fam.counts_to_distribution(
                fam.count_programs(
                    [
                        [
                            ns.parse_s_expression(x)
                            for x in ("(+ (1) (2))", "(+ (1) (+ (1) (2)))")
                        ]
                    ]
                )
            )[0],
        )
        self.assertEqual(
            result,
            {
                ("(+ (1) (2))", Fraction(2, 3)),
                ("(+ (1) (+ (1) (2)))", Fraction(2, 9)),
                ("(+ (1) (+ (1) (+ (1) (2))))", Fraction(2, 27)),
                ("(+ (1) (+ (1) (+ (1) (+ (1) (2)))))", Fraction(2, 81)),
                ("(+ (1) (+ (1) (+ (1) (+ (1) (+ (1) (2))))))", Fraction(2, 243)),
                (
                    "(+ (1) (+ (1) (+ (1) (+ (1) (+ (1) (+ (1) (2)))))))",
                    Fraction(2, 729),
                ),
            },
        )

    def test_enumeration_from_dsl_fitted_unbalanced(self):
        result = self.enumerate_dsl(
            fam,
            fam.counts_to_distribution(
                fam.count_programs(
                    [
                        [
                            ns.parse_s_expression(x)
                            for x in (
                                "(+ (1) (2))",
                                "(+ (1) (2))",
                                "(+ (1) (+ (1) (2)))",
                            )
                        ]
                    ]
                )
            )[0],
        )
        self.assertEqual(
            result,
            {
                ("(+ (1) (2))", Fraction(3, 4)),
                ("(+ (1) (+ (1) (2)))", Fraction(3, 16)),
                ("(+ (1) (+ (1) (+ (1) (2))))", Fraction(3, 64)),
                ("(+ (1) (+ (1) (+ (1) (+ (1) (2)))))", Fraction(3, 256)),
                ("(+ (1) (+ (1) (+ (1) (+ (1) (+ (1) (2))))))", Fraction(3, 1024)),
            },
        )

    def test_enumeration_from_dsl_with_variables_uniform(self):
        result = self.enumerate_dsl(
            fam_with_vars, fam_with_vars.uniform(), min_likelihood=-6
        )

        self.assertEqual(
            result,
            {
                ("(1)", Fraction(1, 4)),
                ("(2)", Fraction(1, 4)),
                ("(+ (1) (1))", Fraction(1, 64)),
                ("(+ (2) (1))", Fraction(1, 64)),
                ("(+ (1) (2))", Fraction(1, 64)),
                ("(+ (2) (2))", Fraction(1, 64)),
                ("(call (lam ($0_0)) (1))", Fraction(1, 80)),
                ("(call (lam (1)) (1))", Fraction(1, 80)),
                ("(call (lam (2)) (1))", Fraction(1, 80)),
                ("(call (lam ($0_0)) (2))", Fraction(1, 80)),
                ("(call (lam (1)) (2))", Fraction(1, 80)),
                ("(call (lam (2)) (2))", Fraction(1, 80)),
            },
        )

    def test_enumeration_from_dsl_with_ordering(self):
        result = self.enumerate_dsl(
            fam_with_ordering, fam_with_ordering.uniform(), min_likelihood=-6
        )
        self.assertEqual(result, {("(+ (1) (2) (3))", Fraction(1))})

    def test_enumeration_from_dsl_with_ordering_231(self):
        result = self.enumerate_dsl(
            fam_with_ordering_231, fam_with_ordering_231.uniform(), min_likelihood=-6
        )
        self.assertEqual(result, {("(+ (2) (3) (1))", Fraction(1))})
