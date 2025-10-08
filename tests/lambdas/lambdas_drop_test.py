import unittest

import numpy as np

import neurosym as ns


def basic_drop_dsl():
    dslf = ns.DSLFactory()

    dslf.production("+", "(f, f) -> f", lambda x, y: x + y)
    dslf.lambdas(include_drops=True, max_env_depth=5, max_arity=5)
    dslf.prune_to("(f, f, f) -> f")
    dsl = dslf.finalize()

    assert "lam :: L<#body|f;f;f> -> (f, f, f) -> #body" in dsl.render()
    assert "drop0_3 :: D<#body - $0::f> -> #body" in dsl.render()
    assert "$0_1 :: V<f@0>" in dsl.render()

    return dsl


class TestEvaluateDrops(unittest.TestCase):
    def assertEquivalence(self, expr1, expr2, dsl):
        expr1, expr2 = ns.parse_s_expression(expr1), ns.parse_s_expression(expr2)
        result1 = dsl.compute(dsl.initialize(expr1))
        result2 = dsl.compute(dsl.initialize(expr2))
        rng = np.random.default_rng(0)
        for _ in range(10):
            v = rng.random(3)
            self.assertEqual(result1(*v), result2(*v))

    def test_basic_drop(self):
        dsl = basic_drop_dsl()
        self.assertEquivalence("(lam ($1_1))", "(lam (drop0_3 ($0_0)))", dsl)
        self.assertEquivalence("(lam ($2_1))", "(lam (drop0_3 ($1_0)))", dsl)
        self.assertEquivalence("(lam ($1_1))", "(lam (drop2_3 ($1_0)))", dsl)
