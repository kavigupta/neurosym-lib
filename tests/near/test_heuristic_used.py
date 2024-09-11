"""
Check that the heuristic is actually being used in neural model search, and that
it works as expected.
"""

import unittest

import neurosym as ns
from neurosym.examples import near

nesting = 10
combinator_dsl = near.debug_nested_dsl.get_combinator_dsl(nesting)
variable_dsl = near.debug_nested_dsl.get_variable_dsl(nesting)


class TestNeuralModels(unittest.TestCase):

    def test_doesnt_work_no_heuristic_combinator(self):
        with self.assertRaises(StopIteration):
            near.debug_nested_dsl.run_near_on_dsl(
                nesting, combinator_dsl, neural_modules={}
            )

    def test_doesnt_work_no_heuristic_variables(self):
        with self.assertRaises(StopIteration):
            near.debug_nested_dsl.run_near_on_dsl(
                nesting, variable_dsl, neural_modules={}
            )

    def test_combinator_dsl_works_with_mlps(self):
        t = ns.TypeDefiner()
        neural_modules = {
            **near.create_modules(
                "mlp",
                [t("{f, 1} -> {f, %s}" % i) for i in range(1, 2 + nesting)],
                near.mlp_factory(hidden_size=10),
            ),
        }
        [result] = near.debug_nested_dsl.run_near_on_dsl(
            nesting, combinator_dsl, neural_modules
        )
        expected = ns.SExpression("terminal", ())
        for i in range(2, nesting + 1):
            expected = ns.SExpression(f"correct_{i}", (expected,))
        self.assertEqual(
            ns.render_s_expression(result.program), ns.render_s_expression(expected)
        )
