"""
Check that the heuristic is actually being used in neural model search, and that
it works as expected.
"""

import unittest

import neurosym as ns
from neurosym.examples import near


class TestNeuralModels(unittest.TestCase):

    def test_doesnt_work_no_heuristic_combinator(self):
        nesting = 10
        with self.assertRaises(StopIteration):
            near.debug_nested_dsl.run_near_on_dsl(
                nesting,
                near.debug_nested_dsl.get_combinator_dsl(nesting),
                neural_modules={},
            )

    def test_no_heuristic_combinator_works_more_time(self):
        nesting = 3
        [result] = near.debug_nested_dsl.run_near_on_dsl(
            nesting,
            near.debug_nested_dsl.get_combinator_dsl(nesting),
            neural_modules={},
        )
        self.assertEqual(
            ns.render_s_expression(result.program),
            self.compute_combinator_soln(nesting),
        )

    def test_doesnt_work_no_heuristic_variables(self):
        nesting = 10
        with self.assertRaises(StopIteration):
            near.debug_nested_dsl.run_near_on_dsl(
                nesting,
                near.debug_nested_dsl.get_variable_dsl(nesting),
                neural_modules={},
            )

    def test_no_heuristic_variables_works_more_time(self):
        nesting = 3
        [result] = near.debug_nested_dsl.run_near_on_dsl(
            nesting,
            near.debug_nested_dsl.get_variable_dsl(nesting),
            neural_modules={},
        )
        self.assertEqual(
            ns.render_s_expression(result.program),
            self.compute_variable_soln(nesting),
        )

    def test_combinator_dsl_works_with_mlps(self):
        nesting = 10
        t = ns.TypeDefiner()
        neural_modules = {
            **near.create_modules(
                "mlp",
                [t("{f, 1} -> {f, %s}" % i) for i in range(1, 2 + nesting)],
                near.mlp_factory(hidden_size=10),
            ),
        }
        [result] = near.debug_nested_dsl.run_near_on_dsl(
            nesting, near.debug_nested_dsl.get_combinator_dsl(nesting), neural_modules
        )
        self.assertEqual(
            ns.render_s_expression(result.program),
            self.compute_combinator_soln(nesting),
        )

    def test_variable_dsl_works_with_transformer(self):
        nesting = 10
        t = ns.TypeDefiner()
        neural_modules = {
            **near.create_modules(
                "transformer",
                [t("{f, %s}" % i) for i in range(1, 2 + nesting)],
                near.transformer_factory(hidden_size=32),
            ),
        }
        [result] = near.debug_nested_dsl.run_near_on_dsl(
            nesting, near.debug_nested_dsl.get_variable_dsl(nesting), neural_modules
        )
        self.assertEqual(
            ns.render_s_expression(result.program),
            self.compute_combinator_soln(nesting),
        )

    def compute_combinator_soln(self, nesting):
        expected = ns.SExpression("terminal", ())
        for i in range(2, nesting + 1):
            expected = ns.SExpression(f"correct_{i}", (expected,))
        return ns.render_s_expression(expected)

    def compute_variable_soln(self, nesting):
        expected = ns.SExpression("terminal", (ns.SExpression("$0_0", ()),))
        for i in range(2, nesting + 1):
            expected = ns.SExpression(f"correct_{i}", (expected,))
        expected = ns.SExpression("lam", (expected,))
        return ns.render_s_expression(expected)
