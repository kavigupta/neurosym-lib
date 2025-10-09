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


class TestSearchGraphDrops(unittest.TestCase):
    def test_search_graph_with_drops(self):
        dsl = basic_drop_dsl()
        sg = ns.DSLSearchGraph(
            dsl,
            hole_set_chooser=ns.ChooseAll(),
            test_predicate=lambda node: True,
            target_type=ns.parse_type("(f, f, f) -> f"),
            metadata_computer=ns.NoMetadataComputer(),
        )
        path = ["??::<(f, f, f) -> f>", "(lam ??::<f|0=f,1=f,2=f>)"]
        node = sg.initial_node()
        self.assertEqual(path.pop(0), ns.render_s_expression(node.program))
        while path:
            target = path.pop(0)
            print("Expanding to find:", target)
            nodes = list(sg.expand_node(node))
            print([ns.render_s_expression(node.program) for node in nodes])
            self.assertIn(
                target, [ns.render_s_expression(node.program) for node in nodes]
            )
            [node] = [n for n in nodes if ns.render_s_expression(n.program) == target]
        print("Final node:", ns.render_s_expression(node.program))
        1 / 0
