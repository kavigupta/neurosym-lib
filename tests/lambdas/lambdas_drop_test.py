import unittest

import numpy as np

import neurosym as ns


def basic_drop_dsl():
    dslf = ns.DSLFactory()

    dslf.production("+", "(f, f) -> f", lambda x, y: x + y)
    dslf.lambdas(include_drops=True, max_env_depth=5, max_arity=5)
    dslf.prune_to("(f, f, f) -> f")
    dsl = dslf.finalize()

    # assert "lam :: L<#body|f;f;f> -> (f, f, f) -> #body" in dsl.render()
    # assert "drop0_3 :: D<#body - $0::f> -> #body" in dsl.render()
    # assert "$0_1 :: V<f@0>" in dsl.render()

    return dsl


class TestDSLWithDrops(unittest.TestCase):
    def test_basic_dsl_with_drops(self):
        dsl = basic_drop_dsl()
        result = dsl.render().split("\n")
        self.assertEqual(
            result,
            [
                "              + :: (f, f) -> f",
                "            lam :: L<#body|f;f;f> -> (f, f, f) -> #body",
                "           $0_0 :: V<f@0>",
                "           $1_0 :: V<f@1>",
                "           $2_0 :: V<f@2>",
                "        drop0_3 :: D<#body - $0::f> -> #body",
                "        drop1_3 :: D<#body - $1::f> -> #body",
                "        drop2_3 :: D<#body - $2::f> -> #body",
            ],
        )


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
        self.assertEquivalence("(lam ($1_0))", "(lam (drop0_3 ($0_0)))", dsl)
        self.assertEquivalence("(lam ($2_0))", "(lam (drop0_3 ($1_0)))", dsl)
        self.assertEquivalence("(lam ($1_0))", "(lam (drop2_3 ($1_0)))", dsl)


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
        path = [
            ["??::<(f, f, f) -> f>", ["(lam ??::<f|0=f,1=f,2=f>)"]],
            [
                "(lam ??::<f|0=f,1=f,2=f>)",
                [
                    "(lam (+ ??::<f|0=f,1=f,2=f> ??::<f|0=f,1=f,2=f>))",
                    "(lam ($0_0))",
                    "(lam ($1_0))",
                    "(lam ($2_0))",
                    "(lam (drop0_3 ??::<f|0=f,1=f>))",
                    "(lam (drop1_3 ??::<f|0=f,1=f>))",
                    "(lam (drop2_3 ??::<f|0=f,1=f>))",
                ],
            ],
            [
                "(lam (drop0_3 ??::<f|0=f,1=f>))",
                [
                    "(lam (drop0_3 (+ ??::<f|0=f,1=f> ??::<f|0=f,1=f>)))",
                    "(lam (drop0_3 ($0_0)))",
                    "(lam (drop0_3 ($1_0)))",
                    "(lam (drop0_3 (drop0_3 ??::<f|0=f>)))",
                    "(lam (drop0_3 (drop1_3 ??::<f|0=f>)))",
                ],
            ],
            [
                "(lam (drop0_3 (drop1_3 ??::<f|0=f>)))",
                [
                    "(lam (drop0_3 (drop1_3 (+ ??::<f|0=f> ??::<f|0=f>))))",
                    "(lam (drop0_3 (drop1_3 ($0_0))))",
                    "(lam (drop0_3 (drop1_3 (drop0_3 ??::<f>))))",
                ],
            ],
            [
                "(lam (drop0_3 (drop1_3 (drop0_3 ??::<f>))))",
                ["(lam (drop0_3 (drop1_3 (drop0_3 (+ ??::<f> ??::<f>)))))"],
            ],
        ]
        nodes = [sg.initial_node()]
        for target, expansions in path:
            [node] = [n for n in nodes if ns.render_s_expression(n.program) == target]
            nodes = list(sg.expand_node(node))
            known_expansions = [ns.render_s_expression(n.program) for n in nodes]
            print("Expanding:", target)
            print("\t", known_expansions)
            self.assertEqual(expansions, known_expansions)
        print("Final node:", ns.render_s_expression(node.program))
        for node in sg.expand_node(node):
            print("\t", ns.render_s_expression(node.program))
