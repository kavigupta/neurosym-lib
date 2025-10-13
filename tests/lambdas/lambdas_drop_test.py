import unittest
from functools import partial

import numpy as np
from torch import nn

import neurosym as ns
from neurosym.examples import near


class TestDSLWithDrops(unittest.TestCase):
    def test_basic_dsl_with_drops(self):
        dsl = near.with_drops.basic_drop_dsl(3)
        result = dsl.render().split("\n")
        print(result)
        self.assertEqual(
            result,
            [
                "              + :: ({f, 1}, {f, 1}) -> {f, 1}",
                "            lam :: L<#body|{f, 1};{f, 1};{f, 1}> -> ({f, 1}, {f, 1}, {f, 1}) -> #body",
                "           $0_0 :: V<{f, 1}@0>",
                "           $1_0 :: V<{f, 1}@1>",
                "           $2_0 :: V<{f, 1}@2>",
                "        drop0_0 :: D<#body - $0::{f, 1}> -> #body",
                "        drop1_0 :: D<#body - $1::{f, 1}> -> #body",
                "        drop2_0 :: D<#body - $2::{f, 1}> -> #body",
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
        dsl = near.with_drops.basic_drop_dsl(3)
        self.assertEquivalence("(lam ($1_0))", "(lam (drop0_0 ($0_0)))", dsl)
        self.assertEquivalence("(lam ($2_0))", "(lam (drop0_0 ($1_0)))", dsl)
        self.assertEquivalence("(lam ($1_0))", "(lam (drop2_0 ($1_0)))", dsl)


class TestSearchGraphDrops(unittest.TestCase):
    def test_search_graph_with_drops(self):
        dsl = near.with_drops.basic_drop_dsl(3)
        sg = ns.DSLSearchGraph(
            dsl,
            hole_set_chooser=ns.ChooseAll(),
            test_predicate=lambda node: True,
            target_type=ns.parse_type("({f, 1}, {f, 1}, {f, 1}) -> {f, 1}"),
            metadata_computer=ns.NoMetadataComputer(),
        )
        path = [
            ["??::<({f, 1}, {f, 1}, {f, 1}) -> {f, 1}>", ["(lam ??::<{f, 1}|0={f, 1},1={f, 1},2={f, 1}>)"]],
            [
                "(lam ??::<{f, 1}|0={f, 1},1={f, 1},2={f, 1}>)",
                [
                    "(lam (+ ??::<{f, 1}|0={f, 1},1={f, 1},2={f, 1}> ??::<{f, 1}|0={f, 1},1={f, 1},2={f, 1}>))",
                    "(lam ($0_0))",
                    "(lam ($1_0))",
                    "(lam ($2_0))",
                    "(lam (drop0_0 ??::<{f, 1}|0={f, 1},1={f, 1}>))",
                    "(lam (drop1_0 ??::<{f, 1}|0={f, 1},1={f, 1}>))",
                    "(lam (drop2_0 ??::<{f, 1}|0={f, 1},1={f, 1}>))",
                ],
            ],
            [
                "(lam (drop0_0 ??::<{f, 1}|0={f, 1},1={f, 1}>))",
                [
                    "(lam (drop0_0 (+ ??::<{f, 1}|0={f, 1},1={f, 1}> ??::<{f, 1}|0={f, 1},1={f, 1}>)))",
                    "(lam (drop0_0 ($0_0)))",
                    "(lam (drop0_0 ($1_0)))",
                    "(lam (drop0_0 (drop0_0 ??::<{f, 1}|0={f, 1}>)))",
                    "(lam (drop0_0 (drop1_0 ??::<{f, 1}|0={f, 1}>)))",
                ],
            ],
            [
                "(lam (drop0_0 (drop1_0 ??::<{f, 1}|0={f, 1}>)))",
                [
                    "(lam (drop0_0 (drop1_0 (+ ??::<{f, 1}|0={f, 1}> ??::<{f, 1}|0={f, 1}>))))",
                    "(lam (drop0_0 (drop1_0 ($0_0))))",
                    "(lam (drop0_0 (drop1_0 (drop0_0 ??::<{f, 1}>))))",
                ],
            ],
            [
                "(lam (drop0_0 (drop1_0 (drop0_0 ??::<{f, 1}>))))",
                ["(lam (drop0_0 (drop1_0 (drop0_0 (+ ??::<{f, 1}> ??::<{f, 1}>)))))"],
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


class TestDropInterface(unittest.TestCase):
    def test_sequential_dsl_astar_interface(self):
        """
        Test sequential_dsl with Astar in the NEARInterface
        """
        datamodule = near.with_drops.add_variables_domain_datamodule()

        original_dsl = near.with_drops.basic_drop_dsl(10, is_vectorized=True)
        print(original_dsl.render())
        interface = near.NEAR(n_epochs=20, max_depth=1000)

        def is_goal(node):
            f2 = original_dsl.compute(original_dsl.initialize(node.program))
            delta = (
                (f2(datamodule.train.inputs) - datamodule.train.outputs) ** 2
            ).mean()
            return delta < 0.01

        interface.register_search_params(
            dsl=original_dsl,
            type_env=ns.TypeDefiner(),
            neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=100),
            search_strategy=partial(ns.search.bounded_astar, max_depth=3),
            loss_callback=nn.functional.mse_loss,
            validation_params=dict(
                cost=near.with_drops.MinimalStepsNearStructuralCostWithDrops,
                structural_cost_weight=0.1,
            ),
            is_goal=is_goal,
        )
        [result] = interface.fit(
            datamodule=datamodule,
            program_signature="{f, 10} -> {f, 1}",
            n_programs=1,
        )
        print(result)
