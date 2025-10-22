# pylint: disable=duplicate-code
# better to have duplication in tests than overly abstract ones
import unittest

import neurosym as ns


class TestSearchGraphShield(unittest.TestCase):

    def getDSL(self):
        dslf = ns.DSLFactory()
        dslf.production("+", "(f, i) -> f", lambda x, y: x + y)
        dslf.production("1f", "() -> f", lambda: 1.0)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("floor", "(f) -> i", int)
        dslf.lambdas()
        dslf.prune_to("(i, f) -> i")
        dsl = dslf.finalize()
        print(dsl.render())
        return dsl

    def test_search_graph_with_shield(self):
        dsl = self.getDSL()
        sg = ns.DSLSearchGraph(
            dsl,
            hole_set_chooser=ns.ChooseAll(),
            test_predicate=lambda node: True,
            target_type=ns.parse_type("(i, f) -> i"),
            metadata_computer=ns.NoMetadataComputer(),
        )
        path = [
            [
                "??::<(i, f) -> i>",
                ["(lam ??::<i|0=f,1=i>)"],
            ],
            [
                "(lam ??::<i|0=f,1=i>)",
                ["(lam (1))", "(lam (floor ??::<f|0=f,1=i>))", "(lam ($1_1))"],
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

    def test_compute_type_forward(self):
        dsl = self.getDSL()
        self.assertEqual(
            dsl.compute_type(ns.parse_s_expression("(lam ($1_1))")).short_repr(),
            "<(i, f) -> i|>",
        )
