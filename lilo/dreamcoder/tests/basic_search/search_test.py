"""
Just checks that the package can be imported
"""

import unittest

import neurosym as ns

dsl = ns.examples.basic_arith_dsl()


class TestSearch(unittest.TestCase):
    def test_bfs(self):
        g = ns.DSLSearchGraph(
            dsl,
            ns.parse_type("i"),
            ns.ChooseFirst(),
            lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
            metadata_computer=ns.NoMetadataComputer(),
        )
        node = next(ns.search.bfs(g)).program
        self.assertEqual(
            node,
            ns.SExpression(
                symbol="+",
                children=(
                    ns.SExpression(
                        symbol="+",
                        children=(
                            ns.SExpression(
                                symbol="+",
                                children=(
                                    ns.SExpression(symbol="1", children=()),
                                    ns.SExpression(symbol="1", children=()),
                                ),
                            ),
                            ns.SExpression(symbol="1", children=()),
                        ),
                    ),
                    ns.SExpression(symbol="1", children=()),
                ),
            ),
        )

    def test_astar(self):
        g = ns.DSLSearchGraph(
            dsl,
            ns.parse_type("i"),
            ns.ChooseFirst(),
            lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
            metadata_computer=ns.NoMetadataComputer(),
        )

        def cost(x):
            if isinstance(x.program, ns.SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        node = next(ns.search.astar(g, cost)).program
        print(node)
        self.assertEqual(
            node,
            ns.SExpression(
                symbol="+",
                children=(
                    ns.SExpression(symbol="1", children=()),
                    ns.SExpression(
                        symbol="+",
                        children=(
                            ns.SExpression(
                                symbol="+",
                                children=(
                                    ns.SExpression(symbol="1", children=()),
                                    ns.SExpression(symbol="1", children=()),
                                ),
                            ),
                            ns.SExpression(symbol="1", children=()),
                        ),
                    ),
                ),
            ),
        )
