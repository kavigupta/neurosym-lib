"""
Just checks that the package can be imported
"""

import unittest

import neurosym as ns

from neurosym.examples.basic_arith import basic_arith_dsl
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst
from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.types.type_string_repr import parse_type

dsl = basic_arith_dsl()


class TestSearch(unittest.TestCase):
    def test_bfs(self):
        g = DSLSearchGraph(
            dsl,
            parse_type("i"),
            ChooseFirst(),
            lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
            metadata_computer=NoMetadataComputer(),
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
        g = DSLSearchGraph(
            dsl,
            parse_type("i"),
            ChooseFirst(),
            lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
            metadata_computer=NoMetadataComputer(),
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
