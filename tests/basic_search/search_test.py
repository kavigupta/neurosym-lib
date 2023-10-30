"""
Just checks that the package can be imported
"""

import unittest

from neurosym.examples.basic_arith import basic_arith_dsl
from neurosym.programs.s_expression import SExpression
from neurosym.search.astar import astar
from neurosym.search.bfs import bfs
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
        node = next(bfs(g)).program
        self.assertEqual(
            node,
            SExpression(
                symbol="+",
                children=(
                    SExpression(
                        symbol="+",
                        children=(
                            SExpression(
                                symbol="+",
                                children=(
                                    SExpression(symbol="1", children=()),
                                    SExpression(symbol="1", children=()),
                                ),
                            ),
                            SExpression(symbol="1", children=()),
                        ),
                    ),
                    SExpression(symbol="1", children=()),
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
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        node = next(astar(g, cost)).program
        print(node)
        self.assertEqual(
            node,
            SExpression(
                symbol="+",
                children=(
                    SExpression(symbol="1", children=()),
                    SExpression(
                        symbol="+",
                        children=(
                            SExpression(
                                symbol="+",
                                children=(
                                    SExpression(symbol="1", children=()),
                                    SExpression(symbol="1", children=()),
                                ),
                            ),
                            SExpression(symbol="1", children=()),
                        ),
                    ),
                ),
            ),
        )
