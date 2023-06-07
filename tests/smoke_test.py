"""
Just checks that the package can be imported
"""

import unittest
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst

from neurosym.examples.basic_arith import basic_arith_dsl, int_type


class TestSmoke(unittest.TestCase):
    def test_bfs(self):
        g = DSLSearchGraph(
            basic_arith_dsl,
            int_type,
            ChooseFirst(),
            lambda x: basic_arith_dsl.compute_on_pytorch(x) == 4,
        )
        node = next(bfs(g))
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
