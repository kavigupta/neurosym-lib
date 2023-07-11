"""
NEAR Integration tests.
"""

import unittest
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst

from neurosym.examples.differentiable_arith import differentiable_arith_dsl, float_type


class TestSmoke(unittest.TestCase):
    def test_bfs(self):
        self.maxDiff = None
        dsl = differentiable_arith_dsl(10)
        g = DSLSearchGraph(
            dsl,
            float_type,
            ChooseFirst(),
            lambda x: dsl.compute_on_pytorch(dsl.initialize(x)) == 4,
        )
        node = next(bfs(g))
        self.assertEqual(
            str(node),
            str(
                SExpression(
                    symbol="int_int_add",
                    children=(
                        SExpression(symbol="one", children=()),
                        SExpression(
                            symbol="int_int_add",
                            children=(
                                SExpression(symbol="one", children=()),
                                SExpression(
                                    symbol="int_int_add",
                                    children=(
                                        SExpression(symbol="one", children=()),
                                        SExpression(symbol="one", children=()),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )
        # @TODO: Run this.
