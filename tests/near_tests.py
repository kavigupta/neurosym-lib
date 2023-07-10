"""
NEAR Integration tests.
"""

import unittest
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst

from neurosym.examples.differentiable_arith import differentiable_arith_dsl, float_type, list_float_type, INPUT_LENGHT

class TestSmoke(unittest.TestCase):
    def test_bfs(self):
        g = DSLSearchGraph(
            differentiable_arith_dsl,
            float_type,
            ChooseFirst(),
            lambda x: differentiable_arith_dsl.compute_on_pytorch(x) == 4,
        )
        node = next(bfs(g))
        self.assertEqual(node.value, SExpression("int_int_add", [SExpression("one"), SExpression("one")]))
        #@TODO: Run this.


