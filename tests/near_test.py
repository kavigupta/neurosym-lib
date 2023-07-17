"""
NEAR Integration tests.
"""

import unittest
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search.bounded_astar import bounded_astar
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst

from neurosym.examples.differentiable_arith import (
    differentiable_arith_dsl,
    float_type,
    list_float_type,
)
import torch


class TestNEAR(unittest.TestCase):
    def test_near_bfs(self):
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

    def test_near_astar(self):
        self.maxDiff = None
        input_size = 10
        dsl = differentiable_arith_dsl(input_size)
        fours = torch.full((input_size,), 4.0)

        def checker(x):
            xx = dsl.compute_on_pytorch(dsl.initialize(x))
            if isinstance(xx, torch.Tensor):
                return torch.all(torch.eq(xx, fours))
            else:
                return False

        g = DSLSearchGraph(
            dsl=dsl,
            target_type=list_float_type,
            hole_set_chooser=ChooseFirst(),
            test_predicate=checker,
        )

        cost = (
            lambda x: len(str(x.children[0]))
            if isinstance(x, SExpression) and x.children
            else 0
        )

        node = next(bounded_astar(g, cost, max_depth=5))
        self.assertEqual(
            node,
            SExpression(
                symbol="Tint_Tint_add",
                children=(
                    SExpression(symbol="ones", children=()),
                    SExpression(
                        symbol="Tint_Tint_add",
                        children=(
                            SExpression(
                                symbol="Tint_Tint_add",
                                children=(
                                    SExpression(symbol="ones", children=()),
                                    SExpression(symbol="ones", children=()),
                                ),
                            ),
                            SExpression(symbol="ones", children=()),
                        ),
                    ),
                ),
            ),
        )
