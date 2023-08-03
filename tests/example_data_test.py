"""
NEAR Integration tests.
"""

import unittest
from neurosym.near.near_graph import near_graph
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search.bounded_astar import bounded_astar

from neurosym.examples.example_rnn_dsl import (
    example_rnn_dsl,
    list_Tfloat_list_outTfloat,
)
import torch

from neurosym.search_graph.metadata_computer import NoMetadataComputer


class TestNEARExample(unittest.TestCase):
    def test_near_astar(self):
        self.maxDiff = None
        input_size = 10
        dsl = example_rnn_dsl(input_size, 4)
        fours = torch.full((input_size,), 4.0)

        # in folder examples/example_rnn, run
        # train_ex_data.npy  train_ex_labels.npy

        def checker(x):
            x = x.program
            xx = dsl.compute_on_pytorch(dsl.initialize(x))
            if isinstance(xx, torch.Tensor):
                return torch.all(torch.eq(xx, fours))
            else:
                return False

        g = near_graph(dsl, list_Tfloat_list_outTfloat, is_goal=checker)

        cost = (
            lambda x: len(str(x.program.children[0]))
            if isinstance(x.program, SExpression) and x.program.children
            else 0
        )
        node = next(bounded_astar(g, cost, max_depth=7)).program
        self.assertEqual(
            node,
            SExpression(
                symbol="Tint_int_add",
                children=(
                    SExpression(symbol="ones", children=()),
                    SExpression(
                        symbol="int_int_add",
                        children=(
                            SExpression(
                                symbol="int_int_add",
                                children=(
                                    SExpression(symbol="one", children=()),
                                    SExpression(symbol="one", children=()),
                                ),
                            ),
                            SExpression(symbol="one", children=()),
                        ),
                    ),
                ),
            ),
        )
