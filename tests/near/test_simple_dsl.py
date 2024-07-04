"""
Test dsl/differentiable_dsl.py with NEAR search graph.

We conduct the following tests:
- Sanity check: We can find a simple program.
- BFS: We can find a simple program with BFS.
- Astar: We can find a simple program with bounded Astar search.
- Enumerate: We can enumerate all programs of a certain size.
NEAR Integration tests.
"""

import unittest

import torch

import neurosym as ns
from neurosym.examples import near

from .utils import assertDSLEnumerable


class TestNEARSimpleDSL(unittest.TestCase):
    def test_simple_dsl_bfs(self):
        """
        input: x: tensor[input_dim]
        goal: f(x) = 4
        expected node.program:
            (int_int_add (one) (int_int_add (one) (int_int_add (one) (one))))
            ie. 1 + (1 + (1 + 1))
        """
        self.maxDiff = None
        input_dim = 10
        dsl = near.differentiable_arith_dsl(input_dim)
        g = near.near_graph(
            dsl,
            ns.parse_type("f"),
            is_goal=lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
        )
        node = next(ns.search.bfs(g)).program
        self.assertEqual(
            str(node),
            str(
                ns.SExpression(
                    symbol="int_int_add",
                    children=(
                        ns.SExpression(symbol="one", children=()),
                        ns.SExpression(
                            symbol="int_int_add",
                            children=(
                                ns.SExpression(symbol="one", children=()),
                                ns.SExpression(
                                    symbol="int_int_add",
                                    children=(
                                        ns.SExpression(symbol="one", children=()),
                                        ns.SExpression(symbol="one", children=()),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
        )

    def test_simple_dsl_astar(self):
        """
        input: x: tensor[input_dim]
        goal: f(x) = [4] * input_dim
        expected node.program:
            (ones input_dim)
        """
        self.maxDiff = None
        input_size = 10
        dsl = near.differentiable_arith_dsl(input_size)
        fours = torch.full((input_size,), 4.0)

        def checker(x):
            x = x.program
            xx = dsl.compute(dsl.initialize(x))
            if isinstance(xx, torch.Tensor):
                return torch.all(torch.eq(xx, fours))
            return False

        max_depth = 7
        g = near.near_graph(
            dsl, ns.parse_type("{f, 10}"), max_depth=max_depth, is_goal=checker
        )

        def cost(x):
            if isinstance(x.program, ns.SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        node = next(
            ns.search.bounded_astar(
                g,
                cost,
                max_depth=max_depth,
            )
        ).program
        self.assertEqual(node.children[0], ns.SExpression(symbol="ones", children=()))

    def test_simple_dsl_enumerate(self):
        """
        Enumerate all programs in dsl upto fixed depth. This test case makes
        sure all DSL combinations upto a fixed depth are valid.
        """
        self.maxDiff = None
        dsl = near.differentiable_arith_dsl(10)

        assertDSLEnumerable(dsl, "$fL")
