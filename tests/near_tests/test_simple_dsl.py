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

import pytest
from neurosym.near.search_graph import near_graph
from neurosym.programs.s_expression import SExpression

from neurosym.search.bfs import bfs
from neurosym.search.bounded_astar import bounded_astar

from neurosym.near.dsls.simple_differentiable_dsl import differentiable_arith_dsl
import torch

from neurosym.types.type_string_repr import TypeDefiner, parse_type


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
        dsl = differentiable_arith_dsl(input_dim)
        g = near_graph(
            dsl,
            parse_type("f"),
            is_goal=lambda x: dsl.compute(dsl.initialize(x.program)) == 4,
        )
        node = next(bfs(g)).program
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

    def test_simple_dsl_astar(self):
        """
        input: x: tensor[input_dim]
        goal: f(x) = [4] * input_dim
        expected node.program:
            (ones input_dim)
        """
        self.maxDiff = None
        input_size = 10
        dsl = differentiable_arith_dsl(input_size)
        fours = torch.full((input_size,), 4.0)

        def checker(x):
            x = x.program
            xx = dsl.compute(dsl.initialize(x))
            if isinstance(xx, torch.Tensor):
                return torch.all(torch.eq(xx, fours))
            else:
                return False

        g = near_graph(dsl, parse_type("{f, 10}"), is_goal=checker)

        def cost(x):
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        node = next(bounded_astar(g, cost, max_depth=7)).program
        self.assertEqual(node.children[0], SExpression(symbol="ones", children=()))

    def test_simple_dsl_enumerate(self):
        """
        Enumerate all programs in dsl upto fixed depth. This test case makes
        sure all DSL combinations upto a fixed depth are valid.
        """
        self.maxDiff = None
        input_dim = 10
        output_dim = 4
        max_depth = 5
        t = TypeDefiner(L=input_dim, O=output_dim)
        t.typedef("fL", "{f, $L}")
        t.typedef("fO", "{f, $O}")

        dsl = differentiable_arith_dsl(input_dim)

        def checker(x):
            """Initialize and return True always"""
            x = x.program
            xx = dsl.compute(dsl.initialize(x))
            print(xx)
            return True

        g = near_graph(dsl, parse_type("{f, 10}"), is_goal=checker)

        def cost(x):
            if isinstance(x.program, SExpression) and x.program.children:
                return len(str(x.program.children[0]))
            return 0

        # should not raise StopIteration.
        for _ in bounded_astar(g, cost, max_depth=max_depth):
            pass
