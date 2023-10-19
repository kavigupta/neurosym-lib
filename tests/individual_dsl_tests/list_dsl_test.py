import unittest
from neurosym.examples.dreamcoder.list_example import list_dsl
from neurosym.near.near_graph import near_graph
from neurosym.programs.s_expression_render import render_s_expression

from neurosym.search.bfs import bfs

from neurosym.types.type_string_repr import parse_type


class TestListDSL(unittest.TestCase):
    def test_basic_dsl(self):
        self.maxDiff = None
        dsl = list_dsl("[i] -> i")

        def is_goal(x):
            try:
                fn = dsl.compute(dsl.initialize(x.program))
                if fn([1, 2, 3]) != 2:
                    return False
                if fn([0, 7, 3, 5]) != 7:
                    return False
                return True
            except:
                return False

        g = near_graph(
            dsl,
            parse_type("[i] -> i"),
            is_goal=is_goal,
        )
        it = bfs(g)
        node = next(it).program
        self.assertEqual(
            render_s_expression(node, for_stitch=False), "(lam_11 (index (1) ($0_12)))"
        )
