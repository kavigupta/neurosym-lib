import unittest

import neurosym as ns
from neurosym.examples.dreamcoder.list_example import list_dsl
from neurosym.examples.near.search_graph import near_graph
from neurosym.programs.s_expression_render import render_s_expression

ldsl = list_dsl("[i] -> i")


class TestListDSL(unittest.TestCase):
    def test_show_dsl(self):
        expected = """
         lam_11 :: L<#body|[i]> -> [i] -> #body
           $0_3 :: V<[i]@0>
        """
        actual = ldsl.render()
        print(actual)
        self.assertTrue(
            {line.strip() for line in expected.strip().split("\n")}.issubset(
                {line.strip() for line in actual.strip().split("\n")}
            )
        )

    def test_basic_dsl(self):
        self.maxDiff = None
        dsl = ldsl

        def is_goal(x):
            try:
                fn = dsl.compute(dsl.initialize(x.program))
                if fn([1, 2, 3]) != 2:
                    return False
                if fn([0, 7, 3, 5]) != 7:
                    return False
                return True
            except:  # pylint: disable=bare-except
                return False

        g = near_graph(
            dsl,
            ns.parse_type("[i] -> i"),
            is_goal=is_goal,
        )
        it = ns.search.bfs(g)
        node = next(it).program
        self.assertEqual(
            render_s_expression(node, for_stitch=False), "(lam_11 (index (1) ($0_3)))"
        )
