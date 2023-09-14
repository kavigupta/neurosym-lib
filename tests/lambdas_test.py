import itertools
import unittest


from neurosym.examples.basic_arith import basic_arith_dsl
from neurosym.search.bfs import bfs
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst
from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.types.type_string_repr import parse_type
from neurosym.programs.s_expression_render import render_s_expression


class TestAllRules(unittest.TestCase):
    def test_basic_arith(self):
        g = DSLSearchGraph(
            basic_arith_dsl(True),
            parse_type("i -> i"),
            ChooseFirst(),
            lambda x: True,
            metadata_computer=NoMetadataComputer(),
        )

        res = [
            render_s_expression(prog.program, False)
            for prog in itertools.islice(bfs(g, 1000), 10)
        ]

        self.assertEqual(
            res,
            [
                "(lam1 (1))",
                "(lam1 (+ (1) (1)))",
                "(lam1 (+ (+ (1) (1)) (1)))",
                "(lam1 (+ (1) (+ (1) (1))))",
                "(lam1 (+ (+ (+ (1) (1)) (1)) (1)))",
                "(lam1 (+ (+ (1) (+ (1) (1))) (1)))",
                "(lam1 (+ (+ (1) (1)) (+ (1) (1))))",
                "(lam1 (+ (1) (+ (+ (1) (1)) (1))))",
                "(lam1 (+ (1) (+ (1) (+ (1) (1)))))",
                "(lam1 (+ (+ (+ (+ (1) (1)) (1)) (1)) (1)))",
            ],
        )
