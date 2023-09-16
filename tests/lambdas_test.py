import itertools
import unittest
from neurosym.dsl.dsl_factory import DSLFactory


from neurosym.examples.basic_arith import basic_arith_dsl
from neurosym.search.bfs import bfs
from neurosym.search_graph.dsl_search_graph import DSLSearchGraph
from neurosym.search_graph.hole_set_chooser import ChooseFirst
from neurosym.search_graph.metadata_computer import NoMetadataComputer
from neurosym.types.type_string_repr import parse_type
from neurosym.programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
)

dsl = basic_arith_dsl(True)


class TestEvaluate(unittest.TestCase):
    def evaluate(self, code):
        return dsl.compute(
            dsl.initialize(parse_s_expression(code, should_not_be_leaf=set()))
        )

    def test_constant(self):
        fn = self.evaluate("(lam167 (1))")
        for i in range(10):
            self.assertEqual(fn(i), 1)

    def test_identity(self):
        fn = self.evaluate("(lam167 ($0:16))")
        for i in range(10):
            self.assertEqual(fn(i), i)

    def test_add_1(self):
        fn = self.evaluate("(lam167 (+ (1) ($0:16)))")
        for i in range(10):
            self.assertEqual(fn(i), i + 1)

    def test_double_constant(self):
        fn = self.evaluate("(lam166 (lam167 (1)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), 1)

    def test_get_inner_var(self):
        fn = self.evaluate("(lam166 (lam167 ($0:16)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), j)

    def test_get_outer_var(self):
        fn = self.evaluate("(lam166 (lam167 ($1:17)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i)

    def test_sub_curried(self):
        fn = self.evaluate("(lam166 (lam167 (+ (+ ($0:16) ($0:16)) ($1:17))))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i + 2 * j)

    def test_two_arg_const(self):
        fn = self.evaluate("(lam162 (1))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), 1)

    def test_two_arg_sub(self):
        # note that the order of arguments is reversed here
        fn = self.evaluate("(lam162 (+ (+ ($0:16) ($0:16)) ($1:17)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), 2 * i + j)

    def test_three_arg_sub(self):
        # note that the order of arguments is reversed here
        fn = self.evaluate(
            "(lam161 (lam167 (+ (+ ($2:18) (+ ($2:18) ($2:18))) (+ (+ ($0:16) ($0:16)) ($1:17)))))"
        )
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.assertEqual(fn(i, j)(k), k * 2 + i + j * 3)


class TestEnumerate(unittest.TestCase):
    def assertEnumerateType(self, typ, expected, filt=lambda x: True):
        g = DSLSearchGraph(
            dsl,
            parse_type(typ),
            ChooseFirst(),
            filt,
            metadata_computer=NoMetadataComputer(),
        )

        res = [
            render_s_expression(prog.program, False)
            for prog in itertools.islice(bfs(g, 1000), 10)
        ]

        print(res)

        self.assertEqual(res, expected)

    def test_single(self):
        self.assertEnumerateType(
            "i -> i",
            [
                "(lam167 (1))",
                "(lam167 ($0:16))",
                "(lam167 (+ (1) (1)))",
                "(lam167 (+ (1) ($0:16)))",
                "(lam167 (+ ($0:16) (1)))",
                "(lam167 (+ ($0:16) ($0:16)))",
                "(lam167 (+ (+ (1) (1)) (1)))",
                "(lam167 (+ (+ (1) (1)) ($0:16)))",
                "(lam167 (+ (+ (1) ($0:16)) (1)))",
                "(lam167 (+ (+ (1) ($0:16)) ($0:16)))",
            ],
        )

    def test_curried(self):
        self.assertEnumerateType(
            "i -> i -> i",
            [
                "(lam166 (lam167 (1)))",
                "(lam166 (lam167 ($0:16)))",
                "(lam166 (lam167 ($1:17)))",
                "(lam166 (lam167 (+ (1) (1))))",
                "(lam166 (lam167 (+ (1) ($0:16))))",
                "(lam166 (lam167 (+ (1) ($1:17))))",
                "(lam166 (lam167 (+ ($0:16) (1))))",
                "(lam166 (lam167 (+ ($0:16) ($0:16))))",
                "(lam166 (lam167 (+ ($0:16) ($1:17))))",
                "(lam166 (lam167 (+ ($1:17) (1))))",
            ],
        )

    def test_2arg(self):
        self.assertEnumerateType(
            "(i, i) -> i",
            [
                "(lam162 (1))",
                "(lam162 ($0:16))",
                "(lam162 ($1:17))",
                "(lam162 (+ (1) (1)))",
                "(lam162 (+ (1) ($0:16)))",
                "(lam162 (+ (1) ($1:17)))",
                "(lam162 (+ ($0:16) (1)))",
                "(lam162 (+ ($0:16) ($0:16)))",
                "(lam162 (+ ($0:16) ($1:17)))",
                "(lam162 (+ ($1:17) (1)))",
            ],
        )

    def test_3arg_mix_curried(self):
        self.assertEnumerateType(
            "(i, i) -> i -> i",
            [
                "(lam161 (lam167 ($0:16)))",
                "(lam161 (lam167 ($1:17)))",
                "(lam161 (lam167 ($2:18)))",
                "(lam161 (lam167 (+ (1) ($0:16))))",
                "(lam161 (lam167 (+ (1) ($1:17))))",
                "(lam161 (lam167 (+ (1) ($2:18))))",
                "(lam161 (lam167 (+ ($0:16) (1))))",
                "(lam161 (lam167 (+ ($0:16) ($0:16))))",
                "(lam161 (lam167 (+ ($0:16) ($1:17))))",
                "(lam161 (lam167 (+ ($0:16) ($2:18))))",
            ],
            filt=lambda x: "$" in str(x.program),
        )
