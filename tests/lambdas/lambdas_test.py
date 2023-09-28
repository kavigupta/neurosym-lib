import itertools
import unittest


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

ba_dsl = basic_arith_dsl(True)


class TestPruning(unittest.TestCase):
    def test_output(self):
        expected = """
            + :: (i, i) -> i
            1 :: () -> i
        lam_0 :: L<i -> i|i;i> -> (i, i) -> i -> i
        lam_1 :: L<i|i;i> -> (i, i) -> i
        lam_2 :: L<i -> i|i> -> i -> i -> i
        lam_3 :: L<i|i> -> i -> i
        $0_0 :: V<i@0>
        $1_1 :: V<i@1>
        $2_2 :: V<i@2>
        """
        actual = ba_dsl.render()
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )


class TestEvaluate(unittest.TestCase):
    def evaluate(self, code):
        return ba_dsl.compute(
            ba_dsl.initialize(parse_s_expression(code, should_not_be_leaf=set()))
        )

    def test_constant(self):
        fn = self.evaluate("(lam_3 (1))")
        for i in range(10):
            self.assertEqual(fn(i), 1)

    def test_identity(self):
        fn = self.evaluate("(lam_3 ($0_0))")
        for i in range(10):
            self.assertEqual(fn(i), i)

    def test_add_1(self):
        fn = self.evaluate("(lam_3 (+ (1) ($0_0)))")
        for i in range(10):
            self.assertEqual(fn(i), i + 1)

    def test_double_constant(self):
        fn = self.evaluate("(lam_2 (lam_3 (1)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), 1)

    def test_get_inner_var(self):
        fn = self.evaluate("(lam_2 (lam_3 ($0_0)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), j)

    def test_get_outer_var(self):
        fn = self.evaluate("(lam_2 (lam_3 ($1_1)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i)

    def test_sub_curried(self):
        fn = self.evaluate("(lam_2 (lam_3 (+ (+ ($0_0) ($0_0)) ($1_1))))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i + 2 * j)

    def test_two_arg_const(self):
        fn = self.evaluate("(lam_1 (1))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), 1)

    def test_two_arg_sub(self):
        # note that the order of arguments is reversed here
        fn = self.evaluate("(lam_1 (+ (+ ($0_0) ($0_0)) ($1_1)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), 2 * i + j)

    def test_three_arg_sub(self):
        # note that the order of arguments is reversed here
        fn = self.evaluate(
            "(lam_0 (lam_3 (+ (+ ($2_2) (+ ($2_2) ($2_2))) (+ (+ ($0_0) ($0_0)) ($1_1)))))"
        )
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.assertEqual(fn(i, j)(k), k * 2 + i + j * 3)


class TestEnumerate(unittest.TestCase):
    def assertEnumerateType(self, typ, expected, filt=lambda x: True):
        g = DSLSearchGraph(
            ba_dsl,
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
                "(lam_3 (1))",
                "(lam_3 ($0_0))",
                "(lam_3 (+ (1) (1)))",
                "(lam_3 (+ (1) ($0_0)))",
                "(lam_3 (+ ($0_0) (1)))",
                "(lam_3 (+ ($0_0) ($0_0)))",
                "(lam_3 (+ (+ (1) (1)) (1)))",
                "(lam_3 (+ (+ (1) (1)) ($0_0)))",
                "(lam_3 (+ (+ (1) ($0_0)) (1)))",
                "(lam_3 (+ (+ (1) ($0_0)) ($0_0)))",
            ],
        )

    def test_curried(self):
        self.assertEnumerateType(
            "i -> i -> i",
            [
                "(lam_2 (lam_3 (1)))",
                "(lam_2 (lam_3 ($0_0)))",
                "(lam_2 (lam_3 ($1_1)))",
                "(lam_2 (lam_3 (+ (1) (1))))",
                "(lam_2 (lam_3 (+ (1) ($0_0))))",
                "(lam_2 (lam_3 (+ (1) ($1_1))))",
                "(lam_2 (lam_3 (+ ($0_0) (1))))",
                "(lam_2 (lam_3 (+ ($0_0) ($0_0))))",
                "(lam_2 (lam_3 (+ ($0_0) ($1_1))))",
                "(lam_2 (lam_3 (+ ($1_1) (1))))",
            ],
        )

    def test_2arg(self):
        self.assertEnumerateType(
            "(i, i) -> i",
            [
                "(lam_1 (1))",
                "(lam_1 ($0_0))",
                "(lam_1 ($1_1))",
                "(lam_1 (+ (1) (1)))",
                "(lam_1 (+ (1) ($0_0)))",
                "(lam_1 (+ (1) ($1_1)))",
                "(lam_1 (+ ($0_0) (1)))",
                "(lam_1 (+ ($0_0) ($0_0)))",
                "(lam_1 (+ ($0_0) ($1_1)))",
                "(lam_1 (+ ($1_1) (1)))",
            ],
        )

    def test_3arg_mix_curried(self):
        self.assertEnumerateType(
            "(i, i) -> i -> i",
            [
                "(lam_0 (lam_3 ($0_0)))",
                "(lam_0 (lam_3 ($1_1)))",
                "(lam_0 (lam_3 ($2_2)))",
                "(lam_0 (lam_3 (+ (1) ($0_0))))",
                "(lam_0 (lam_3 (+ (1) ($1_1))))",
                "(lam_0 (lam_3 (+ (1) ($2_2))))",
                "(lam_0 (lam_3 (+ ($0_0) (1))))",
                "(lam_0 (lam_3 (+ ($0_0) ($0_0))))",
                "(lam_0 (lam_3 (+ ($0_0) ($1_1))))",
                "(lam_0 (lam_3 (+ ($0_0) ($2_2))))",
            ],
            filt=lambda x: "$" in str(x.program),
        )
