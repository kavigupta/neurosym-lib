import itertools
import unittest

import neurosym as ns


def make_compute_dsl():
    dslf = ns.DSLFactory()
    dslf.concrete("+", "(i, i) -> i", lambda x, y: x + y)
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("double", "(i) -> i", lambda x: x * 2)
    dslf.concrete("triple", "(i) -> i", lambda x: x * 3)
    dslf.lambdas()
    dslf.prune_to("i", "i -> i", "i -> i -> i", "(i, i) -> i", "(i, i) -> i -> i")
    return dslf.finalize()


compute_dsl = make_compute_dsl()


class TestEvaluate(unittest.TestCase):
    def test_show_dsl(self):
        expected = """
            + :: (i, i) -> i
            1 :: () -> i
        double :: i -> i
        triple :: i -> i
        lam_0 :: L<#body|i;i> -> (i, i) -> #body
        lam_1 :: L<#body|i> -> i -> #body
        $0_0 :: V<i@0>
        $1_0 :: V<i@1>
        $2_0 :: V<i@2>
        """
        actual = compute_dsl.render()
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def evaluate(self, code):
        return compute_dsl.compute(compute_dsl.initialize(ns.parse_s_expression(code)))

    def test_constant(self):
        fn = self.evaluate("(lam_1 (1))")
        for i in range(10):
            self.assertEqual(fn(i), 1)

    def test_identity(self):
        fn = self.evaluate("(lam_1 ($0_0))")
        for i in range(10):
            self.assertEqual(fn(i), i)

    def test_add_1(self):
        fn = self.evaluate("(lam_1 (+ (1) ($0_0)))")
        for i in range(10):
            self.assertEqual(fn(i), i + 1)

    def test_double_constant(self):
        fn = self.evaluate("(lam_1 (lam_1 (1)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), 1)

    def test_get_inner_var(self):
        fn = self.evaluate("(lam_1 (lam_1 ($0_0)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), j)

    def test_get_outer_var(self):
        fn = self.evaluate("(lam_1 (lam_1 ($1_0)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i)

    def test_sub_curried(self):
        fn = self.evaluate("(lam_1 (lam_1 (+ (double ($0_0)) ($1_0))))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i)(j), i + 2 * j)

    def test_two_arg_const(self):
        fn = self.evaluate("(lam_0 (1))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), 1)

    def test_two_arg_sub(self):
        fn = self.evaluate("(lam_0 (+ (double ($0_0)) ($1_0)))")
        for i in range(10):
            for j in range(10):
                self.assertEqual(fn(i, j), j * 2 + i)

    def test_three_arg_sub(self):
        fn = self.evaluate(
            """
            (lam_0 (lam_1
                (+
                    (+
                        (triple ($0_0))
                        (double ($1_0)))
                    ($2_0))))
            """
        )
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    self.assertEqual(fn(i, j)(k), k * 3 + j * 2 + i)


class TestEnumerateBasicArithmetic(unittest.TestCase):
    def test_show_dsl(self):
        self.maxDiff = None
        expected = """
            + :: (i, i) -> i
            1 :: () -> i
        lam_0 :: L<#body|i;i> -> (i, i) -> #body
        lam_1 :: L<#body|i> -> i -> #body
        $0_0 :: V<i@0>
        $1_0 :: V<i@1>
        $2_0 :: V<i@2>
        """
        actual = ns.examples.basic_arith_dsl(True).render()
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def assertEnumerateType(self, typ, expected, filt=lambda x: True):
        g = ns.DSLSearchGraph(
            ns.examples.basic_arith_dsl(True),
            ns.parse_type(typ),
            ns.ChooseFirst(),
            filt,
            metadata_computer=ns.NoMetadataComputer(),
        )

        res = [
            ns.render_s_expression(prog.program)
            for prog in itertools.islice(ns.search.bfs(g, 1000), 10)
        ]

        print(res)

        self.assertEqual(res, expected)

    def test_single(self):
        self.assertEnumerateType(
            "i -> i",
            [
                "(lam_1 (1))",
                "(lam_1 ($0_0))",
                "(lam_1 (+ (1) (1)))",
                "(lam_1 (+ (1) ($0_0)))",
                "(lam_1 (+ ($0_0) (1)))",
                "(lam_1 (+ ($0_0) ($0_0)))",
                "(lam_1 (+ (+ (1) (1)) (1)))",
                "(lam_1 (+ (+ (1) (1)) ($0_0)))",
                "(lam_1 (+ (+ (1) ($0_0)) (1)))",
                "(lam_1 (+ (+ (1) ($0_0)) ($0_0)))",
            ],
        )

    def test_curried(self):
        self.assertEnumerateType(
            "i -> i -> i",
            [
                "(lam_1 (lam_1 (1)))",
                "(lam_1 (lam_1 ($0_0)))",
                "(lam_1 (lam_1 ($1_0)))",
                "(lam_1 (lam_1 (+ (1) (1))))",
                "(lam_1 (lam_1 (+ (1) ($0_0))))",
                "(lam_1 (lam_1 (+ (1) ($1_0))))",
                "(lam_1 (lam_1 (+ ($0_0) (1))))",
                "(lam_1 (lam_1 (+ ($0_0) ($0_0))))",
                "(lam_1 (lam_1 (+ ($0_0) ($1_0))))",
                "(lam_1 (lam_1 (+ ($1_0) (1))))",
            ],
        )

    def test_2arg(self):
        self.assertEnumerateType(
            "(i, i) -> i",
            [
                "(lam_0 (1))",
                "(lam_0 ($0_0))",
                "(lam_0 ($1_0))",
                "(lam_0 (+ (1) (1)))",
                "(lam_0 (+ (1) ($0_0)))",
                "(lam_0 (+ (1) ($1_0)))",
                "(lam_0 (+ ($0_0) (1)))",
                "(lam_0 (+ ($0_0) ($0_0)))",
                "(lam_0 (+ ($0_0) ($1_0)))",
                "(lam_0 (+ ($1_0) (1)))",
            ],
        )

    def test_3arg_mix_curried(self):
        self.assertEnumerateType(
            "(i, i) -> i -> i",
            [
                "(lam_0 (lam_1 ($0_0)))",
                "(lam_0 (lam_1 ($1_0)))",
                "(lam_0 (lam_1 ($2_0)))",
                "(lam_0 (lam_1 (+ (1) ($0_0))))",
                "(lam_0 (lam_1 (+ (1) ($1_0))))",
                "(lam_0 (lam_1 (+ (1) ($2_0))))",
                "(lam_0 (lam_1 (+ ($0_0) (1))))",
                "(lam_0 (lam_1 (+ ($0_0) ($0_0))))",
                "(lam_0 (lam_1 (+ ($0_0) ($1_0))))",
                "(lam_0 (lam_1 (+ ($0_0) ($2_0))))",
            ],
            filt=lambda x: "$" in str(x.program),
        )


def make_varied_type_dsl():
    dslf = ns.DSLFactory()
    dslf.concrete("^", "(f, i) -> f", lambda x, y: x**y)
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("1f", "() -> f", lambda x: 1.0)
    dslf.concrete("double", "(i) -> i", lambda x: x * 2)
    dslf.lambdas()
    dslf.prune_to("f", "f -> f", "(f, i) -> f", "(f, i) -> i")
    return dslf.finalize()


class TestVariedTypes(unittest.TestCase):
    def test_show_dsl(self):
        self.maxDiff = None
        expected = """
              ^ :: (f, i) -> f
              1 :: () -> i
             1f :: () -> f
         double :: i -> i
          lam_0 :: L<#body|f;i> -> (f, i) -> #body
          lam_1 :: L<#body|f> -> f -> #body
           $0_0 :: V<f@0>
           $1_0 :: V<f@1>
           $0_1 :: V<i@0>
        """
        actual = make_varied_type_dsl().render()
        print(actual)
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def assertEnumerateType(self, typ, expected, filt=lambda x: True):
        g = ns.DSLSearchGraph(
            make_varied_type_dsl(),
            ns.parse_type(typ),
            ns.ChooseFirst(),
            filt,
            metadata_computer=ns.NoMetadataComputer(),
        )

        res = [
            ns.render_s_expression(prog.program)
            for prog in itertools.islice(ns.search.bfs(g, 1000), 10)
        ]

        print(res)

        self.assertEqual(res, expected)

    def test_fi(self):
        self.assertEnumerateType(
            "(f, i) -> f",
            [
                "(lam_0 (1f))",
                "(lam_0 ($1_0))",
                "(lam_0 (^ (1f) (1)))",
                "(lam_0 (^ (1f) ($0_1)))",
                "(lam_0 (^ ($1_0) (1)))",
                "(lam_0 (^ ($1_0) ($0_1)))",
                "(lam_0 (^ (1f) (double (1))))",
                "(lam_0 (^ (1f) (double ($0_1))))",
                "(lam_0 (^ ($1_0) (double (1))))",
                "(lam_0 (^ ($1_0) (double ($0_1))))",
            ],
        )
