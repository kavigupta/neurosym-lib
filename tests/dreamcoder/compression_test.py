import unittest
from functools import lru_cache

import numpy as np

import neurosym as ns

arith_dsl = ns.examples.mutable_arith_combinators_dsl

out_t = ns.parse_type("(i) -> i")


@lru_cache(maxsize=None)
def corpus():
    fam = ns.BigramProgramDistributionFamily(arith_dsl, valid_root_types=[out_t])
    dist = fam.uniform()
    return sorted(
        {
            fam.sample(dist, np.random.RandomState(i), depth_limit=20)
            for i in range(100)
        },
        key=str,
    )


class BasicDSLTest(unittest.TestCase):
    def assertDSL(self, dsl, expected):
        dsl = "\n".join(
            sorted([line.strip() for line in dsl.split("\n") if line.strip()])
        )
        expected = "\n".join(
            sorted([line.strip() for line in expected.split("\n") if line.strip()])
        )
        print(dsl)
        self.maxDiff = None
        self.assertEqual(dsl, expected)

    def test_basic_expand(self):
        self.assertDSL(
            arith_dsl.render(),
            """
            * :: (i -> i, i -> i) -> i -> i
            + :: (i -> i, i -> i) -> i -> i
            1 :: () -> i -> i
            count[counter] :: () -> i -> i
            even? :: (i -> i) -> i -> i
            ite :: (i -> i, i -> i, i -> i) -> i -> i
            x :: () -> i -> i
            """,
        )

    def test_independent_mutability(self):
        prog = ns.parse_s_expression("(ite (even? (x)) (count) (count))")
        fn_1 = arith_dsl.compute(arith_dsl.initialize(prog))
        fn_2 = arith_dsl.compute(arith_dsl.initialize(prog))
        self.assertEqual([fn_1(2), fn_1(4), fn_1(8)], [1, 2, 3])
        self.assertEqual([fn_1(2), fn_1(4), fn_1(8)], [4, 5, 6])
        # fn_2 is independent of fn_1
        self.assertEqual([fn_2(2), fn_2(4), fn_2(8)], [1, 2, 3])
        self.assertEqual([fn_1(1), fn_1(2), fn_1(3)], [1, 7, 2])
        self.assertEqual([fn_1(1), fn_1(2), fn_1(3)], [3, 8, 4])


class CompressionTest(unittest.TestCase):
    def fuzzy_check_fn_same(self, fn_1, fn_2):
        inputs = np.random.RandomState(0).randint(100, size=100)
        outputs_1 = [fn_1(x) for x in inputs]
        outputs_2 = [fn_2(x) for x in inputs]
        self.assertEqual(outputs_1, outputs_2)

    def test_single_step(self):
        dsl2, rewritten = ns.compression.single_step_compression(arith_dsl, corpus())
        self.assertEqual(len(rewritten), len(corpus()))
        for orig, rewr in zip(corpus(), rewritten):
            self.fuzzy_check_fn_same(
                arith_dsl.compute(arith_dsl.initialize(orig)),
                dsl2.compute(dsl2.initialize(rewr)),
            )

    def test_multi_step(self):
        dsl2, rewritten = ns.compression.multi_step_compression(arith_dsl, corpus(), 5)
        self.assertEqual(len(rewritten), len(corpus()))
        for orig, rewr in zip(corpus(), rewritten):
            self.fuzzy_check_fn_same(
                arith_dsl.compute(arith_dsl.initialize(orig)),
                dsl2.compute(dsl2.initialize(rewr)),
            )


class BasicProcessDSL(unittest.TestCase):
    def assertDSL(self, dsl, expected):
        dsl = "\n".join(
            sorted([line.strip() for line in dsl.split("\n") if line.strip()])
        )
        expected = "\n".join(
            sorted([line.strip() for line in expected.split("\n") if line.strip()])
        )
        print(dsl)
        self.maxDiff = None
        self.assertEqual(dsl, expected)

    def test_basic_expand(self):
        self.assertDSL(
            ns.examples.basic_arith_dsl(True).render(),
            """
            $0_0 :: V<i@0>
            $1_0 :: V<i@1>
            $2_0 :: V<i@2>
            + :: (i, i) -> i
            1 :: () -> i
            lam_0 :: L<#body|i;i> -> (i, i) -> #body
            lam_1 :: L<#body|i> -> i -> #body
            """,
        )

    def setUp(self):
        self.dsl = ns.examples.basic_arith_dsl(True)
        print(ns.examples.basic_arith_dsl(True).render())
        self.fn_code = "(lam_0 (+ ($1_0) ($0_0)))"

    def test_compute(self):
        fn = self.dsl.compute(self.dsl.initialize(ns.parse_s_expression(self.fn_code)))
        self.assertEqual(fn(1, 2), 3)
        self.assertEqual(fn(3, 4), 7)
        self.assertEqual(fn(1000, -24), 976)

    def test_basic_compress(self):
        code = [
            "(lam_0 (+ (+ (1) (1)) ($0_0)))",
            "(lam_0 (+ (1) ($0_0)))",
        ]
        code = [ns.parse_s_expression(x) for x in code]
        dsl2, rewritten = ns.compression.single_step_compression(self.dsl, code)
        self.assertEqual(len(rewritten), len(code))
        self.assertEqual(
            [ns.render_s_expression(x) for x in rewritten],
            [
                "(__10 (+ (1) (1)))",
                "(__10 (1))",
            ],
        )
        self.assertEqual(
            dsl2.productions[-1].render().strip(),
            "__10 :: i -> (i, i) -> i = (lam-abstr (#0) (lam_0 (+ #0 ($0_0))))",
        )

    def test_multi_argument(self):
        code = [
            "(lam_0 (+ (1) ($1_0)))",
            "(lam_0 (+ (1) (2)))",
        ]
        code = [ns.parse_s_expression(x) for x in code]
        dsl2, rewritten = ns.compression.single_step_compression(self.dsl, code)
        self.assertEqual(len(dsl2.productions), len(self.dsl.productions))
        self.assertEqual(rewritten, code)

    def test_compress_yoinking_variables(self):
        code = [
            "(lam_1 (lam_1 (+ (1) ($1_0))))",
            "(lam_1 (+ (1) (2)))",
        ]
        code = [ns.parse_s_expression(x) for x in code]
        dsl2, rewritten = ns.compression.single_step_compression(self.dsl, code)
        self.assertEqual(
            dsl2.productions[-1].render().strip(),
            "__10 :: i -> i -> i = (lam-abstr (#0) (lam_1 (+ (1) #0)))",
        )
        self.assertEqual(len(rewritten), len(code))
        self.assertEqual(
            [ns.render_s_expression(x) for x in rewritten],
            [
                "(lam_1 (__10 ($0_0)))",
                "(__10 (2))",
            ],
        )
