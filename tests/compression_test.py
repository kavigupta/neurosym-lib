from functools import lru_cache
import unittest

import numpy as np
from neurosym.compression.process_abstraction import (
    multi_step_compression,
    single_step_compression,
)

from neurosym.dsl.pcfg import PCFGPattern
from neurosym.examples.basic_arith import basic_arith_dsl
from neurosym.examples.mutable_arith_combinators import mutable_arith_combinators
from neurosym.programs.s_expression_render import (
    parse_s_expression,
    render_s_expression,
)
from neurosym.types.type_string_repr import parse_type

out_t = parse_type("(i) -> i")


@lru_cache(maxsize=None)
def corpus():
    return sorted(
        {
            PCFGPattern.of(mutable_arith_combinators, out_t)
            .uniform()
            .sample(np.random.RandomState(i), 20)
            for i in range(100)
        },
        key=str,
    )


class BasicDSLTest(unittest.TestCase):
    def test_independent_mutability(self):
        prog = parse_s_expression("(ite (even? (x)) (count) (count))", set())
        fn_1 = mutable_arith_combinators.compute(
            mutable_arith_combinators.initialize(prog)
        )
        fn_2 = mutable_arith_combinators.compute(
            mutable_arith_combinators.initialize(prog)
        )
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
        dsl2, rewritten = single_step_compression(mutable_arith_combinators, corpus())
        self.assertEqual(len(rewritten), len(corpus()))
        for orig, rewr in zip(corpus(), rewritten):
            self.fuzzy_check_fn_same(
                mutable_arith_combinators.compute(
                    mutable_arith_combinators.initialize(orig)
                ),
                dsl2.compute(dsl2.initialize(rewr)),
            )

    def test_multi_step(self):
        dsl2, rewritten = multi_step_compression(mutable_arith_combinators, corpus(), 5)
        self.assertEqual(len(rewritten), len(corpus()))
        for orig, rewr in zip(corpus(), rewritten):
            self.fuzzy_check_fn_same(
                mutable_arith_combinators.compute(
                    mutable_arith_combinators.initialize(orig)
                ),
                dsl2.compute(dsl2.initialize(rewr)),
            )


class BasicProcessDSL(unittest.TestCase):
    def setUp(self):
        self.dsl = basic_arith_dsl(True)
        print(basic_arith_dsl(True).render())
        self.fn_code = "(lam_0 (+ ($1_0) ($0_0)))"

    def test_compute(self):
        fn = self.dsl.compute(
            self.dsl.initialize(parse_s_expression(self.fn_code, set()))
        )
        self.assertEqual(fn(1, 2), 3)
        self.assertEqual(fn(3, 4), 7)
        self.assertEqual(fn(1000, -24), 976)

    def test_basic_compress(self):
        code = [
            "(lam_0 (+ (+ (1) (1)) ($0_0)))",
            "(lam_0 (+ (1) ($0_0)))",
        ]
        code = [parse_s_expression(x, set()) for x in code]
        dsl2, rewritten = single_step_compression(self.dsl, code)
        self.assertEqual(len(rewritten), len(code))
        self.assertEqual(
            [render_s_expression(x, for_stitch=False) for x in rewritten],
            [
                "(__10 (+ (1) (1)))",
                "(__10 (1))",
            ],
        )
        self.assertEqual(
            dsl2.productions[-1].render().strip(),
            "__10 :: i -> (i, i) -> i = (lam (#0) (lam_0 (+ #0 ($0_0))))",
        )
