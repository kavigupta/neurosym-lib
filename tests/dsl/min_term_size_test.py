import unittest

from parameterized import parameterized

import neurosym as ns


class TestMinimalTermSize(unittest.TestCase):

    def instrumentDSL(self, dsl):
        count = 0
        original_expansions_for_type = dsl.expansions_for_type

        def expansions_for_type_instrumented(typ):
            nonlocal count
            result = original_expansions_for_type(typ)
            count += 1
            return result

        dsl.expansions_for_type = expansions_for_type_instrumented
        return lambda: count

    def basicDSL(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> literal", lambda x: x)
        dslf.concrete("id", "literal -> literal", lambda x: x)
        dslf.concrete("+", "(literal, literal) -> sum", lambda x, y: x + y)
        dslf.concrete("*", "(sum, sum) -> product", lambda x, y: x * y)
        return dslf.finalize()

    def test_basic_arith(self):
        dsl = self.basicDSL()
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type("literal"), ns.Environment.empty())
            ),
            1,  # (1)
        )
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type("sum"), ns.Environment.empty())
            ),
            3,  # (+ (1) (1))
        )
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type("product"), ns.Environment.empty())
            ),
            7,  # (* (+ (1) (1)) (+ (1) (1)))
        )

    def test_basic_arith_symbol_weights(self):
        dsl = self.basicDSL()
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(
                    ns.parse_type("product"), ns.Environment.empty()
                ),
                {"+": 0.1, "*": 10},
            ),
            14.2,  # (* (+ (1) (1)) (+ (1) (1)))
        )

    num_nesting = 100

    def heavilyNestedBinaryDSL(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> t0", lambda x: x)
        for i in range(1, self.num_nesting):
            for c in "fg":
                dslf.concrete(
                    f"{c}_{i}", f"(t{i-1}, t{i-1}) -> t{i}", lambda x, y: x + y
                )
        return dslf.finalize()

    @parameterized.expand([(i,) for i in range(1, num_nesting)])
    def test_heavily_nested(self, i):
        dsl = self.heavilyNestedBinaryDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type(f"t{i}"), ns.Environment.empty())
            ),
            2 ** (i + 1) - 1,
        )
        self.assertLessEqual(count(), 0.5 * i**2 + 20)

    num_linear_nesting = 1000

    def heavilyHeavilyNestedLinearDSL(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> t0", lambda x: x)
        for i in range(1, self.num_linear_nesting):
            dslf.concrete(f"f_{i}", f"(t{i-1}) -> t{i}", lambda x: x)
        return dslf.finalize()

    @parameterized.expand([(i,) for i in range(1, num_linear_nesting, 100)])
    def test_heavily_heavily_nested_linear(self, i):
        dsl = self.heavilyHeavilyNestedLinearDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type(f"t{i}"), ns.Environment.empty())
            ),
            i + 1,
        )
        self.assertLessEqual(count(), 3 * i + 10)

    def lambdasDSL(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "a -> literal", lambda x: x)
        dslf.concrete("id", "literal -> literal", lambda x: x)
        dslf.concrete("+", "(literal, literal) -> sum", lambda x, y: x + y)
        dslf.concrete("*", "(sum, sum) -> product", lambda x, y: x * y)
        dslf.lambdas()
        return dslf.finalize()

    def test_lambdas_basic(self):
        dsl = self.lambdasDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(
                    ns.parse_type("a -> literal"), ns.Environment.empty()
                )
            ),
            1 + 2,  # (lam (1 ($0)))
        )

        self.assertLessEqual(count(), 10)

    def test_lambdas_more_nesting(self):
        dsl = self.lambdasDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(
                    ns.parse_type("a -> sum"), ns.Environment.empty()
                )
            ),
            1 + 1 + 2 * 2,  # (lam (+ (1 ($0)) (1 ($0))))
        )

        self.assertLessEqual(count(), 20)

    def test_lambdas_even_more_nesting(self):
        dsl = self.lambdasDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(
                    ns.parse_type("a -> product"), ns.Environment.empty()
                )
            ),
            1 + 3 + 2 * 4,  # (lam (* (+ (1 ($0)) (1 ($0))) (+ (1 ($0)) (1 ($0))))
        )

        self.assertLessEqual(count(), 20)

    def test_impossible_type(self):
        dsl = self.lambdasDSL()
        count = self.instrumentDSL(dsl)
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(
                    ns.parse_type("product -> a"), ns.Environment.empty()
                )
            ),
            float("inf"),  # (lam (* (+ (1 ($0)) (1 ($0))) (+ (1 ($0)) (1 ($0))))
        )

        self.assertLessEqual(count(), 5)

    def choiceNestedDSL(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1a", "() -> a0", lambda x: x)
        dslf.concrete("1b", "() -> b0", lambda x: x)
        for i in range(1, self.num_nesting):
            for sym in "ab":
                dslf.concrete(
                    f"{sym}_{i}",
                    f"({sym}{i-1}, {sym}{i-1}) -> {sym}{i}",
                    lambda x, y: x + y,
                )
        dslf.concrete("done_a", f"a{self.num_nesting - 1} -> t", lambda x: x)
        dslf.concrete("done_b", f"b{self.num_nesting - 1} -> t", lambda x: x)
        return dslf.finalize()

    def test_choice_heavily_nested(self):
        depth = self.num_nesting - 1
        dsl = self.choiceNestedDSL()
        print(dsl.render())
        count = self.instrumentDSL(dsl)
        symbol_costs = {}
        # we would like to force the model to pick a even though
        # it is initially more unpleasant to use. So we give the last
        # symbol in b a very high cost.

        a_unpleasant = 2
        b_bad = 10

        for i in range(self.num_nesting):
            symbol_costs[f"a_{i}"] = a_unpleasant
            symbol_costs[f"b_{i}"] = 1
        symbol_costs["a_0"] = 1
        symbol_costs["b_1"] = b_bad
        self.assertEqual(
            dsl.minimal_term_size_for_type(
                ns.TypeWithEnvironment(ns.parse_type("t"), ns.Environment.empty()),
                symbol_costs,
            ),
            # done_a
            1
            # a_{depth} * 2^0 + a_{depth - 1} * 2^1 + ... + a_1 * 2^{depth - 1}
            + 2 * (2**depth - 1)
            # 1a * 2^depth
            + 2**depth,
        )
        self.assertLessEqual(count(), depth**2 + 20)
