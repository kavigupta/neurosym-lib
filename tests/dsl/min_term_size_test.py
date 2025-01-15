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
