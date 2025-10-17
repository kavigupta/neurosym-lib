# pylint: disable=duplicate-code
# we don't care about duplicate code in test outputs

import unittest

import neurosym as ns

from ..utils import assertDSL


class TestTypeRegresion(unittest.TestCase):
    def assertExpansions(self, actual, expected):
        self.maxDiff = None
        actual = set(ns.render_type(x) for x in actual)
        expected = set(expected)
        print(actual)
        self.assertEqual(actual, expected)

    def test_enumerate_types(self):
        self.assertExpansions(
            ns.bottom_up_enumerate_types(
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[
                    (1, ns.ListType),
                    (2, lambda x, y: ns.ArrowType((x,), y)),
                ],
                max_overall_depth=4,
                max_expansion_steps=3,
            ),
            [
                "(b -> b) -> [[b]]",
                "(b -> b) -> [[i]]",
                "(b -> b) -> [b]",
                "(b -> b) -> [i]",
                "(b -> b) -> b",
                "(b -> b) -> b -> b",
                "(b -> b) -> b -> i",
                "(b -> b) -> i",
                "(b -> b) -> i -> b",
                "(b -> b) -> i -> i",
                "(b -> i) -> [[b]]",
                "(b -> i) -> [[i]]",
                "(b -> i) -> [b]",
                "(b -> i) -> [i]",
                "(b -> i) -> b",
                "(b -> i) -> b -> b",
                "(b -> i) -> b -> i",
                "(b -> i) -> i",
                "(b -> i) -> i -> b",
                "(b -> i) -> i -> i",
                "(i -> b) -> [[b]]",
                "(i -> b) -> [[i]]",
                "(i -> b) -> [b]",
                "(i -> b) -> [i]",
                "(i -> b) -> b",
                "(i -> b) -> b -> b",
                "(i -> b) -> b -> i",
                "(i -> b) -> i",
                "(i -> b) -> i -> b",
                "(i -> b) -> i -> i",
                "(i -> i) -> [[b]]",
                "(i -> i) -> [[i]]",
                "(i -> i) -> [b]",
                "(i -> i) -> [i]",
                "(i -> i) -> b",
                "(i -> i) -> b -> b",
                "(i -> i) -> b -> i",
                "(i -> i) -> i",
                "(i -> i) -> i -> b",
                "(i -> i) -> i -> i",
                "[[[b]]]",
                "[[[i]]]",
                "[[b -> b]]",
                "[[b -> i]]",
                "[[b] -> [b]]",
                "[[b] -> [i]]",
                "[[b] -> b]",
                "[[b] -> i]",
                "[[b]]",
                "[[b]] -> [[b]]",
                "[[b]] -> [[i]]",
                "[[b]] -> [b]",
                "[[b]] -> [i]",
                "[[b]] -> b",
                "[[b]] -> b -> b",
                "[[b]] -> b -> i",
                "[[b]] -> i",
                "[[b]] -> i -> b",
                "[[b]] -> i -> i",
                "[[i -> b]]",
                "[[i -> i]]",
                "[[i] -> [b]]",
                "[[i] -> [i]]",
                "[[i] -> b]",
                "[[i] -> i]",
                "[[i]]",
                "[[i]] -> [[b]]",
                "[[i]] -> [[i]]",
                "[[i]] -> [b]",
                "[[i]] -> [i]",
                "[[i]] -> b",
                "[[i]] -> b -> b",
                "[[i]] -> b -> i",
                "[[i]] -> i",
                "[[i]] -> i -> b",
                "[[i]] -> i -> i",
                "[b -> [b]]",
                "[b -> [i]]",
                "[b -> b]",
                "[b -> i]",
                "[b]",
                "[b] -> [[b]]",
                "[b] -> [[i]]",
                "[b] -> [b]",
                "[b] -> [i]",
                "[b] -> b",
                "[b] -> b -> b",
                "[b] -> b -> i",
                "[b] -> i",
                "[b] -> i -> b",
                "[b] -> i -> i",
                "[i -> [b]]",
                "[i -> [i]]",
                "[i -> b]",
                "[i -> i]",
                "[i]",
                "[i] -> [[b]]",
                "[i] -> [[i]]",
                "[i] -> [b]",
                "[i] -> [i]",
                "[i] -> b",
                "[i] -> b -> b",
                "[i] -> b -> i",
                "[i] -> i",
                "[i] -> i -> b",
                "[i] -> i -> i",
                "b",
                "b -> [[b]]",
                "b -> [[i]]",
                "b -> [b]",
                "b -> [i]",
                "b -> b",
                "b -> b -> b",
                "b -> b -> i",
                "b -> i",
                "b -> i -> b",
                "b -> i -> i",
                "i",
                "i -> [[b]]",
                "i -> [[i]]",
                "i -> [b]",
                "i -> [i]",
                "i -> b",
                "i -> b -> b",
                "i -> b -> i",
                "i -> i",
                "i -> i -> b",
                "i -> i -> i",
            ],
        )

    def test_nested_expansion_1(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("[#a] -> #b"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_overall_depth=4,
            ),
            [
                "[b] -> b",
                "[b] -> b -> b",
                "[b] -> b -> i",
                "[b] -> i",
                "[b] -> i -> b",
                "[b] -> i -> i",
                "[i] -> b",
                "[i] -> b -> b",
                "[i] -> b -> i",
                "[i] -> i",
                "[i] -> i -> b",
                "[i] -> i -> i",
            ],
        )

    def test_nested_expansion_2(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("([#a], [#b]) -> #c"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_overall_depth=4,
            ),
            [
                "([b], [b]) -> b",
                "([b], [b]) -> b -> b",
                "([b], [b]) -> b -> i",
                "([b], [b]) -> i",
                "([b], [b]) -> i -> b",
                "([b], [b]) -> i -> i",
                "([b], [i]) -> b",
                "([b], [i]) -> b -> b",
                "([b], [i]) -> b -> i",
                "([b], [i]) -> i",
                "([b], [i]) -> i -> b",
                "([b], [i]) -> i -> i",
                "([i], [b]) -> b",
                "([i], [b]) -> b -> b",
                "([i], [b]) -> b -> i",
                "([i], [b]) -> i",
                "([i], [b]) -> i -> b",
                "([i], [b]) -> i -> i",
                "([i], [i]) -> b",
                "([i], [i]) -> b -> b",
                "([i], [i]) -> b -> i",
                "([i], [i]) -> i",
                "([i], [i]) -> i -> b",
                "([i], [i]) -> i -> i",
            ],
        )

    def test_nested_expansion_3(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("[[[[([#a], [#b]) -> #c]]]]"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_overall_depth=4,
            ),
            [],
        )

    def test_step_expansion_1(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("[#a] -> #a"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_expansion_steps=1,
            ),
            [
                "[b -> b] -> b -> b",
                "[b -> i] -> b -> i",
                "[b] -> b",
                "[i -> b] -> i -> b",
                "[i -> i] -> i -> i",
                "[i] -> i",
            ],
        )

    def test_step_expansion_2(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("[[[[#a] -> #a]]]"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_expansion_steps=1,
            ),
            [
                "[[[[b -> b] -> b -> b]]]",
                "[[[[b -> i] -> b -> i]]]",
                "[[[[b] -> b]]]",
                "[[[[i -> b] -> i -> b]]]",
                "[[[[i -> i] -> i -> i]]]",
                "[[[[i] -> i]]]",
            ],
        )

    def test_exclude_all_variables(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("[[[[#a] -> #a]]]"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_expansion_steps=1,
                exclude_variables=["a"],
            ),
            ["[[[[#a] -> #a]]]"],
        )

    def test_exclude_just_one(self):
        self.assertExpansions(
            ns.type_expansions(
                ns.parse_type("(#a, #b) -> #a"),
                terminals=[ns.parse_type(x) for x in ["b", "i"]],
                constructors=[(2, lambda x, y: ns.ArrowType((x,), y))],
                max_expansion_steps=1,
                exclude_variables=["a"],
            ),
            [
                "(#a, b) -> #a",
                "(#a, i -> b) -> #a",
                "(#a, i -> i) -> #a",
                "(#a, b -> b) -> #a",
                "(#a, i) -> #a",
                "(#a, b -> i) -> #a",
            ],
        )


class TestDSLExpand(unittest.TestCase):
    def test_basic_expand(self):
        dslf = ns.DSLFactory()
        dslf.production("+", "i -> i -> i", lambda x: lambda y: x + y)
        dslf.production("first", "(#a, #b) -> #a", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            + :: i -> i -> i
            first_0 :: (#a, (i -> i, i -> i) -> i -> i) -> #a
            first_1 :: (#a, (i -> i, i) -> i -> i) -> #a
            first_2 :: (#a, (i, i -> i) -> i) -> #a
            first_3 :: (#a, (i, i) -> i) -> #a
            first_4 :: (#a, i -> i -> i) -> #a
            first_5 :: (#a, i -> i) -> #a
            first_6 :: (#a, i) -> #a
            """,
        )

    def test_basic_expand_two(self):
        dslf = ns.DSLFactory(max_overall_depth=3)
        dslf.production("even?", "i -> b", lambda x: lambda y: x + y)
        dslf.production("first", "(#a, #b) -> #a", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            even? :: i -> b
            first_0 :: (#a, b) -> #a
            first_1 :: (#a, i) -> #a
            """,
        )

    def test_larger_expand(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("ite", "(b, #b, #a, #a) -> #a", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            ite_0 :: (b, () -> i, #a, #a) -> #a
            ite_1 :: (b, (b, b, b, b) -> b, #a, #a) -> #a
            ite_2 :: (b, (b, b, i, i) -> i, #a, #a) -> #a
            ite_3 :: (b, (b, i, b, b) -> b, #a, #a) -> #a
            ite_4 :: (b, (b, i, i, i) -> i, #a, #a) -> #a
            ite_5 :: (b, b, #a, #a) -> #a
            ite_6 :: (b, i, #a, #a) -> #a
            """,
        )

    def test_filtered_variable(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("1f", "() -> f", lambda: 1)
        dslf.filtered_type_variable(
            "num", lambda x: isinstance(x, ns.AtomicType) and x.name in ["i", "f"]
        )
        dslf.production("+", "%num -> %num -> %num", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                1 :: () -> i
                1f :: () -> f
                + :: %num -> %num -> %num
            """,
        )

    def test_expand_with_lambdas(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("id", "#a -> #a", lambda x: x)
        dslf.lambdas()
        dslf.prune_to("i -> i")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                $0_0 :: V<i@0>
                1 :: () -> i
                id :: #a -> #a
                lam :: L<#body|i> -> i -> #body
            """,
        )

    def test_expand_with_lambdas_no_prune(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("id", "#a -> #a", lambda x: x)
        dslf.lambdas()
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                $0_0 :: V<() -> i@0>
                $0_1 :: V<i -> i@0>
                $0_2 :: V<i@0>
                $1_0 :: V<() -> i@1>
                $1_1 :: V<i -> i@1>
                $1_2 :: V<i@1>
                $2_0 :: V<() -> i@2>
                $2_1 :: V<i -> i@2>
                $2_2 :: V<i@2>
                $3_0 :: V<() -> i@3>
                $3_1 :: V<i -> i@3>
                $3_2 :: V<i@3>
                1 :: () -> i
                id :: #a -> #a
                lam_0 :: L<#body|> -> () -> #body
                lam_1 :: L<#body|() -> i> -> (() -> i) -> #body
                lam_2 :: L<#body|i -> i> -> (i -> i) -> #body
                lam_3 :: L<#body|i> -> i -> #body
            """,
        )

    def test_expand_with_lambdas_in_not_top_level(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("id", "((i, #a) -> #a) -> i", lambda x, y: x)
        dslf.lambdas()
        dslf.prune_to("i -> i")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                $0_0 :: V<() -> i@0>
                $0_1 :: V<i -> i@0>
                $0_2 :: V<i@0>
                $1_2 :: V<i@1>
                $2_2 :: V<i@2>
                $3_2 :: V<i@3>
                1 :: () -> i
                id_0 :: ((i, () -> i) -> () -> i) -> i
                id_1 :: ((i, i -> i) -> i -> i) -> i
                id_2 :: ((i, i) -> i) -> i
                lam_0 :: L<#body|> -> () -> #body
                lam_1 :: L<#body|i;() -> i> -> (i, () -> i) -> #body
                lam_2 :: L<#body|i;i -> i> -> (i, i -> i) -> #body
                lam_3 :: L<#body|i;i> -> (i, i) -> #body
                lam_4 :: L<#body|i> -> i -> #body
            """,
        )

    def test_expand_with_lambdas_large_number_intermediates(self):
        level = 2
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas(max_type_depth=6)
        dslf.prune_to(
            ns.render_type(
                ns.ArrowType((ns.AtomicType("i"),) * level, ns.AtomicType("i"))
            )
        )
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                $0_0 :: V<i@0>
                $1_0 :: V<i@1>
                + :: (i, i) -> i
                1 :: () -> i
                lam :: L<#body|i;i> -> (i, i) -> #body
            """,
        )

    def test_expand_with_lambdas_large_number_intermediates_large_domain(self):
        level = 10
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas(max_type_depth=5)
        dslf.prune_to(
            ns.render_type(
                ns.ArrowType((ns.AtomicType("i"),) * level, ns.AtomicType("i"))
            )
        )
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                $0_0 :: V<i@0>
                $1_0 :: V<i@1>
                $2_0 :: V<i@2>
                $3_0 :: V<i@3>
                $4_0 :: V<i@4>
                $5_0 :: V<i@5>
                $6_0 :: V<i@6>
                $7_0 :: V<i@7>
                $8_0 :: V<i@8>
                $9_0 :: V<i@9>
                + :: (i, i) -> i
                1 :: () -> i
                lam :: L<#body|i;i;i;i;i;i;i;i;i;i> -> (i, i, i, i, i, i, i, i, i, i) -> #body
            """,
        )
