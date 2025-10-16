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
            first_0 :: (#a, ((i, i) -> i, (i, i) -> i) -> (i, i) -> i) -> #a
            first_1 :: (#a, ((i, i) -> i, i -> i) -> (i, i) -> i) -> #a
            first_10 :: (#a, i -> i) -> #a
            first_11 :: (#a, i) -> #a
            first_2 :: (#a, ((i, i) -> i, i) -> (i, i) -> i) -> #a
            first_3 :: (#a, (i -> i, (i, i) -> i) -> i -> i) -> #a
            first_4 :: (#a, (i -> i, i -> i) -> i -> i) -> #a
            first_5 :: (#a, (i -> i, i) -> i -> i) -> #a
            first_6 :: (#a, (i, (i, i) -> i) -> i) -> #a
            first_7 :: (#a, (i, i -> i) -> i) -> #a
            first_8 :: (#a, (i, i) -> i) -> #a
            first_9 :: (#a, i -> i -> i) -> #a
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
            ite_1 :: (b, (b, () -> i, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_10 :: (b, (b, (b, b, b, b) -> b, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_11 :: (b, (b, (b, b, b, b) -> b, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_12 :: (b, (b, (b, b, b, b) -> b, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_13 :: (b, (b, (b, b, b, b) -> b, b, b) -> b, #a, #a) -> #a
            ite_14 :: (b, (b, (b, b, b, b) -> b, i, i) -> i, #a, #a) -> #a
            ite_15 :: (b, (b, (b, b, i, i) -> i, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_16 :: (b, (b, (b, b, i, i) -> i, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_17 :: (b, (b, (b, b, i, i) -> i, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_18 :: (b, (b, (b, b, i, i) -> i, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_19 :: (b, (b, (b, b, i, i) -> i, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_2 :: (b, (b, () -> i, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_20 :: (b, (b, (b, b, i, i) -> i, b, b) -> b, #a, #a) -> #a
            ite_21 :: (b, (b, (b, b, i, i) -> i, i, i) -> i, #a, #a) -> #a
            ite_22 :: (b, (b, (b, i, b, b) -> b, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_23 :: (b, (b, (b, i, b, b) -> b, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_24 :: (b, (b, (b, i, b, b) -> b, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_25 :: (b, (b, (b, i, b, b) -> b, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_26 :: (b, (b, (b, i, b, b) -> b, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_27 :: (b, (b, (b, i, b, b) -> b, b, b) -> b, #a, #a) -> #a
            ite_28 :: (b, (b, (b, i, b, b) -> b, i, i) -> i, #a, #a) -> #a
            ite_29 :: (b, (b, (b, i, i, i) -> i, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_3 :: (b, (b, () -> i, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_30 :: (b, (b, (b, i, i, i) -> i, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_31 :: (b, (b, (b, i, i, i) -> i, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_32 :: (b, (b, (b, i, i, i) -> i, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_33 :: (b, (b, (b, i, i, i) -> i, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_34 :: (b, (b, (b, i, i, i) -> i, b, b) -> b, #a, #a) -> #a
            ite_35 :: (b, (b, (b, i, i, i) -> i, i, i) -> i, #a, #a) -> #a
            ite_36 :: (b, (b, b, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_37 :: (b, (b, b, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_38 :: (b, (b, b, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_39 :: (b, (b, b, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_4 :: (b, (b, () -> i, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_40 :: (b, (b, b, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_41 :: (b, (b, b, b, b) -> b, #a, #a) -> #a
            ite_42 :: (b, (b, b, i, i) -> i, #a, #a) -> #a
            ite_43 :: (b, (b, i, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_44 :: (b, (b, i, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
            ite_45 :: (b, (b, i, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i, #a, #a) -> #a
            ite_46 :: (b, (b, i, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b, #a, #a) -> #a
            ite_47 :: (b, (b, i, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_48 :: (b, (b, i, b, b) -> b, #a, #a) -> #a
            ite_49 :: (b, (b, i, i, i) -> i, #a, #a) -> #a
            ite_5 :: (b, (b, () -> i, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i, #a, #a) -> #a
            ite_50 :: (b, b, #a, #a) -> #a
            ite_51 :: (b, i, #a, #a) -> #a
            ite_6 :: (b, (b, () -> i, b, b) -> b, #a, #a) -> #a
            ite_7 :: (b, (b, () -> i, i, i) -> i, #a, #a) -> #a
            ite_8 :: (b, (b, (b, b, b, b) -> b, () -> i, () -> i) -> () -> i, #a, #a) -> #a
            ite_9 :: (b, (b, (b, b, b, b) -> b, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b, #a, #a) -> #a
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
                $0_1 :: V<(() -> i) -> () -> i@0>
                $0_2 :: V<(i -> i) -> i -> i@0>
                $0_3 :: V<i -> i@0>
                $0_4 :: V<i@0>
                $1_0 :: V<() -> i@1>
                $1_1 :: V<(() -> i) -> () -> i@1>
                $1_2 :: V<(i -> i) -> i -> i@1>
                $1_3 :: V<i -> i@1>
                $1_4 :: V<i@1>
                $2_0 :: V<() -> i@2>
                $2_1 :: V<(() -> i) -> () -> i@2>
                $2_2 :: V<(i -> i) -> i -> i@2>
                $2_3 :: V<i -> i@2>
                $2_4 :: V<i@2>
                $3_0 :: V<() -> i@3>
                $3_1 :: V<(() -> i) -> () -> i@3>
                $3_2 :: V<(i -> i) -> i -> i@3>
                $3_3 :: V<i -> i@3>
                $3_4 :: V<i@3>
                1 :: () -> i
                id :: #a -> #a
                lam_0 :: L<#body|> -> () -> #body
                lam_1 :: L<#body|() -> i> -> (() -> i) -> #body
                lam_2 :: L<#body|(() -> i) -> () -> i> -> ((() -> i) -> () -> i) -> #body
                lam_3 :: L<#body|(i -> i) -> i -> i> -> ((i -> i) -> i -> i) -> #body
                lam_4 :: L<#body|i -> i> -> (i -> i) -> #body
                lam_5 :: L<#body|i> -> i -> #body
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
                $0_1 :: V<((i, i) -> i) -> i@0>
                $0_2 :: V<(i, () -> i) -> () -> i@0>
                $0_3 :: V<(i, (i, i) -> i) -> (i, i) -> i@0>
                $0_4 :: V<(i, i) -> i@0>
                $0_5 :: V<i -> i@0>
                $0_6 :: V<i@0>
                $1_6 :: V<i@1>
                $2_6 :: V<i@2>
                $3_6 :: V<i@3>
                1 :: () -> i
                id_0 :: ((i, () -> i) -> () -> i) -> i
                id_1 :: ((i, ((i, i) -> i) -> i) -> ((i, i) -> i) -> i) -> i
                id_2 :: ((i, (i, () -> i) -> () -> i) -> (i, () -> i) -> () -> i) -> i
                id_3 :: ((i, (i, (i, i) -> i) -> (i, i) -> i) -> (i, (i, i) -> i) -> (i, i) -> i) -> i
                id_4 :: ((i, (i, i) -> i) -> (i, i) -> i) -> i
                id_5 :: ((i, i -> i) -> i -> i) -> i
                id_6 :: ((i, i) -> i) -> i
                lam_0 :: L<#body|> -> () -> #body
                lam_1 :: L<#body|(i, i) -> i> -> ((i, i) -> i) -> #body
                lam_2 :: L<#body|i;() -> i> -> (i, () -> i) -> #body
                lam_3 :: L<#body|i;((i, i) -> i) -> i> -> (i, ((i, i) -> i) -> i) -> #body
                lam_4 :: L<#body|i;(i, () -> i) -> () -> i> -> (i, (i, () -> i) -> () -> i) -> #body
                lam_5 :: L<#body|i;(i, (i, i) -> i) -> (i, i) -> i> -> (i, (i, (i, i) -> i) -> (i, i) -> i) -> #body
                lam_6 :: L<#body|i;(i, i) -> i> -> (i, (i, i) -> i) -> #body
                lam_7 :: L<#body|i;i -> i> -> (i, i -> i) -> #body
                lam_8 :: L<#body|i;i> -> (i, i) -> #body
                lam_9 :: L<#body|i> -> i -> #body
            """,
        )

    def test_expand_with_lambdas_large_number_intermediates(self):
        level = 2
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas(require_arities=[level], max_type_depth=6)
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
        dslf.lambdas(require_arities=[level], max_type_depth=5)
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

    def test_expand_with_lambdas_large_number_intermediates_large_domain_larger_depth(
        self,
    ):
        level = 40
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas(require_arities=[level], max_type_depth=level)
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
                $10_0 :: V<i@10>
                $11_0 :: V<i@11>
                $12_0 :: V<i@12>
                $13_0 :: V<i@13>
                $14_0 :: V<i@14>
                $15_0 :: V<i@15>
                $16_0 :: V<i@16>
                $17_0 :: V<i@17>
                $18_0 :: V<i@18>
                $19_0 :: V<i@19>
                $1_0 :: V<i@1>
                $20_0 :: V<i@20>
                $21_0 :: V<i@21>
                $22_0 :: V<i@22>
                $23_0 :: V<i@23>
                $24_0 :: V<i@24>
                $25_0 :: V<i@25>
                $26_0 :: V<i@26>
                $27_0 :: V<i@27>
                $28_0 :: V<i@28>
                $29_0 :: V<i@29>
                $2_0 :: V<i@2>
                $30_0 :: V<i@30>
                $31_0 :: V<i@31>
                $32_0 :: V<i@32>
                $33_0 :: V<i@33>
                $34_0 :: V<i@34>
                $35_0 :: V<i@35>
                $36_0 :: V<i@36>
                $37_0 :: V<i@37>
                $38_0 :: V<i@38>
                $39_0 :: V<i@39>
                $3_0 :: V<i@3>
                $4_0 :: V<i@4>
                $5_0 :: V<i@5>
                $6_0 :: V<i@6>
                $7_0 :: V<i@7>
                $8_0 :: V<i@8>
                $9_0 :: V<i@9>
                + :: (i, i) -> i
                1 :: () -> i
                lam :: L<#body|i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i;i> -> (i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i) -> #body
            """,
        )
