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
        dslf.concrete("+", "i -> i -> i", lambda x: lambda y: x + y)
        dslf.concrete("first", "(#a, #b) -> #a", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            + :: i -> i -> i
            first_0 :: (#a, ((i, i) -> i, (i, i) -> i) -> (i, i) -> i) -> #a
            first_1 :: (#a, ((i, i) -> i, (i, i) -> i) -> i -> i) -> #a
            first_2 :: (#a, ((i, i) -> i, (i, i) -> i) -> i) -> #a
            first_3 :: (#a, ((i, i) -> i, i -> i) -> (i, i) -> i) -> #a
            first_4 :: (#a, ((i, i) -> i, i -> i) -> i -> i) -> #a
            first_5 :: (#a, ((i, i) -> i, i -> i) -> i) -> #a
            first_6 :: (#a, ((i, i) -> i, i) -> (i, i) -> i) -> #a
            first_7 :: (#a, ((i, i) -> i, i) -> i -> i) -> #a
            first_8 :: (#a, ((i, i) -> i, i) -> i) -> #a
            first_9 :: (#a, ((i, i) -> i) -> (i, i) -> i) -> #a
            first_10 :: (#a, ((i, i) -> i) -> i -> i) -> #a
            first_11 :: (#a, ((i, i) -> i) -> i) -> #a
            first_12 :: (#a, (i -> i, (i, i) -> i) -> (i, i) -> i) -> #a
            first_13 :: (#a, (i -> i, (i, i) -> i) -> i -> i) -> #a
            first_14 :: (#a, (i -> i, (i, i) -> i) -> i) -> #a
            first_15 :: (#a, (i -> i, i -> i) -> (i, i) -> i) -> #a
            first_16 :: (#a, (i -> i, i -> i) -> i -> i) -> #a
            first_17 :: (#a, (i -> i, i -> i) -> i) -> #a
            first_18 :: (#a, (i -> i, i) -> (i, i) -> i) -> #a
            first_19 :: (#a, (i -> i, i) -> i -> i) -> #a
            first_20 :: (#a, (i -> i, i) -> i) -> #a
            first_21 :: (#a, (i -> i) -> (i, i) -> i) -> #a
            first_22 :: (#a, (i -> i) -> i -> i) -> #a
            first_23 :: (#a, (i -> i) -> i) -> #a
            first_24 :: (#a, (i, (i, i) -> i) -> (i, i) -> i) -> #a
            first_25 :: (#a, (i, (i, i) -> i) -> i -> i) -> #a
            first_26 :: (#a, (i, (i, i) -> i) -> i) -> #a
            first_27 :: (#a, (i, i -> i) -> (i, i) -> i) -> #a
            first_28 :: (#a, (i, i -> i) -> i -> i) -> #a
            first_29 :: (#a, (i, i -> i) -> i) -> #a
            first_30 :: (#a, (i, i) -> (i, i) -> i) -> #a
            first_31 :: (#a, (i, i) -> i -> i) -> #a
            first_32 :: (#a, (i, i) -> i) -> #a
            first_33 :: (#a, i -> (i, i) -> i) -> #a
            first_34 :: (#a, i -> i -> i) -> #a
            first_35 :: (#a, i -> i) -> #a
            first_36 :: (#a, i) -> #a
            """,
        )

    def test_basic_expand_two(self):
        dslf = ns.DSLFactory(max_overall_depth=3)
        dslf.concrete("even?", "i -> b", lambda x: lambda y: x + y)
        dslf.concrete("first", "(#a, #b) -> #a", lambda x: x)
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
        dslf.concrete("1", "() -> i", lambda: 1)
        dslf.concrete("ite", "(b, #b, #a, #a) -> #a", lambda x: x)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                1 :: () -> i
                ite_0 :: (b, () -> () -> () -> b, #a, #a) -> #a
                ite_1 :: (b, () -> () -> () -> i, #a, #a) -> #a
                ite_2 :: (b, () -> () -> b, #a, #a) -> #a
                ite_3 :: (b, () -> () -> i, #a, #a) -> #a
                ite_4 :: (b, () -> b, #a, #a) -> #a
                ite_5 :: (b, () -> i, #a, #a) -> #a
                ite_6 :: (b, (b, b, b, b) -> b, #a, #a) -> #a
                ite_7 :: (b, (b, b, b, b) -> i, #a, #a) -> #a
                ite_8 :: (b, (b, b, b, i) -> b, #a, #a) -> #a
                ite_9 :: (b, (b, b, b, i) -> i, #a, #a) -> #a
                ite_10 :: (b, (b, b, i, b) -> b, #a, #a) -> #a
                ite_11 :: (b, (b, b, i, b) -> i, #a, #a) -> #a
                ite_12 :: (b, (b, b, i, i) -> b, #a, #a) -> #a
                ite_13 :: (b, (b, b, i, i) -> i, #a, #a) -> #a
                ite_14 :: (b, (b, i, b, b) -> b, #a, #a) -> #a
                ite_15 :: (b, (b, i, b, b) -> i, #a, #a) -> #a
                ite_16 :: (b, (b, i, b, i) -> b, #a, #a) -> #a
                ite_17 :: (b, (b, i, b, i) -> i, #a, #a) -> #a
                ite_18 :: (b, (b, i, i, b) -> b, #a, #a) -> #a
                ite_19 :: (b, (b, i, i, b) -> i, #a, #a) -> #a
                ite_20 :: (b, (b, i, i, i) -> b, #a, #a) -> #a
                ite_21 :: (b, (b, i, i, i) -> i, #a, #a) -> #a
                ite_22 :: (b, (i, b, b, b) -> b, #a, #a) -> #a
                ite_23 :: (b, (i, b, b, b) -> i, #a, #a) -> #a
                ite_24 :: (b, (i, b, b, i) -> b, #a, #a) -> #a
                ite_25 :: (b, (i, b, b, i) -> i, #a, #a) -> #a
                ite_26 :: (b, (i, b, i, b) -> b, #a, #a) -> #a
                ite_27 :: (b, (i, b, i, b) -> i, #a, #a) -> #a
                ite_28 :: (b, (i, b, i, i) -> b, #a, #a) -> #a
                ite_29 :: (b, (i, b, i, i) -> i, #a, #a) -> #a
                ite_30 :: (b, (i, i, b, b) -> b, #a, #a) -> #a
                ite_31 :: (b, (i, i, b, b) -> i, #a, #a) -> #a
                ite_32 :: (b, (i, i, b, i) -> b, #a, #a) -> #a
                ite_33 :: (b, (i, i, b, i) -> i, #a, #a) -> #a
                ite_34 :: (b, (i, i, i, b) -> b, #a, #a) -> #a
                ite_35 :: (b, (i, i, i, b) -> i, #a, #a) -> #a
                ite_36 :: (b, (i, i, i, i) -> b, #a, #a) -> #a
                ite_37 :: (b, (i, i, i, i) -> i, #a, #a) -> #a
                ite_38 :: (b, b, #a, #a) -> #a
                ite_39 :: (b, i, #a, #a) -> #a
            """,
        )

    def test_filtered_variable(self):
        dslf = ns.DSLFactory()
        dslf.concrete("1", "() -> i", lambda: 1)
        dslf.concrete("1f", "() -> f", lambda: 1)
        dslf.filtered_type_variable(
            "num", lambda x: isinstance(x, ns.AtomicType) and x.name in ["i", "f"]
        )
        dslf.concrete("+", "%num -> %num -> %num", lambda x: x)
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
