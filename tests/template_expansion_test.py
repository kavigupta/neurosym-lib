import unittest
from neurosym.dsl.dsl_factory import DSLFactory


class TestAllRules(unittest.TestCase):
    def assertDSL(self, dsl, expected):
        dsl = "\n".join(
            sorted([line.strip() for line in dsl.split("\n") if line.strip()])
        )
        expected = "\n".join(
            sorted([line.strip() for line in expected.split("\n") if line.strip()])
        )
        self.maxDiff = None
        self.assertEqual(dsl, expected)

    def test_basic_expand(self):
        dslf = DSLFactory()
        dslf.concrete("+", "i -> i -> i", lambda x: lambda y: x + y)
        dslf.concrete("id", "#a -> #a", lambda x: x)
        dsl = dslf.finalize()
        self.assertDSL(
            dsl.render(),
            """
                + :: i -> i -> i
            id_0 :: ((i -> i) -> i -> i) -> (i -> i) -> i -> i
            id_1 :: ((i -> i) -> i) -> (i -> i) -> i
            id_2 :: (i -> i -> i) -> i -> i -> i
            id_3 :: (i -> i) -> i -> i
            id_4 :: i -> i
            """,
        )

    def test_basic_expand_two(self):
        dslf = DSLFactory(max_overall_depth=3)
        dslf.concrete("even?", "i -> b", lambda x: lambda y: x + y)
        dslf.concrete("id", "#a -> #a", lambda x: x)
        dsl = dslf.finalize()
        self.assertDSL(
            dsl.render(),
            """
            even? :: i -> b
            id_0 :: b -> b
            id_1 :: i -> i
            """,
        )

    def test_larger_expand(self):
        dslf = DSLFactory()
        dslf.concrete("1", "() -> i", lambda: 1)
        dslf.concrete("ite", "(b, #a, #a, #a) -> #a", lambda x: x)
        dsl = dslf.finalize()
        self.assertDSL(
            dsl.render(),
            """
                1 :: () -> i
            ite_0 :: (b, () -> () -> () -> b, () -> () -> () -> b, () -> () -> () -> b) -> () -> () -> () -> b
            ite_1 :: (b, () -> () -> () -> i, () -> () -> () -> i, () -> () -> () -> i) -> () -> () -> () -> i
            ite_2 :: (b, () -> () -> b, () -> () -> b, () -> () -> b) -> () -> () -> b
            ite_3 :: (b, () -> () -> i, () -> () -> i, () -> () -> i) -> () -> () -> i
            ite_4 :: (b, () -> b, () -> b, () -> b) -> () -> b
            ite_5 :: (b, () -> i, () -> i, () -> i) -> () -> i
            ite_6 :: (b, (b, b, b, b) -> b, (b, b, b, b) -> b, (b, b, b, b) -> b) -> (b, b, b, b) -> b
            ite_7 :: (b, (b, b, b, b) -> i, (b, b, b, b) -> i, (b, b, b, b) -> i) -> (b, b, b, b) -> i
            ite_8 :: (b, (b, b, b, i) -> b, (b, b, b, i) -> b, (b, b, b, i) -> b) -> (b, b, b, i) -> b
            ite_9 :: (b, (b, b, b, i) -> i, (b, b, b, i) -> i, (b, b, b, i) -> i) -> (b, b, b, i) -> i
            ite_10 :: (b, (b, b, i, b) -> b, (b, b, i, b) -> b, (b, b, i, b) -> b) -> (b, b, i, b) -> b
            ite_11 :: (b, (b, b, i, b) -> i, (b, b, i, b) -> i, (b, b, i, b) -> i) -> (b, b, i, b) -> i
            ite_12 :: (b, (b, b, i, i) -> b, (b, b, i, i) -> b, (b, b, i, i) -> b) -> (b, b, i, i) -> b
            ite_13 :: (b, (b, b, i, i) -> i, (b, b, i, i) -> i, (b, b, i, i) -> i) -> (b, b, i, i) -> i
            ite_14 :: (b, (b, i, b, b) -> b, (b, i, b, b) -> b, (b, i, b, b) -> b) -> (b, i, b, b) -> b
            ite_15 :: (b, (b, i, b, b) -> i, (b, i, b, b) -> i, (b, i, b, b) -> i) -> (b, i, b, b) -> i
            ite_16 :: (b, (b, i, b, i) -> b, (b, i, b, i) -> b, (b, i, b, i) -> b) -> (b, i, b, i) -> b
            ite_17 :: (b, (b, i, b, i) -> i, (b, i, b, i) -> i, (b, i, b, i) -> i) -> (b, i, b, i) -> i
            ite_18 :: (b, (b, i, i, b) -> b, (b, i, i, b) -> b, (b, i, i, b) -> b) -> (b, i, i, b) -> b
            ite_19 :: (b, (b, i, i, b) -> i, (b, i, i, b) -> i, (b, i, i, b) -> i) -> (b, i, i, b) -> i
            ite_20 :: (b, (b, i, i, i) -> b, (b, i, i, i) -> b, (b, i, i, i) -> b) -> (b, i, i, i) -> b
            ite_21 :: (b, (b, i, i, i) -> i, (b, i, i, i) -> i, (b, i, i, i) -> i) -> (b, i, i, i) -> i
            ite_22 :: (b, (i, b, b, b) -> b, (i, b, b, b) -> b, (i, b, b, b) -> b) -> (i, b, b, b) -> b
            ite_23 :: (b, (i, b, b, b) -> i, (i, b, b, b) -> i, (i, b, b, b) -> i) -> (i, b, b, b) -> i
            ite_24 :: (b, (i, b, b, i) -> b, (i, b, b, i) -> b, (i, b, b, i) -> b) -> (i, b, b, i) -> b
            ite_25 :: (b, (i, b, b, i) -> i, (i, b, b, i) -> i, (i, b, b, i) -> i) -> (i, b, b, i) -> i
            ite_26 :: (b, (i, b, i, b) -> b, (i, b, i, b) -> b, (i, b, i, b) -> b) -> (i, b, i, b) -> b
            ite_27 :: (b, (i, b, i, b) -> i, (i, b, i, b) -> i, (i, b, i, b) -> i) -> (i, b, i, b) -> i
            ite_28 :: (b, (i, b, i, i) -> b, (i, b, i, i) -> b, (i, b, i, i) -> b) -> (i, b, i, i) -> b
            ite_29 :: (b, (i, b, i, i) -> i, (i, b, i, i) -> i, (i, b, i, i) -> i) -> (i, b, i, i) -> i
            ite_30 :: (b, (i, i, b, b) -> b, (i, i, b, b) -> b, (i, i, b, b) -> b) -> (i, i, b, b) -> b
            ite_31 :: (b, (i, i, b, b) -> i, (i, i, b, b) -> i, (i, i, b, b) -> i) -> (i, i, b, b) -> i
            ite_32 :: (b, (i, i, b, i) -> b, (i, i, b, i) -> b, (i, i, b, i) -> b) -> (i, i, b, i) -> b
            ite_33 :: (b, (i, i, b, i) -> i, (i, i, b, i) -> i, (i, i, b, i) -> i) -> (i, i, b, i) -> i
            ite_34 :: (b, (i, i, i, b) -> b, (i, i, i, b) -> b, (i, i, i, b) -> b) -> (i, i, i, b) -> b
            ite_35 :: (b, (i, i, i, b) -> i, (i, i, i, b) -> i, (i, i, i, b) -> i) -> (i, i, i, b) -> i
            ite_36 :: (b, (i, i, i, i) -> b, (i, i, i, i) -> b, (i, i, i, i) -> b) -> (i, i, i, i) -> b
            ite_37 :: (b, (i, i, i, i) -> i, (i, i, i, i) -> i, (i, i, i, i) -> i) -> (i, i, i, i) -> i
            ite_38 :: (b, b, b, b) -> b
            ite_39 :: (b, i, i, i) -> i
            """,
        )
