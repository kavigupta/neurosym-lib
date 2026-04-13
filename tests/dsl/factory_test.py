import unittest

import pytest

import neurosym as ns
from neurosym.examples import near

from ..utils import assertDSL


class TestDuplicateProduction(unittest.TestCase):
    def test_basic_duplicate(self):
        dslf = ns.DSLFactory()
        ident = lambda x: x
        dslf.production("1", "() -> i", ident)
        dslf.production("1", "() -> f", ident)
        dslf.prune_to("i", "f")
        self.assertRaisesRegex(
            ValueError,
            "^Duplicate declarations for production: 1$",
            dslf.finalize,
        )

    def test_exact_duplicate_allowed(self):
        dslf = ns.DSLFactory()
        ident = lambda x: x
        dslf.production("1", "() -> i", ident)
        dslf.production("1", "() -> i", ident)
        dslf.prune_to("i")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            """,
        )


class TestPruning(unittest.TestCase):
    def test_basic_pruning(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda x: x)
        dslf.production("identity", "i -> i", lambda x: x)
        dslf.production("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.production("convert", "#x -> f", float)
        dslf.prune_to("f")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            add :: (i, i) -> i
            convert_0 :: f -> f
            convert_1 :: i -> f
            identity :: i -> i
            """,
        )

    def test_pruning_error(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda x: x)
        dslf.production("identity", "i -> i", lambda x: x)
        dslf.production("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.production("convert", "#x -> i", float)
        dslf.prune_to("f")
        self.assertRaisesRegex(
            TypeError,
            "All productions for .* were pruned. Check that the target types are correct.",
            dslf.finalize,
        )

    def test_pruning_specific(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda x: x)
        dslf.production("identity", "f -> i", lambda x: x)
        dslf.production("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.prune_to("i")
        self.assertRaises(
            TypeError,
            "All productions for identity were pruned. Check that the target types are correct.",
            dslf.finalize,
        )

    def test_allow_pruning(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda x: x)
        dslf.production("identity", "f -> i", lambda x: x)
        dslf.production("add", "(i, i) -> i", lambda x, y: x + y)
        dslf.prune_to("i", tolerate_pruning_entire_productions=True)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            add :: (i, i) -> i
            """,
        )

    def test_pruning_with_call(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda x: x)
        dslf.production("call", "(i -> i, i) -> i", lambda x, y: x(y))
        dslf.lambdas()
        dslf.prune_to("i")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            $0 :: V<$0>
            $1 :: V<$1>
            $2 :: V<$2>
            $3 :: V<$3>
            1 :: () -> i
            call :: (i -> i, i) -> i
            lam :: L<#body|#__lam_0> -> #__lam_0 -> #body
            """,
        )


class TestScalability(unittest.TestCase):
    """Ensure constructibility-based finalize handles realistic DSL sizes."""

    @pytest.mark.timeout(5)
    def test_many_atomic_types_with_polymorphism(self):
        dslf = ns.DSLFactory()
        n = 50
        types = [f"t{i}" for i in range(n)]
        for t in types:
            dslf.production(f"mk_{t}", f"() -> {t}", lambda: 1)
        dslf.production("id", "#a -> #a", lambda x: x)
        dslf.production("pair", "(#a, #b) -> #a", lambda x, y: x)
        dslf.prune_to(*types)
        dsl = dslf.finalize()
        self.assertGreater(len(dsl.productions), n)

    @pytest.mark.timeout(2)
    def test_linear_chain(self):
        dslf = ns.DSLFactory()
        n = 30
        dslf.production("start", "() -> t0", lambda: 1)
        for i in range(n - 1):
            dslf.production(f"step{i}", f"t{i} -> t{i+1}", lambda x: x)
        dslf.prune_to(f"t{n-1}")
        dsl = dslf.finalize()
        self.assertEqual(len(dsl.productions), n)

    @pytest.mark.timeout(2)
    def test_near_like_dsl(self):
        dslf = ns.DSLFactory()
        dslf.production("input", "() -> [f]", lambda: [1.0])
        dslf.production("mlp", "[f] -> [f]", lambda x: x)
        dslf.production("rnn", "([f]) -> [f] -> [f]", lambda x: x)
        dslf.production(
            "ite",
            "([f] -> [f], [f] -> [f], [f] -> [f]) -> [f] -> [f]",
            lambda x: x,
        )
        dslf.production(
            "compose",
            "(#a -> #b, #b -> #c) -> #a -> #c",
            lambda f, g: lambda x: f(g(x)),
        )
        dslf.lambdas()
        dslf.prune_to("[f] -> [f]")
        dsl = dslf.finalize()
        self.assertGreater(len(dsl.productions), 5)

    @pytest.mark.timeout(2)
    def test_ite_with_lambdas(self):
        dslf = ns.DSLFactory()
        dslf.production("0", "() -> i", lambda: 0)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("true", "() -> b", lambda: True)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.production("*", "(i, i) -> i", lambda x, y: x * y)
        dslf.production("eq", "(i, i) -> b", lambda x, y: x == y)
        dslf.production("ite", "(b, #a, #a) -> #a", lambda c, t, f: t if c else f)
        dslf.production(
            "compose",
            "(#a -> #b, #b -> #c) -> #a -> #c",
            lambda f, g: lambda x: f(g(x)),
        )
        dslf.lambdas()
        dslf.prune_to("i -> i", "i")
        dsl = dslf.finalize()
        self.assertGreater(len(dsl.productions), 10)

    @pytest.mark.timeout(2)
    def test_deep_list_nesting(self):
        dslf = ns.DSLFactory(max_overall_depth=10)
        dslf.production("mk", "() -> i", lambda: 1)
        dslf.production("wrap", "#a -> [#a]", lambda x: [x])
        dslf.production("head", "[#a] -> #a", lambda x: x[0])
        dslf.prune_to("i", "[i]", "[[i]]", "[[[i]]]")
        dsl = dslf.finalize()
        self.assertGreater(len(dsl.productions), 2)

    @pytest.mark.timeout(10)
    def test_list_dsl_higher_depth(self):
        dsl = ns.examples.dreamcoder.list_dsl(
            "[i] -> i", "[i] -> [i]", "i -> i", max_overall_depth=7
        )
        self.assertGreater(len(dsl.productions), 30)

    @pytest.mark.timeout(5)
    def test_attention_ecg_dsl_21_channels(self):
        dsl = near.attention_ecg_dsl(num_channels=21, features_per_channel=14, num_classes=5)
        self.assertGreater(len(dsl.productions), 20)

    @pytest.mark.timeout(5)
    def test_attention_ecg_dsl_21_channels_with_shields(self):
        dsl = near.attention_ecg_dsl(
            num_channels=21,
            features_per_channel=14,
            num_classes=5,
            use_shields=True,
        )
        self.assertGreater(len(dsl.productions), 40)

    @pytest.mark.timeout(5)
    def test_attention_ecg_dsl_100_channels(self):
        dsl = near.attention_ecg_dsl(
            num_channels=100, features_per_channel=14, num_classes=5
        )
        self.assertGreater(len(dsl.productions), 100)
