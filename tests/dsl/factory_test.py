import unittest

import pytest

import neurosym as ns

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
            $0_0 :: V<i@0>
            $1_0 :: V<i@1>
            $2_0 :: V<i@2>
            $3_0 :: V<i@3>
            1 :: () -> i
            call :: (i -> i, i) -> i
            lam :: L<#body|i> -> i -> #body
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
        dslf = ns.DSLFactory(max_overall_depth=7)
        # Replicate list_dsl productions but with higher depth
        for i in range(6):
            dslf.production(str(i), "() -> i", lambda i=i: i)
        dslf.production("empty", "() -> [#T]", lambda: [])
        dslf.production("singleton", "#T -> [#T]", lambda x: [x])
        dslf.production("range", "i -> [i]", lambda x: list(range(x)))
        dslf.production("++", "([#T], [#T]) -> [#T]", lambda x, y: x + y)
        dslf.production("true", "() -> b", lambda: True)
        dslf.production("not", "b -> b", lambda x: not x)
        dslf.production("and", "(b, b) -> b", lambda x, y: x and y)
        dslf.production("or", "(b, b) -> b", lambda x, y: x or y)
        dslf.production("i", "(b, #T, #T) -> #T", lambda x, y, z: y if x else z)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.production("*", "(i, i) -> i", lambda x, y: x * y)
        dslf.production("negate", "i -> i", lambda x: -x)
        dslf.production("eq?", "(i, i) -> b", lambda x, y: x == y)
        dslf.production("gt?", "(i, i) -> b", lambda x, y: x > y)
        dslf.production("sum", "[i] -> i", sum)
        dslf.production("reverse", "[#T] -> [#T]", lambda x: x[::-1])
        dslf.production("index", "(i, [#T]) -> #T", lambda x, y: y[x])
        dslf.lambdas(max_type_depth=3)
        dslf.prune_to("[i] -> i", "[i] -> [i]", "i -> i", prune_variables=False)
        dsl = dslf.finalize()
        self.assertGreater(len(dsl.productions), 30)
