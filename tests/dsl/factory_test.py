import unittest

import neurosym as ns

from ..utils import assertDSL


class TestDuplicateProduction(unittest.TestCase):
    def test_basic_duplicate(self):
        dslf = ns.DSLFactory()
        ident = lambda x: x
        dslf.production("1", "() -> i", ident)
        dslf.production("1", "() -> f", ident)
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
            convert :: #x -> f
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
            $0 :: V<@0>
            $1 :: V<@1>
            $2 :: V<@2>
            $3 :: V<@3>
            1 :: () -> i
            call :: (i -> i, i) -> i
            lam :: L<#body|#_arg0> -> (#_arg0) -> #body
            """,
        )
