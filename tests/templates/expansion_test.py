# pylint: disable=duplicate-code
# we don't care about duplicate code in test outputs

import unittest

import neurosym as ns

from ..utils import assertDSL


class TestDSLExpand(unittest.TestCase):
    def test_prune_unreachable(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.production("to_f", "i -> f", float)
        dslf.prune_to("i", tolerate_pruning_entire_productions=True)
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            + :: (i, i) -> i
            """,
        )

    def test_prune_keeps_reachable_chain(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("to_f", "i -> f", float)
        dslf.production("to_g", "f -> g", lambda x: x)
        dslf.prune_to("g")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
            1 :: () -> i
            to_f :: i -> f
            to_g :: f -> g
            """,
        )

    def test_filtered_variable(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("1f", "() -> f", lambda: 1)
        dslf.filtered_type_variable(
            "num", lambda x: isinstance(x, ns.AtomicType) and x.name in ["i", "f"]
        )
        dslf.production("+", "(%num, %num) -> %num", lambda x, y: x + y)
        dslf.prune_to("i", "f")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                1 :: () -> i
                1f :: () -> f
                + :: (%num, %num) -> %num
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
                $0 :: V<$0>
                $1 :: V<$1>
                $2 :: V<$2>
                $3 :: V<$3>
                1 :: () -> i
                id :: #a -> #a
                lam :: L<#body|#__lam_0> -> #__lam_0 -> #body
            """,
        )

    def test_expand_with_lambdas_broad_targets(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("id", "#a -> #a", lambda x: x)
        dslf.lambdas()
        dslf.prune_to("i", "() -> i", "i -> i", "(i -> i) -> i")
        dsl = dslf.finalize()
        assertDSL(
            self,
            dsl.render(),
            """
                1 :: () -> i
                id :: #a -> #a
                lam_0 :: L<#body|> -> () -> #body
                lam_1 :: L<#body|#__lam_0> -> #__lam_0 -> #body
                $0 :: V<$0>
                $1 :: V<$1>
                $2 :: V<$2>
                $3 :: V<$3>
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
                $0 :: V<$0>
                $1 :: V<$1>
                $2 :: V<$2>
                $3 :: V<$3>
                1 :: () -> i
                id :: ((i, i) -> i) -> i
                lam_0 :: L<#body|#__lam_0> -> #__lam_0 -> #body
                lam_1 :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
            """,
        )

    def test_expand_with_lambdas_large_number_intermediates(self):
        level = 2
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas()
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
                $0 :: V<$0>
                $1 :: V<$1>
                + :: (i, i) -> i
                1 :: () -> i
                lam :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
            """,
        )

    def test_expand_with_lambdas_large_number_intermediates_large_domain(self):
        level = 10
        dslf = ns.DSLFactory(max_env_depth=level)
        dslf.production("1", "() -> i", lambda: 1)
        dslf.production("+", "(i, i) -> i", lambda x, y: x + y)
        dslf.lambdas()
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
                $0 :: V<$0>
                $1 :: V<$1>
                $2 :: V<$2>
                $3 :: V<$3>
                $4 :: V<$4>
                $5 :: V<$5>
                $6 :: V<$6>
                $7 :: V<$7>
                $8 :: V<$8>
                $9 :: V<$9>
                + :: (i, i) -> i
                1 :: () -> i
                lam :: L<#body|#__lam_0;#__lam_1;#__lam_2;#__lam_3;#__lam_4;#__lam_5;#__lam_6;#__lam_7;#__lam_8;#__lam_9> -> (#__lam_0, #__lam_1, #__lam_2, #__lam_3, #__lam_4, #__lam_5, #__lam_6, #__lam_7, #__lam_8, #__lam_9) -> #body
            """,
        )
