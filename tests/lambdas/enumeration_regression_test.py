# pylint: disable=duplicate-code
# DSL rendering strings naturally overlap across test files

import unittest

import neurosym as ns


class EnumerationRegressionTest(unittest.TestCase):
    def assertRenderingEqual(self, actual, expected):
        print(actual)
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def rendered_dsl(self, lambdas_kwargs=None, known_types=(), **kwargs):
        if lambdas_kwargs is None:
            lambdas_kwargs = {}
        dslf = ns.DSLFactory(**kwargs)
        dslf.lambdas(**lambdas_kwargs)
        dslf.prune_to(*known_types, tolerate_pruning_entire_productions=True)
        return dslf.finalize().render()

    def test_basic(self):
        self.assertRenderingEqual(
            self.rendered_dsl(known_types=("i", "i -> i", "(i, i) -> i")),
            """
            lam_0 :: L<#body|#__lam_0> -> #__lam_0 -> #body
            lam_1 :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
            $0 :: V<$0>
            $1 :: V<$1>
            $2 :: V<$2>
            $3 :: V<$3>
            """,
        )

    def test_limited_arity(self):
        self.assertRenderingEqual(
            self.rendered_dsl(known_types=("i", "i -> i -> i")),
            """
            lam :: L<#body|#__lam_0> -> #__lam_0 -> #body
            $0 :: V<$0>
            $1 :: V<$1>
            $2 :: V<$2>
            $3 :: V<$3>
            """,
        )

    def test_limited_type_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(
                known_types=("i", "i -> i", "(i, i) -> i", "(i -> i) -> i"),
            ),
            """
            lam_0 :: L<#body|#__lam_0> -> #__lam_0 -> #body
            lam_1 :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
            $0 :: V<$0>
            $1 :: V<$1>
            $2 :: V<$2>
            $3 :: V<$3>
            """,
        )

    def test_limited_type_depth_env_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(
                max_env_depth=2,
                known_types=("i", "i -> i", "(i, i) -> i", "(i -> i) -> i"),
            ),
            """
            lam_0 :: L<#body|#__lam_0> -> #__lam_0 -> #body
            lam_1 :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
            $0 :: V<$0>
            $1 :: V<$1>
            """,
        )
