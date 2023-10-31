import unittest

import neurosym as ns


class EnumerationRegressionTest(unittest.TestCase):
    def assertRenderingEqual(self, actual, expected):
        print(actual)
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def rendered_dsl(self, lambdas_kwargs=None):
        if lambdas_kwargs is None:
            lambdas_kwargs = {}
        dslf = ns.DSLFactory()
        dslf.known_types("i")
        dslf.lambdas(**lambdas_kwargs)
        return dslf.finalize().render()

    def test_basic(self):
        self.assertRenderingEqual(
            self.rendered_dsl(),
            """
            lam_0 :: L<#body|(i, i) -> i;(i, i) -> i> -> ((i, i) -> i, (i, i) -> i) -> #body
            lam_1 :: L<#body|(i, i) -> i;i -> i> -> ((i, i) -> i, i -> i) -> #body
            lam_2 :: L<#body|(i, i) -> i;i> -> ((i, i) -> i, i) -> #body
            lam_3 :: L<#body|(i, i) -> i> -> ((i, i) -> i) -> #body
            lam_4 :: L<#body|i -> i;(i, i) -> i> -> (i -> i, (i, i) -> i) -> #body
            lam_5 :: L<#body|i -> i;i -> i> -> (i -> i, i -> i) -> #body
            lam_6 :: L<#body|i -> i;i> -> (i -> i, i) -> #body
            lam_7 :: L<#body|i -> i> -> (i -> i) -> #body
            lam_8 :: L<#body|i;(i, i) -> i> -> (i, (i, i) -> i) -> #body
            lam_9 :: L<#body|i;i -> i> -> (i, i -> i) -> #body
            lam_10 :: L<#body|i;i> -> (i, i) -> #body
            lam_11 :: L<#body|i> -> i -> #body
            $0_0 :: V<(i, i) -> i@0>
            $1_0 :: V<(i, i) -> i@1>
            $2_0 :: V<(i, i) -> i@2>
            $3_0 :: V<(i, i) -> i@3>
            $0_1 :: V<i -> i@0>
            $1_1 :: V<i -> i@1>
            $2_1 :: V<i -> i@2>
            $3_1 :: V<i -> i@3>
            $0_2 :: V<i@0>
            $1_2 :: V<i@1>
            $2_2 :: V<i@2>
            $3_2 :: V<i@3>
            """,
        )

    def test_limited_arity(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_arity=1)),
            """
            lam_0 :: L<#body|i -> i> -> (i -> i) -> #body
            lam_1 :: L<#body|i> -> i -> #body
            $0_0 :: V<i -> i@0>
            $1_0 :: V<i -> i@1>
            $2_0 :: V<i -> i@2>
            $3_0 :: V<i -> i@3>
            $0_1 :: V<i@0>
            $1_1 :: V<i@1>
            $2_1 :: V<i@2>
            $3_1 :: V<i@3>
            """,
        )

    def test_limited_type_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_type_depth=3.5)),
            """
            lam_0 :: L<#body|i -> i> -> (i -> i) -> #body
            lam_1 :: L<#body|i;i> -> (i, i) -> #body
            lam_2 :: L<#body|i> -> i -> #body
            $0_0 :: V<i -> i@0>
            $1_0 :: V<i -> i@1>
            $2_0 :: V<i -> i@2>
            $3_0 :: V<i -> i@3>
            $0_1 :: V<i@0>
            $1_1 :: V<i@1>
            $2_1 :: V<i@2>
            $3_1 :: V<i@3>
            """,
        )

    def test_limited_type_depth_env_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_type_depth=3.5, max_env_depth=2)),
            """
            lam_0 :: L<#body|i -> i> -> (i -> i) -> #body
            lam_1 :: L<#body|i;i> -> (i, i) -> #body
            lam_2 :: L<#body|i> -> i -> #body
            $0_0 :: V<i -> i@0>
            $1_0 :: V<i -> i@1>
            $0_1 :: V<i@0>
            $1_1 :: V<i@1>
            """,
        )
