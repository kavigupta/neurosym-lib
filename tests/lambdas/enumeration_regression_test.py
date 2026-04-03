# pylint: disable=duplicate-code
# we don't care about duplicate code in test outputs

import unittest

import neurosym as ns


class EnumerationRegressionTest(unittest.TestCase):
    def assertRenderingEqual(self, actual, expected):
        print(actual)
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def rendered_dsl(self, **kwargs):
        dslf = ns.DSLFactory(**kwargs)
        dslf.lambdas()
        return dslf.finalize().render()

    def test_basic(self):
        self.assertRenderingEqual(
            self.rendered_dsl(),
            """
            lam_0 :: L<#body|> -> () -> #body
            lam_1 :: L<#body|#_arg0> -> (#_arg0) -> #body
            lam_2 :: L<#body|#_arg0;#_arg1> -> (#_arg0;#_arg1) -> #body
            lam_3 :: L<#body|#_arg0;#_arg1;#_arg2> -> (#_arg0;#_arg1;#_arg2) -> #body
            lam_4 :: L<#body|#_arg0;#_arg1;#_arg2;#_arg3> -> (#_arg0;#_arg1;#_arg2;#_arg3) -> #body
            $0 :: V<@0>
            $1 :: V<@1>
            $2 :: V<@2>
            $3 :: V<@3>
            """,
        )

    def test_limited_env_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(max_env_depth=2),
            """
            lam_0 :: L<#body|> -> () -> #body
            lam_1 :: L<#body|#_arg0> -> (#_arg0) -> #body
            lam_2 :: L<#body|#_arg0;#_arg1> -> (#_arg0;#_arg1) -> #body
            $0 :: V<@0>
            $1 :: V<@1>
            """,
        )
