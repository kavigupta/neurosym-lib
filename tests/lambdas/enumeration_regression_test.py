import itertools
import unittest
from neurosym.dsl.dsl_factory import DSLFactory


from neurosym.examples.basic_arith import basic_arith_dsl


class EnumerationRegressionTest(unittest.TestCase):
    def assertRenderingEqual(self, actual, expected):
        print(actual)
        self.assertEqual(
            {line.strip() for line in actual.strip().split("\n")},
            {line.strip() for line in expected.strip().split("\n")},
        )

    def rendered_dsl(self, lambdas_kwargs={}):
        dslf = DSLFactory()
        dslf.known_types("i")
        dslf.lambdas(**lambdas_kwargs)
        return dslf.finalize().render()

    def test_basic(self):
        self.assertRenderingEqual(
            self.rendered_dsl(),
            """
            lam_0 :: L<(i, i) -> i|(i, i) -> i;(i, i) -> i> -> ((i, i) -> i, (i, i) -> i) -> (i, i) -> i
            lam_1 :: L<i -> i|(i, i) -> i;(i, i) -> i> -> ((i, i) -> i, (i, i) -> i) -> i -> i
            lam_2 :: L<i|(i, i) -> i;(i, i) -> i> -> ((i, i) -> i, (i, i) -> i) -> i
            lam_3 :: L<(i, i) -> i|(i, i) -> i;i -> i> -> ((i, i) -> i, i -> i) -> (i, i) -> i
            lam_4 :: L<i -> i|(i, i) -> i;i -> i> -> ((i, i) -> i, i -> i) -> i -> i
            lam_5 :: L<i|(i, i) -> i;i -> i> -> ((i, i) -> i, i -> i) -> i
            lam_6 :: L<(i, i) -> i|(i, i) -> i;i> -> ((i, i) -> i, i) -> (i, i) -> i
            lam_7 :: L<i -> i|(i, i) -> i;i> -> ((i, i) -> i, i) -> i -> i
            lam_8 :: L<i|(i, i) -> i;i> -> ((i, i) -> i, i) -> i
            lam_9 :: L<(i, i) -> i|(i, i) -> i> -> ((i, i) -> i) -> (i, i) -> i
            lam_10 :: L<i -> i|(i, i) -> i> -> ((i, i) -> i) -> i -> i
            lam_11 :: L<i|(i, i) -> i> -> ((i, i) -> i) -> i
            lam_12 :: L<(i, i) -> i|i -> i;(i, i) -> i> -> (i -> i, (i, i) -> i) -> (i, i) -> i
            lam_13 :: L<i -> i|i -> i;(i, i) -> i> -> (i -> i, (i, i) -> i) -> i -> i
            lam_14 :: L<i|i -> i;(i, i) -> i> -> (i -> i, (i, i) -> i) -> i
            lam_15 :: L<(i, i) -> i|i -> i;i -> i> -> (i -> i, i -> i) -> (i, i) -> i
            lam_16 :: L<i -> i|i -> i;i -> i> -> (i -> i, i -> i) -> i -> i
            lam_17 :: L<i|i -> i;i -> i> -> (i -> i, i -> i) -> i
            lam_18 :: L<(i, i) -> i|i -> i;i> -> (i -> i, i) -> (i, i) -> i
            lam_19 :: L<i -> i|i -> i;i> -> (i -> i, i) -> i -> i
            lam_20 :: L<i|i -> i;i> -> (i -> i, i) -> i
            lam_21 :: L<(i, i) -> i|i -> i> -> (i -> i) -> (i, i) -> i
            lam_22 :: L<i -> i|i -> i> -> (i -> i) -> i -> i
            lam_23 :: L<i|i -> i> -> (i -> i) -> i
            lam_24 :: L<(i, i) -> i|i;(i, i) -> i> -> (i, (i, i) -> i) -> (i, i) -> i
            lam_25 :: L<i -> i|i;(i, i) -> i> -> (i, (i, i) -> i) -> i -> i
            lam_26 :: L<i|i;(i, i) -> i> -> (i, (i, i) -> i) -> i
            lam_27 :: L<(i, i) -> i|i;i -> i> -> (i, i -> i) -> (i, i) -> i
            lam_28 :: L<i -> i|i;i -> i> -> (i, i -> i) -> i -> i
            lam_29 :: L<i|i;i -> i> -> (i, i -> i) -> i
            lam_30 :: L<(i, i) -> i|i;i> -> (i, i) -> (i, i) -> i
            lam_31 :: L<i -> i|i;i> -> (i, i) -> i -> i
            lam_32 :: L<i|i;i> -> (i, i) -> i
            lam_33 :: L<(i, i) -> i|i> -> i -> (i, i) -> i
            lam_34 :: L<i -> i|i> -> i -> i -> i
            lam_35 :: L<i|i> -> i -> i
            $0_0 :: V<(i, i) -> i@0>
            $1_1 :: V<(i, i) -> i@1>
            $2_2 :: V<(i, i) -> i@2>
            $3_3 :: V<(i, i) -> i@3>
            $0_4 :: V<i -> i@0>
            $1_5 :: V<i -> i@1>
            $2_6 :: V<i -> i@2>
            $3_7 :: V<i -> i@3>
            $0_8 :: V<i@0>
            $1_9 :: V<i@1>
            $2_10 :: V<i@2>
            $3_11 :: V<i@3>
            """,
        )

    def test_limited_arity(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_arity=1)),
            """
            lam_0 :: L<i -> i|i -> i> -> (i -> i) -> i -> i
            lam_1 :: L<i|i -> i> -> (i -> i) -> i
            lam_2 :: L<i -> i|i> -> i -> i -> i
            lam_3 :: L<i|i> -> i -> i
            $0_0 :: V<i -> i@0>
            $1_1 :: V<i -> i@1>
            $2_2 :: V<i -> i@2>
            $3_3 :: V<i -> i@3>
            $0_4 :: V<i@0>
            $1_5 :: V<i@1>
            $2_6 :: V<i@2>
            $3_7 :: V<i@3>
            """,
        )

    def test_limited_type_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_type_depth=3.5)),
            """
            lam_0 :: L<i -> i|i -> i> -> (i -> i) -> i -> i
            lam_1 :: L<i|i -> i> -> (i -> i) -> i
            lam_2 :: L<i|i;i> -> (i, i) -> i
            lam_3 :: L<i -> i|i> -> i -> i -> i
            lam_4 :: L<i|i> -> i -> i
            $0_0 :: V<i -> i@0>
            $1_1 :: V<i -> i@1>
            $2_2 :: V<i -> i@2>
            $3_3 :: V<i -> i@3>
            $0_4 :: V<i@0>
            $1_5 :: V<i@1>
            $2_6 :: V<i@2>
            $3_7 :: V<i@3>
            """,
        )

    def test_limited_type_depth_env_depth(self):
        self.assertRenderingEqual(
            self.rendered_dsl(lambdas_kwargs=dict(max_type_depth=3.5, max_env_depth=2)),
            """
            lam_0 :: L<i -> i|i -> i> -> (i -> i) -> i -> i
            lam_1 :: L<i|i -> i> -> (i -> i) -> i
            lam_2 :: L<i|i;i> -> (i, i) -> i
            lam_3 :: L<i -> i|i> -> i -> i -> i
            lam_4 :: L<i|i> -> i -> i
            $0_0 :: V<i -> i@0>
            $1_1 :: V<i -> i@1>
            $0_2 :: V<i@0>
            $1_3 :: V<i@1>
            """,
        )
