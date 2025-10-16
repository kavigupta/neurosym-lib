import unittest

import neurosym as ns
from neurosym.examples import near

ldsl = ns.examples.dreamcoder.list_dsl("[i] -> i")


class TestListDSL(unittest.TestCase):

    def test_show_dsl(self):
        actual = ldsl.render()
        print(actual)
        expected = """
              0 :: () -> i
              1 :: () -> i
              2 :: () -> i
              3 :: () -> i
              4 :: () -> i
              5 :: () -> i
        empty_0 :: () -> [b -> b]
        empty_1 :: () -> [(i, b) -> b]
        empty_2 :: () -> [(i, b) -> i]
        empty_3 :: () -> [(i, b, b) -> b]
        empty_4 :: () -> [(i, b, i) -> b]
        empty_5 :: () -> [(i, i) -> b]
        empty_6 :: () -> [(i, i) -> i]
        empty_7 :: () -> [(i, i, b) -> i]
        empty_8 :: () -> [(i, i, i) -> i]
        empty_9 :: () -> [i -> b]
       empty_10 :: () -> [[b] -> b]
       empty_11 :: () -> [[i] -> b]
       empty_12 :: () -> [[i] -> i]
       empty_13 :: () -> [b]
       empty_14 :: () -> [i]
       empty_15 :: () -> [[b -> b]]
       empty_16 :: () -> [[i -> b]]
       empty_17 :: () -> [[b]]
       empty_18 :: () -> [[i]]
       empty_19 :: () -> [[[b]]]
       empty_20 :: () -> [[[i]]]
      singleton :: #T -> [#T]
          range :: i -> [i]
             ++ :: ([#T], [#T]) -> [#T]
         mapi_0 :: ((i, b) -> #R, [b]) -> [#R]
         mapi_1 :: ((i, i) -> #R, [i]) -> [#R]
      reducei_0 :: ((i, #R, b) -> #R, #R, [b]) -> #R
      reducei_1 :: ((i, #R, i) -> #R, #R, [i]) -> #R
           true :: () -> b
            not :: b -> b
            and :: (b, b) -> b
             or :: (b, b) -> b
              i :: (b, #T, #T) -> #T
           sort :: [#T] -> [#T]
              + :: (i, i) -> i
              * :: (i, i) -> i
         negate :: i -> i
            mod :: (i, i) -> i
            eq? :: (i, i) -> b
            gt? :: (i, i) -> b
       is-prime :: i -> b
      is-square :: i -> b
            sum :: [i] -> i
        reverse :: [#T] -> [#T]
          all_0 :: (b -> b, [b]) -> b
          all_1 :: (i -> b, [i]) -> b
          all_2 :: ([b] -> b, [[b]]) -> b
          all_3 :: ([i] -> b, [[i]]) -> b
          any_0 :: (b -> b, [b]) -> b
          any_1 :: (i -> b, [i]) -> b
          any_2 :: ([b] -> b, [[b]]) -> b
          any_3 :: ([i] -> b, [[i]]) -> b
          index :: (i, [#T]) -> #T
         filter :: (#T -> b, [#T]) -> [#T]
          slice :: (i, i, [#T]) -> [#T]
          lam_0 :: L<#body|b> -> b -> #body
          lam_1 :: L<#body|i;b> -> (i, b) -> #body
          lam_2 :: L<#body|i;b;b> -> (i, b, b) -> #body
          lam_3 :: L<#body|i;b;i> -> (i, b, i) -> #body
          lam_4 :: L<#body|i;i> -> (i, i) -> #body
          lam_5 :: L<#body|i;i;b> -> (i, i, b) -> #body
          lam_6 :: L<#body|i;i;i> -> (i, i, i) -> #body
          lam_7 :: L<#body|i> -> i -> #body
          lam_8 :: L<#body|[b]> -> [b] -> #body
          lam_9 :: L<#body|[i]> -> [i] -> #body
           $0_0 :: V<b@0>
           $1_0 :: V<b@1>
           $2_0 :: V<b@2>
           $3_0 :: V<b@3>
           $0_1 :: V<i@0>
           $1_1 :: V<i@1>
           $2_1 :: V<i@2>
           $3_1 :: V<i@3>
           $0_2 :: V<[b]@0>
           $1_2 :: V<[b]@1>
           $2_2 :: V<[b]@2>
           $3_2 :: V<[b]@3>
           $0_3 :: V<[i]@0>
           $1_3 :: V<[i]@1>
           $2_3 :: V<[i]@2>
           $3_3 :: V<[i]@3>
        """
        self.assertEqual(
            {line.strip() for line in expected.strip().split("\n")},
            {line.strip() for line in actual.strip().split("\n")},
        )

    def test_basic_dsl(self):
        self.maxDiff = None
        dsl = ldsl

        def is_goal(x):
            try:
                fn = dsl.compute(dsl.initialize(x.program))
                if fn([1, 2, 3]) != 2:
                    return False
                if fn([0, 7, 3, 5]) != 7:
                    return False
                return True
            except:  # pylint: disable=bare-except
                return False

        g = near.near_graph(
            dsl,
            ns.parse_type("[i] -> i"),
            is_goal=is_goal,
            cost=lambda x: 0,
        )
        it = ns.search.bfs(g)
        node = next(it)
        self.assertEqual(
            ns.render_s_expression(node),
            "(lam_11 (index (1) ($0_3)))",
        )
