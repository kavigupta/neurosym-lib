# pylint: disable=duplicate-code
# we don't care about duplicate code in test outputs

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
          empty :: () -> [#T]
      singleton :: #T -> [#T]
          range :: i -> [i]
             ++ :: ([#T], [#T]) -> [#T]
         mapi_0 :: ((i, b) -> #R, [b]) -> [#R]
         mapi_1 :: ((i, i) -> #R, [i]) -> [#R]
         mapi_2 :: ((i, [b]) -> #R, [[b]]) -> [#R]
         mapi_3 :: ((i, [i]) -> #R, [[i]]) -> [#R]
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
          lam_0 :: L<#body|#__lam_0> -> #__lam_0 -> #body
          lam_1 :: L<#body|#__lam_0;#__lam_1> -> (#__lam_0, #__lam_1) -> #body
          lam_2 :: L<#body|#__lam_0;#__lam_1;#__lam_2> -> (#__lam_0, #__lam_1, #__lam_2) -> #body
             $0 :: V<$0>
             $1 :: V<$1>
             $2 :: V<$2>
             $3 :: V<$3>
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
        it = ns.search.BFS()(g)
        node = next(it)
        self.assertEqual(
            ns.render_s_expression(node),
            "(lam_0 (index (1) ($0)))",
        )
