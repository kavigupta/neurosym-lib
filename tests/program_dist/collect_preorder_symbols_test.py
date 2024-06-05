import unittest

import neurosym as ns

from .bigram_test import fam_with_vars


class CollectPreorderSymbolsTest(unittest.TestCase):
    def assertCollectPreorder(self, expected, code, tree_dist, **kwargs):
        result = ns.collect_preorder_symbols(
            ns.parse_s_expression(code), tree_dist, **kwargs
        )
        result = [
            (ns.render_s_expression(s_exp), [tree_dist.symbols[idx][0] for idx in alts])
            for s_exp, alts, _ in result
        ]

        print(result)
        self.assertEqual(result, expected)

    def test_collect_preorder_symbols(self):
        self.assertCollectPreorder(
            [
                ("(+ (1) (call (lam ($0_0)) (1)))", ["+", "1", "2", "call"]),
                ("(1)", ["+", "1", "2", "call"]),
                ("(call (lam ($0_0)) (1))", ["+", "1", "2", "call"]),
                ("(lam ($0_0))", ["lam"]),
                ("($0_0)", ["$0_0", "+", "1", "2", "call"]),
                ("(1)", ["+", "1", "2", "call"]),
            ],
            "(+ (1) (call (lam ($0_0)) (1)))",
            fam_with_vars.tree_distribution_skeleton,
        )

    def test_collect_preorder_symbols_replace_node(self):
        self.assertCollectPreorder(
            [
                ("(call (lam (+ (1) (1))) (1))", ["+", "1", "2", "call"]),
                ("(lam (+ (1) (1)))", ["lam"]),
                # note that it's now (+ ($0_0) ($0_0)) instead of (+ (1) (1))
                ("(+ ($0_0) ($0_0))", ["$0_0", "+", "1", "2", "call"]),
                ("($0_0)", ["$0_0", "+", "1", "2", "call"]),
                ("($0_0)", ["$0_0", "+", "1", "2", "call"]),
                ("(1)", ["+", "1", "2", "call"]),
            ],
            "(call (lam (+ (1) (1))) (1))",
            fam_with_vars.tree_distribution_skeleton,
            replace_node_midstream=lambda sexp, _1, _2, _3: (
                ns.parse_s_expression("(+ ($0_0) ($0_0))")
                if sexp.symbol == "+"
                else sexp
            ),
        )
