import unittest

import neurosym as ns


class TestReplaceSymbol(unittest.TestCase):
    def test_basic(self):
        s_exp = ns.parse_s_expression("(foo (bar) (baz))")
        id_to_symbol = {id(s_exp.children[0]): "abc", id(s_exp): "def"}
        replaced = s_exp.replace_symbols_by_id(id_to_symbol)
        self.assertEqual(ns.render_s_expression(replaced), "(def (abc) (baz))")

    def test_with_duplicates(self):
        s_exp = ns.parse_s_expression("(foo (bar) (bar) (bar))")
        id_to_symbol = {id(s_exp.children[0]): "abc", id(s_exp.children[1]): "def"}
        replaced = s_exp.replace_symbols_by_id(id_to_symbol)
        self.assertEqual(ns.render_s_expression(replaced), "(foo (abc) (def) (bar))")


class TestReplaceNode(unittest.TestCase):
    def test_basic(self):
        s_exp = ns.parse_s_expression("(foo (bar) (baz))")
        id_to_node = {
            id(s_exp.children[0]): ns.parse_s_expression("(uvw)"),
            id(s_exp.children[1]): ns.parse_s_expression("(abc (def (ghi)))"),
        }
        replaced = s_exp.replace_nodes_by_id(id_to_node)
        self.assertEqual(
            ns.render_s_expression(replaced), "(foo (uvw) (abc (def (ghi))))"
        )

    def test_with_duplicates(self):
        s_exp = ns.parse_s_expression("(foo (bar) (bar) (bar))")
        id_to_node = {
            id(s_exp.children[0]): ns.parse_s_expression("(uvw)"),
            id(s_exp): ns.parse_s_expression("(abc (def (ghi)))"),
        }
        replaced = s_exp.replace_nodes_by_id(id_to_node)
        self.assertEqual(ns.render_s_expression(replaced), "(abc (def (ghi)))")
