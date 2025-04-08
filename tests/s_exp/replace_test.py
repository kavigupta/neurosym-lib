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


class TestReplaceFirst(unittest.TestCase):
    def compute_dsl(self):
        dslf = ns.DSLFactory()
        dslf.production("1", "() -> i", lambda: 1)

        def counter_fn(counter):
            counter[0] += 1
            return counter[0]

        dslf.production("c", "() -> i", counter_fn, dict(counter=lambda: [0]))
        dslf.production(
            "+",
            "(i, i) -> i",
            lambda x, y, counter: x + y + counter_fn(counter),
            dict(counter=lambda: [0]),
        )
        dsl = dslf.finalize()
        return dsl

    def test_replace_one_of_multiple(self):
        dsl = self.compute_dsl()
        a = dsl.initialize(ns.parse_s_expression("(+ (+ (1) (c)) (c))"))
        a_states = list(a.all_state_values())
        b = dsl.initialize(ns.parse_s_expression("(c)"))
        c, replaced = a.replace_first("c", b)
        self.assertTrue(replaced)
        self.assertEqual(
            ns.render_s_expression(c.uninitialize()), "(+ (+ (1) (c)) (c))"
        )
        # exact copy of the right child
        self.assertIs(a.children[1], c.children[1])
        # exact copy of the left child of the left child
        self.assertIs(a.children[0].children[0], c.children[0].children[0])
        # b placed in
        self.assertIs(b, c.children[0].children[1])
        # a is not mutated.
        a_states_new = list(a.all_state_values())
        self.assertEqual(len(a_states), len(a_states_new))
        for old, new in zip(a_states, a_states_new):
            self.assertIs(old, new)

    def test_replace_in_s_expression(self):
        dsl = self.compute_dsl()
        a = ns.parse_s_expression("(+ (+ (1) (c)) (c))")
        b = dsl.initialize(ns.parse_s_expression("(+ (1) (c))"))
        c, replaced = a.replace_first("c", b)
        self.assertTrue(replaced)
        self.assertEqual(ns.render_s_expression(c), "(+ (+ (1) (+ (1) (c))) (c))")
