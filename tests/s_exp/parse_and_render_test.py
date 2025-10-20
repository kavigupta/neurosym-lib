import unittest

import neurosym as ns


class TestParseRender(unittest.TestCase):
    def assertParseRender(self, s_exp, text, **kwargs):
        self.assertEqual(ns.render_s_expression(s_exp, **kwargs), text)
        self.assertEqual(str(ns.parse_s_expression(text, **kwargs)), str(s_exp))

    def test_basic_render(self):
        self.assertParseRender(
            ns.SExpression("foo", (ns.SExpression("bar", ()),)),
            "(foo (bar))",
        )
        self.assertParseRender(
            ns.SExpression("foo", (ns.SExpression("bar", ()),)),
            "(foo leaf-bar)",
            for_stitch=True,
        )

    def test_variable_render(self):
        self.assertParseRender(
            ns.SExpression("foo", (ns.SExpression("$0", ()),)),
            "(foo ($0))",
        )
        self.assertParseRender(
            ns.SExpression("foo", (ns.SExpression("$0", ()),)),
            "(foo $0)",
            for_stitch=True,
        )

    def test_parse_with_head(self):
        self.maxDiff = 1000
        # somewhat nonstandard, we need to pass in allow_sexp_head=True
        self.assertParseRender(
            ns.SExpression(ns.SExpression("f", ("x",)), (ns.SExpression("g", ()),)),
            "((f x) (g))",
            allow_sexp_head=True,
        )
