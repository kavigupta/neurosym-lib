import ast
import json
import unittest
from functools import lru_cache

from parameterized import parameterized

import neurosym as ns


class ParseUnparseInverseTest(unittest.TestCase):
    def canonicalize(self, python_code):
        return ast.unparse(ast.parse(python_code))

    def assert_valid_s_exp(self, s_exp, no_leaves):
        if not isinstance(s_exp, ns.SExpression):
            assert isinstance(s_exp, str)
            if no_leaves:
                self.fail(f"leaf: {s_exp}")
            return
        if s_exp.symbol not in {"/seq"}:
            self.assertTrue(isinstance(s_exp.symbol, str), repr(s_exp.symbol))
            if not no_leaves:
                self.assertTrue(len(s_exp.children) >= 1, repr(s_exp))
        for y in s_exp.children:
            self.assert_valid_s_exp(y, no_leaves)

    def check_s_exp(self, s_exp, no_leaves=False):
        self.maxDiff = None
        parsed = ns.s_exp_to_python_ast(s_exp)
        print(parsed)
        s_exp_update = ns.render_s_expression(ns.parse_s_expression(s_exp))
        self.assertEqual(
            s_exp_update,
            ns.render_s_expression(parsed.to_ns_s_exp(dict(no_leaves=no_leaves))),
        )

    def check_with_args(self, test_code, no_leaves=False):
        test_code = self.canonicalize(test_code)
        s_exp = ns.python_to_s_exp(
            test_code, renderer_kwargs=dict(columns=80), no_leaves=no_leaves
        )
        self.assert_valid_s_exp(ns.parse_s_expression(s_exp), no_leaves=no_leaves)
        self.check_s_exp(s_exp, no_leaves=no_leaves)
        s_exp_parsed = ns.s_exp_to_python_ast(s_exp, {})
        print(repr(s_exp_parsed))
        modified = ns.s_exp_to_python(s_exp)
        self.assertEqual(test_code, modified)

    def check(self, test_code):
        self.check_with_args(test_code)
        self.check_with_args(test_code, no_leaves=True)

    def test_basic_one_liners(self):
        self.check("x = 2")
        self.check("7")
        self.check("import abc")

    def test_imports(self):
        self.check("import os")
        self.check("from os import path")
        self.check("import os.path")

    def test_sequence_of_statements(self):
        self.maxDiff = None
        self.check("x = 2\ny = 3\nz = 4")
        self.assertEqual(
            ns.python_to_s_exp(
                "x = 2\ny = 3\nz = 4", renderer_kwargs=dict(columns=80000)
            ),
            ns.render_s_expression(
                ns.parse_s_expression(
                    """
                    (Module
                        (/seq
                            (Assign (list (Name &x:0 Store)) (Constant i2 None) None)
                            (Assign (list (Name &y:0 Store)) (Constant i3 None) None)
                            (Assign (list (Name &z:0 Store)) (Constant i4 None) None))
                        nil)
                    """
                )
            ),
        )

    def test_globals(self):
        self.assertEqual(
            ns.python_to_s_exp("import os", renderer_kwargs=dict(columns=80)),
            "(Module (/seq (Import (list (alias g_os None)))) nil)",
        )

    def test_builtins(self):
        self.check("print(True)")
        self.check("0")
        self.check("x = None")

    def test_if_expr(self):
        self.check("2 if x == 3 else 4")

    def test_strings(self):
        self.check("' '")
        self.check("x = 'abc '")
        self.check("x = 'a=é b=\\udc80 d=\U0010ffff'")
        self.check("x = '\\uABCD'")

    def test_lambda(self):
        self.check("lambda: 1 + 2")

    def test_subscript_basic(self):
        self.check("x[2]")

    def test_subscript_tuple(self):
        self.check("x[2, 3]")

    def test_if(self):
        self.check("if True: pass")

    def test_unparse_sequence(self):
        # should work with or without the Module wrapper
        self.assertEqual(
            ns.s_exp_to_python(
                "(Module (/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None)) nil)"
            ),
            "x = 2",
        )

        self.assertEqual(
            ns.s_exp_to_python(
                "(/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None))"
            ),
            "x = 2",
        )

    @parameterized.expand([(i,) for i in range(100)])
    def test_realistic(self, i):
        try:
            print(small_set_examples()[i])
            self.check(small_set_examples()[i])
        except Exception as e:
            self.assertFalse(f"Error: {e}")
            raise e

    def test_subscript_s_exp(self):
        self.check_s_exp(
            "(_slice_tuple (Tuple (list (_slice_content (Constant i3 None))) Load))"
        )


@lru_cache(None)
def small_set_examples():
    with open("test_data/small_set.json") as f:
        return json.load(f)
