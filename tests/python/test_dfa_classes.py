import ast
import re
import unittest
from textwrap import dedent

from parameterized import parameterized

import neurosym as ns
from tests.python.parse_test import small_set_examples

dfa = ns.python_dfa()

reasonable_classifications = [
    ("alias", "alias"),
    ("AnnAssign", "S"),
    ("Assert", "S"),
    ("Assign", "S"),
    ("AsyncFunctionDef", "S"),
    ("AsyncFor", "S"),
    ("AsyncWith", "S"),
    ("Attribute", "E"),
    ("Attribute", "L"),
    ("AugAssign", "S"),
    ("Await", "E"),
    ("BinOp", "E"),
    ("BoolOp", "E"),
    ("Call", "E"),
    ("ClassDef", "S"),
    ("Compare", "E"),
    ("Constant", "E"),
    ("Constant", "F"),
    ("Delete", "S"),
    ("Dict", "E"),
    ("DictComp", "E"),
    *[
        (x, "O")
        for x in [
            "Add",
            "Sub",
            "Mult",
            "MatMult",
            "Div",
            "Mod",
            "Pow",
            "LShift",
            "RShift",
            "BitOr",
            "BitXor",
            "BitAnd",
            "FloorDiv",
            "Invert",
            "Not",
            "UAdd",
            "USub",
            "Eq",
            "NotEq",
            "Lt",
            "LtE",
            "Gt",
            "GtE",
            "Is",
            "IsNot",
            "In",
            "NotIn",
            "And",
            "Or",
        ]
    ],
    ("ExceptHandler", "EH"),
    ("Expr", "S"),
    ("For", "S"),
    ("FormattedValue", "F"),
    ("FunctionDef", "S"),
    ("GeneratorExp", "E"),
    ("Global", "S"),
    ("If", "S"),
    ("IfExp", "E"),
    ("Import", "S"),
    ("ImportFrom", "S"),
    ("JoinedStr", "E"),
    ("JoinedStr", "F"),
    ("Lambda", "E"),
    ("List", "E"),
    ("List", "L"),
    ("ListComp", "E"),
    ("Module", "M"),
    ("Name", "E"),
    ("Name", "L"),
    ("NamedExpr", "E"),
    ("Nonlocal", "S"),
    ("Pass", "S"),
    ("Raise", "S"),
    ("Return", "S"),
    ("Break", "S"),
    ("Continue", "S"),
    ("Set", "E"),
    ("SetComp", "E"),
    ("_slice_content", "SliceRoot"),
    ("_slice_slice", "SliceRoot"),
    ("_slice_tuple", "SliceRoot"),
    ("Tuple", "SliceTuple"),
    ("Slice", "Slice"),
    ("Starred", "L"),
    ("Starred", "Starred"),
    ("_starred_content", "StarredRoot"),
    ("_starred_content", "L"),
    ("_starred_starred", "StarredRoot"),
    ("_starred_starred", "L"),
    ("Subscript", "E"),
    ("Subscript", "L"),
    ("Try", "S"),
    ("Tuple", "E"),
    ("Tuple", "L"),
    ("UnaryOp", "E"),
    ("While", "S"),
    ("With", "S"),
    ("Yield", "E"),
    ("YieldFrom", "E"),
    ("arg", "A"),
    ("arguments", "As"),
    ("comprehension", "C"),
    ("keyword", "K"),
    ("list", "[A]"),
    ("list", "[C]"),
    ("list", "[SliceRoot]"),
    ("list", "[E]"),
    ("list", "[StarredRoot]"),
    ("list", "[EH]"),
    ("list", "[F]"),
    ("list", "[K]"),
    ("list", "[L]"),
    ("list", "[W]"),
    ("list", "[O]"),
    ("list", "[alias]"),
    ("list", "[NameStr]"),
    ("list", "[TI]"),
    ("/seq", "seqS"),
    ("withitem", "W"),
    ("const-None", "E"),
    # formatting
    ("const-None", "F"),
    ("const-s.*", "F"),
    # vararg
    ("const-None", "A"),
    # type
    (".*", "TA"),
    ("const-None", "TC"),
    # left value
    ("const-None", "L"),
    # Load/Store/Del
    ("Load", "Ctx"),
    ("Store", "Ctx"),
    ("Del", "Ctx"),
    # name
    ("const-[&g].*", "Name"),
    ("const-[&g].*", "NullableName"),
    ("const-None", "NullableName"),
    ("const-s.*", "NameStr"),
    ("const-[&g].*", "NameStr"),  # imports
    ("const-s.*", "NullableNameStr"),
    ("const-None", "NullableNameStr"),
    ("const-[&g].*", "NullableNameStr"),
    # values
    ("const-None", "Const"),
    ("const-True", "Const"),
    ("const-False", "Const"),
    ("const-Ellipsis", "Const"),
    ("const-[sbijf].*", "Const"),
    # constkind
    ("const-None", "ConstKind"),
    ("const-s.*", "ConstKind"),
    # constants
    ("const-i[01]", "bool"),
    ("const-i.*", "int"),
]


class TestClassifications(unittest.TestCase):
    def classify_in_code(self, code, start_state):
        classified = [
            (ns.render_s_expression(x), tag)
            for x, tag in ns.run_dfa_on_program(
                dfa, code.to_ns_s_exp(dict()), start_state
            )
            if isinstance(x, ns.SExpression)
        ]
        print(classified)
        return classified

    def test_module_classify(self):
        self.assertEqual(
            self.classify_in_code(ns.python_to_python_ast("x = 2"), "M"),
            [
                (
                    "(Module (/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None)) nil)",
                    "M",
                ),
                (
                    "(/seq (Assign (list (Name &x:0 Store)) (Constant i2 None) None))",
                    "seqS",
                ),
                ("(Assign (list (Name &x:0 Store)) (Constant i2 None) None)", "S"),
                ("(list (Name &x:0 Store))", "[L]"),
                ("(Name &x:0 Store)", "L"),
                ("(Constant i2 None)", "E"),
            ],
        )

    def test_statement_classify(self):
        self.assertEqual(
            self.classify_in_code(ns.python_statement_to_python_ast("x = 2"), "S"),
            [
                (
                    "(Assign (list (Name &x:0 Store)) (Constant i2 None) None)",
                    "S",
                ),
                ("(list (Name &x:0 Store))", "[L]"),
                ("(Name &x:0 Store)", "L"),
                ("(Constant i2 None)", "E"),
            ],
        )


class DFATest(unittest.TestCase):
    def check_reasonable_classification(self, tag_to_check, state_to_check):
        for tag, state in reasonable_classifications:
            mat = re.match("^" + tag + "$", tag_to_check)
            if mat and state == state_to_check:
                return
        self.fail(f"Unknown classification {tag_to_check} {state_to_check}")

    def classify_elements_in_code_with_config(self, code, **kwargs):
        print("#" * 80)
        print(code)
        code = ns.python_to_python_ast(code).to_ns_s_exp(kwargs)
        print(ns.render_s_expression(code))
        classified = ns.run_dfa_on_program(dfa, code, "M")
        result = sorted(
            {
                (x.symbol, state)
                for (x, state) in classified
                if isinstance(x, ns.SExpression)
            }
        )
        print(code)
        for x, state in result:
            self.check_reasonable_classification(x, state)

    def classify_elements_in_code(self, code):
        self.classify_elements_in_code_with_config(code)
        self.classify_elements_in_code_with_config(code, no_leaves=True)

    @parameterized.expand([(i,) for i in range(100)])
    def test_realistic(self, i):
        code = small_set_examples()[i]
        for element in ast.parse(code).body:
            self.classify_elements_in_code(ast.unparse(element))

    def test_function(self):
        self.classify_elements_in_code(
            dedent(
                """
                def f(x):
                    pass
                """
            )
        )
        self.classify_elements_in_code(
            dedent(
                """
                def f(x, *y, z=2, **w):
                    pass
                """
            )
        )

    def test_with(self):
        self.classify_elements_in_code(
            dedent(
                """
                with x:
                    pass
                """
            )
        )
        self.classify_elements_in_code(
            dedent(
                """
                with x as y:
                    pass
                """
            )
        )

    def test_annotation(self):
        self.classify_elements_in_code(
            dedent(
                """
                @asyncio.coroutine
                def onOpen(self):
                    pass
                """
            )
        )

    def test_class(self):
        self.classify_elements_in_code(
            dedent(
                """
                class MyClientProtocol(WebSocketClientProtocol):
                    pass
                """
            )
        )

    def test_exception(self):
        self.classify_elements_in_code(
            dedent(
                """
                try:
                    x = 2
                except:
                    pass
                """
            )
        )

    def test_comparison(self):
        self.classify_elements_in_code("x == 2")

    def test_keyword(self):
        self.classify_elements_in_code("f(x=2)")

    def test_aug_assign(self):
        self.classify_elements_in_code("(x := 2)")

    def test_assign(self):
        self.classify_elements_in_code("x = 2")
        self.classify_elements_in_code("x = 2, 3")
        self.classify_elements_in_code("x += 2")

    def test_tuple(self):
        self.classify_elements_in_code("(2, 3)")

    def test_unpacking(self):
        self.classify_elements_in_code("a, b = 1, 2")
        self.classify_elements_in_code("a, *b = 1, 2, 3")
        self.classify_elements_in_code("[a, b] = 1, 2, 3")

    def test_slicing_direct(self):
        self.classify_elements_in_code("x[2]")
        self.classify_elements_in_code("x[2:3]")
        self.classify_elements_in_code("x[2:3] = 4")

    def test_slicing_tuple(self):
        self.classify_elements_in_code("x[2:3, 3]")
        self.classify_elements_in_code("x[2:3, 3] = 5")

    def test_starred(self):
        self.classify_elements_in_code("f(2, 3, *x)")
        self.classify_elements_in_code("(2, 3, *x)")
        self.classify_elements_in_code("[2, 3, *x]")
        self.classify_elements_in_code("{2, 3, *x}")

    def test_kwstarred(self):
        self.classify_elements_in_code("f(2, 3, **x)")

    def test_import(self):
        self.classify_elements_in_code("import x")
        self.classify_elements_in_code("from x import y")
        self.classify_elements_in_code("from x import y as z")
        self.classify_elements_in_code("from . import x")

    def test_global_nonlocal(self):
        self.classify_elements_in_code("global x")
        self.classify_elements_in_code("nonlocal x")

    def test_joined_str(self):
        self.classify_elements_in_code("f'2 {459.67:.1f}'")

    def test_type_annotation(self):
        self.classify_elements_in_code("x: int = 2")
        self.classify_elements_in_code("def f(x: int) -> int: pass")
        self.classify_elements_in_code("x: List[int] = []")
        self.classify_elements_in_code("x: Tuple[int, int] = (x, y)")

    def test_code(self):
        self.classify_elements_in_code(
            dedent(
                r"""
                {x: 2 for x in range(10)}
                """
            )
        )


class TestExprNodeValidity(unittest.TestCase):
    def e_nodes(self, code):
        print("#" * 80)
        print(code)
        code = ns.python_to_python_ast(code)
        e_nodes = [
            ns.render_s_expression(x)
            for x, state in ns.run_dfa_on_program(dfa, code.to_ns_s_exp(dict()), "M")
            if state == "E" and isinstance(x, ns.SExpression)
        ]
        return e_nodes

    def assertENodeReal(self, node):
        print(node)
        code = ns.s_exp_to_python_ast(node)
        print(code)
        code_in_function_call = ns.make_python_ast.make_call(
            ns.PythonSymbol(name="hi", scope=None), code
        )
        code_in_function_call = code_in_function_call.to_python()
        print(code_in_function_call)
        code_in_function_call = ns.python_statement_to_python_ast(code_in_function_call)
        assert code_in_function_call.typ == ast.Expr
        code_in_function_call = code_in_function_call.children[0]
        assert code_in_function_call.typ == ast.Call
        code_in_function_call = code_in_function_call.children[1]
        code_in_function_call = code_in_function_call.children[0].content
        print(code_in_function_call)
        code_in_function_call = code_in_function_call.to_python()
        print(code_in_function_call)
        self.maxDiff = None
        self.assertEqual(code.to_python(), code_in_function_call)

    def assertENodesReal(self, code):
        e_nodes = self.e_nodes(code)
        for node in e_nodes:
            self.assertENodeReal(node)

    def test_e_nodes_basic(self):
        e_nodes = self.e_nodes("x == 2")
        self.assertEqual(
            e_nodes,
            [
                "(Compare (Name g_x Load) (list Eq) (list (Constant i2 None)))",
                "(Name g_x Load)",
                "(Constant i2 None)",
            ],
        )

    def test_slice(self):
        self.assertENodesReal("y = x[2:3]")

    @parameterized.expand([(i,) for i in range(100)])
    def test_realistic(self, i):
        code = small_set_examples()[i]
        for element in ast.parse(code).body:
            self.assertENodesReal(ast.unparse(element))
