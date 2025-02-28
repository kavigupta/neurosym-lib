import unittest

import neurosym as ns


class TestRender(unittest.TestCase):
    def test_atomic(self):
        self.assertEqual(ns.render_type(ns.AtomicType("int")), "int")

    def test_tensor(self):
        self.assertEqual(
            ns.render_type(ns.TensorType(ns.AtomicType("int"), (3, 4))), "{int, 3, 4}"
        )

    def test_list(self):
        self.assertEqual(ns.render_type(ns.ListType(ns.AtomicType("int"))), "[int]")

    def test_arrow(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (ns.AtomicType("int"), ns.AtomicType("float")),
                    ns.AtomicType("bool"),
                )
            ),
            "(int, float) -> bool",
        )

    def test_arrow_single_arg(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool"))
            ),
            "int -> bool",
        )

    def test_arrow_single_arg_list(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (ns.ListType(ns.AtomicType("int")),), ns.AtomicType("bool")
                )
            ),
            "[int] -> bool",
        )

    def test_arrow_single_arg_tensor(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (ns.TensorType(ns.AtomicType("int"), (3, 4)),),
                    ns.AtomicType("bool"),
                )
            ),
            "{int, 3, 4} -> bool",
        )

    def test_arrow_single_arg_arrow(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool")),),
                    ns.AtomicType("bool"),
                )
            ),
            "(int -> bool) -> bool",
        )

    def test_arrow_single_arg_arrow_list(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (
                        ns.ListType(
                            ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool"))
                        ),
                    ),
                    ns.AtomicType("bool"),
                )
            ),
            "[int -> bool] -> bool",
        )

    def test_arrow_returns_arrow(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (ns.AtomicType("int"),),
                    ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool")),
                )
            ),
            "int -> int -> bool",
        )

    def test_no_args(self):
        self.assertEqual(
            ns.render_type(ns.ArrowType((), ns.AtomicType("bool"))),
            "() -> bool",
        )

    def test_nested(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    (
                        ns.TensorType(ns.AtomicType("int"), (3, 4)),
                        ns.ListType(ns.AtomicType("float")),
                    ),
                    ns.AtomicType("bool"),
                )
            ),
            "({int, 3, 4}, [float]) -> bool",
        )

    def test_complex(self):
        self.assertEqual(
            ns.render_type(
                ns.ArrowType(
                    input_type=(
                        ns.TensorType(dtype=ns.AtomicType(name="f"), shape=(10,)),
                    ),
                    output_type=ns.TensorType(
                        dtype=ns.AtomicType(name="f"), shape=(10,)
                    ),
                )
            ),
            "{f, 10} -> {f, 10}",
        )

    def test_variable(self):
        self.assertEqual(ns.render_type(ns.TypeVariable(name="f")), "#f")


class TestLex(unittest.TestCase):
    def test_tensor(self):
        self.assertEqual(
            ns.lex_type("{int, 3, 4}"), ["{", "int", ",", "3", ",", "4", "}"]
        )

    def test_list(self):
        self.assertEqual(ns.lex_type("[int]"), ["[", "int", "]"])

    def test_arrow(self):
        self.assertEqual(
            ns.lex_type("(int, float) -> bool"),
            ["(", "int", ",", "float", ")", "->", "bool"],
        )

    def test_no_args(self):
        self.assertEqual(ns.lex_type("() -> bool"), ["(", ")", "->", "bool"])

    def test_dollar(self):
        self.assertEqual(ns.lex_type("$x"), ["$x"])

    def test_dollar_arrow(self):
        self.assertEqual(ns.lex_type("$fL -> $fL"), ["$fL", "->", "$fL"])


class TestParse(unittest.TestCase):
    def test_atomic(self):
        self.assertEqual(ns.parse_type("int"), ns.AtomicType("int"))

    def test_tensor(self):
        self.assertEqual(
            ns.parse_type("{int, 3, 4}"), ns.TensorType(ns.AtomicType("int"), (3, 4))
        )

    def test_list(self):
        self.assertEqual(ns.parse_type("[int]"), ns.ListType(ns.AtomicType("int")))

    def test_arrow(self):
        self.assertEqual(
            ns.parse_type("(int, float) -> bool"),
            ns.ArrowType(
                (ns.AtomicType("int"), ns.AtomicType("float")), ns.AtomicType("bool")
            ),
        )

    def test_arrow_single_arg(self):
        self.assertEqual(
            ns.parse_type("int -> bool"),
            ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool")),
        )

    def test_arrow_single_arg_list(self):
        self.assertEqual(
            ns.parse_type("[int] -> bool"),
            ns.ArrowType((ns.ListType(ns.AtomicType("int")),), ns.AtomicType("bool")),
        )

    def test_arrow_returns_arrow(self):
        self.assertEqual(
            ns.parse_type("int -> int -> bool"),
            ns.ArrowType(
                (ns.AtomicType("int"),),
                ns.ArrowType((ns.AtomicType("int"),), ns.AtomicType("bool")),
            ),
        )

    def test_multi_arg_arrow_returns_arrow(self):
        t = "(int, float) -> int -> bool"
        self.assertEqual(
            ns.render_type(ns.parse_type(t)),
            t,
        )

    def test_no_args(self):
        self.assertEqual(
            ns.parse_type("() -> bool"),
            ns.ArrowType((), ns.AtomicType("bool")),
        )

    def test_nested(self):
        self.assertEqual(
            ns.parse_type("({int, 3, 4}, [float]) -> bool"),
            ns.ArrowType(
                (
                    ns.TensorType(ns.AtomicType("int"), (3, 4)),
                    ns.ListType(ns.AtomicType("float")),
                ),
                ns.AtomicType("bool"),
            ),
        )

    def test_single_arg(self):
        self.assertEqual(
            ns.render_type(ns.parse_type("([{f, 10}]) -> {f, 20}")),
            "[{f, 10}] -> {f, 20}",
        )

    def test_unparenthesized_arrow_inside_argument(self):
        t = "(i -> b, [i]) -> [i]"
        self.assertEqual(ns.render_type(ns.parse_type(t)), t)

    def test_bad_parse(self):
        self.assertRaises(Exception, lambda: ns.render_type(ns.parse_type("f -> f]")))

    def test_list_of_arrows(self):
        t = "[i -> i]"
        self.assertEqual(ns.render_type(ns.parse_type(t)), t)
