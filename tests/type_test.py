import unittest

from neurosym.types.type import AtomicType, TensorType, ListType, ArrowType
from neurosym.types.type_string_repr import lex, parse_type, render_type


class TestRender(unittest.TestCase):
    def test_atomic(self):
        self.assertEqual(render_type(AtomicType("int")), "int")

    def test_tensor(self):
        self.assertEqual(
            render_type(TensorType(AtomicType("int"), (3, 4))), "{int, 3, 4}"
        )

    def test_list(self):
        self.assertEqual(render_type(ListType(AtomicType("int"))), "[int]")

    def test_arrow(self):
        self.assertEqual(
            render_type(
                ArrowType((AtomicType("int"), AtomicType("float")), AtomicType("bool"))
            ),
            "(int, float) -> bool",
        )

    def test_arrow_single_arg(self):
        self.assertEqual(
            render_type(ArrowType((AtomicType("int"),), AtomicType("bool"))),
            "int -> bool",
        )

    def test_arrow_single_arg_list(self):
        self.assertEqual(
            render_type(ArrowType((ListType(AtomicType("int")),), AtomicType("bool"))),
            "[int] -> bool",
        )

    def test_arrow_single_arg_tensor(self):
        self.assertEqual(
            render_type(
                ArrowType((TensorType(AtomicType("int"), (3, 4)),), AtomicType("bool"))
            ),
            "{int, 3, 4} -> bool",
        )

    def test_arrow_single_arg_arrow(self):
        self.assertEqual(
            render_type(
                ArrowType(
                    (ArrowType((AtomicType("int"),), AtomicType("bool")),),
                    AtomicType("bool"),
                )
            ),
            "(int -> bool) -> bool",
        )

    def test_arrow_single_arg_arrow_list(self):
        self.assertEqual(
            render_type(
                ArrowType(
                    (ListType(ArrowType((AtomicType("int"),), AtomicType("bool"))),),
                    AtomicType("bool"),
                )
            ),
            "[int -> bool] -> bool",
        )

    def test_arrow_returns_arrow(self):
        self.assertEqual(
            render_type(
                ArrowType(
                    (AtomicType("int"),),
                    ArrowType((AtomicType("int"),), AtomicType("bool")),
                )
            ),
            "int -> int -> bool",
        )

    def test_no_args(self):
        self.assertEqual(
            render_type(ArrowType((), AtomicType("bool"))),
            "() -> bool",
        )

    def test_nested(self):
        self.assertEqual(
            render_type(
                ArrowType(
                    (
                        TensorType(AtomicType("int"), (3, 4)),
                        ListType(AtomicType("float")),
                    ),
                    AtomicType("bool"),
                )
            ),
            "({int, 3, 4}, [float]) -> bool",
        )

    def test_complex(self):
        self.assertEqual(
            render_type(
                ArrowType(
                    input_type=(TensorType(dtype=AtomicType(name="f"), shape=(10,)),),
                    output_type=TensorType(dtype=AtomicType(name="f"), shape=(10,)),
                )
            ),
            "{f, 10} -> {f, 10}",
        )


class TestLex(unittest.TestCase):
    def test_tensor(self):
        self.assertEqual(lex("{int, 3, 4}"), ["{", "int", ",", "3", ",", "4", "}"])

    def test_list(self):
        self.assertEqual(lex("[int]"), ["[", "int", "]"])

    def test_arrow(self):
        self.assertEqual(
            lex("(int, float) -> bool"), ["(", "int", ",", "float", ")", "->", "bool"]
        )

    def test_no_args(self):
        self.assertEqual(lex("() -> bool"), ["(", ")", "->", "bool"])

    def test_dollar(self):
        self.assertEqual(lex("$x"), ["$x"])

    def test_dollar_arrow(self):
        self.assertEqual(lex("$fL -> $fL"), ["$fL", "->", "$fL"])


class TestParse(unittest.TestCase):
    def test_atomic(self):
        self.assertEqual(parse_type("int"), AtomicType("int"))

    def test_tensor(self):
        self.assertEqual(
            parse_type("{int, 3, 4}"), TensorType(AtomicType("int"), (3, 4))
        )

    def test_list(self):
        self.assertEqual(parse_type("[int]"), ListType(AtomicType("int")))

    def test_arrow(self):
        self.assertEqual(
            parse_type("(int, float) -> bool"),
            ArrowType((AtomicType("int"), AtomicType("float")), AtomicType("bool")),
        )

    def test_arrow_single_arg(self):
        self.assertEqual(
            parse_type("int -> bool"),
            ArrowType((AtomicType("int"),), AtomicType("bool")),
        )

    def test_arrow_single_arg_list(self):
        self.assertEqual(
            parse_type("[int] -> bool"),
            ArrowType((ListType(AtomicType("int")),), AtomicType("bool")),
        )

    def test_arrow_returns_arrow(self):
        self.assertEqual(
            parse_type("int -> int -> bool"),
            ArrowType(
                (AtomicType("int"),),
                ArrowType((AtomicType("int"),), AtomicType("bool")),
            ),
        )

    def test_no_args(self):
        self.assertEqual(
            parse_type("() -> bool"),
            ArrowType((), AtomicType("bool")),
        )

    def test_nested(self):
        self.assertEqual(
            parse_type("({int, 3, 4}, [float]) -> bool"),
            ArrowType(
                (
                    TensorType(AtomicType("int"), (3, 4)),
                    ListType(AtomicType("float")),
                ),
                AtomicType("bool"),
            ),
        )

    def test_single_arg(self):
        self.assertEqual(
            render_type(parse_type("([{f, 10}]) -> {f, 20}")),
            "[{f, 10}] -> {f, 20}",
        )

    def test_bad_parse(self):
        self.assertRaises(Exception, lambda: render_type(parse_type("f -> f")))
