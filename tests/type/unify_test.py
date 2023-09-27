import unittest

from neurosym.types.type import UnificationError
from neurosym.types.type_string_repr import parse_type


class TestUnify(unittest.TestCase):
    def test_unify_atomic(self):
        self.assertEqual(
            parse_type("int").unify(parse_type("int")),
            dict(),
        )

    def test_unify_atomic_fail(self):
        with self.assertRaises(UnificationError):
            parse_type("int").unify(parse_type("float"))

    def test_unify_var_atomic(self):
        self.assertEqual(
            parse_type("#a").unify(parse_type("int")),
            dict(a=parse_type("int")),
        )

    def test_unify_tensor(self):
        self.assertEqual(
            parse_type("{int, 3, 4}").unify(parse_type("{int, 3, 4}")),
            dict(),
        )

    def test_unify_tensor_var(self):
        self.assertEqual(
            parse_type("{#a, 3, 4}").unify(parse_type("{int, 3, 4}")),
            dict(a=parse_type("int")),
        )
        self.assertEqual(
            parse_type("#a").unify(parse_type("{int, 3, 4}")),
            dict(a=parse_type("{int, 3, 4}")),
        )

    def test_unify_tensor_fail(self):
        with self.assertRaises(UnificationError):
            parse_type("{int, 3, 4}").unify(parse_type("{int, 3, 5}"))

        with self.assertRaises(UnificationError):
            parse_type("{int, 3, 4}").unify(parse_type("{float, 3, 4}"))

    def test_unify_list(self):
        self.assertEqual(
            parse_type("[int]").unify(parse_type("[int]")),
            dict(),
        )

    def test_unify_list_var(self):
        self.assertEqual(
            parse_type("#a").unify(parse_type("[int]")),
            dict(a=parse_type("[int]")),
        )
        self.assertEqual(
            parse_type("[#a]").unify(parse_type("[int]")),
            dict(a=parse_type("int")),
        )
        self.assertEqual(
            parse_type("[#a]").unify(parse_type("[[int]]")),
            dict(a=parse_type("[int]")),
        )

    def test_unify_list_fail(self):
        with self.assertRaises(UnificationError):
            parse_type("[int]").unify(parse_type("[float]"))

    def test_unify_arrow(self):
        self.assertEqual(
            parse_type("(int, float) -> bool").unify(
                parse_type("(int, float) -> bool")
            ),
            dict(),
        )

    def test_unify_arrow_var(self):
        self.assertEqual(
            parse_type("#a").unify(parse_type("(int, float) -> bool")),
            dict(a=parse_type("(int, float) -> bool")),
        )
        self.assertEqual(
            parse_type("(#a, float) -> bool").unify(parse_type("(int, float) -> bool")),
            dict(a=parse_type("int")),
        )
        self.assertEqual(
            parse_type("(#a, #a) -> bool").unify(parse_type("(int, int) -> bool")),
            dict(a=parse_type("int")),
        )

    def test_unify_arrow_fail(self):
        with self.assertRaises(UnificationError):
            parse_type("(int, float) -> bool").unify(parse_type("(int, int) -> bool"))

        with self.assertRaises(UnificationError):
            parse_type("(int, float) -> bool").unify(parse_type("(float, int) -> bool"))

        with self.assertRaises(UnificationError):
            parse_type("(int, float) -> bool").unify(parse_type("(#a, #a) -> bool"))

        with self.assertRaises(UnificationError):
            parse_type("int -> bool").unify(parse_type("#a -> #a"))
