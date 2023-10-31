import unittest

import neurosym as ns


class TestUnify(unittest.TestCase):
    def test_unify_atomic(self):
        self.assertEqual(
            ns.parse_type("int").unify(ns.parse_type("int")),
            dict(),
        )

    def test_unify_atomic_fail(self):
        with self.assertRaises(ns.UnificationError):
            ns.parse_type("int").unify(ns.parse_type("float"))

    def test_unify_var_atomic(self):
        self.assertEqual(
            ns.parse_type("#a").unify(ns.parse_type("int")),
            dict(a=ns.parse_type("int")),
        )

    def test_unify_tensor(self):
        self.assertEqual(
            ns.parse_type("{int, 3, 4}").unify(ns.parse_type("{int, 3, 4}")),
            dict(),
        )

    def test_unify_tensor_var(self):
        self.assertEqual(
            ns.parse_type("{#a, 3, 4}").unify(ns.parse_type("{int, 3, 4}")),
            dict(a=ns.parse_type("int")),
        )
        self.assertEqual(
            ns.parse_type("#a").unify(ns.parse_type("{int, 3, 4}")),
            dict(a=ns.parse_type("{int, 3, 4}")),
        )

    def test_unify_tensor_fail(self):
        with self.assertRaises(ns.UnificationError):
            ns.parse_type("{int, 3, 4}").unify(ns.parse_type("{int, 3, 5}"))

        with self.assertRaises(ns.UnificationError):
            ns.parse_type("{int, 3, 4}").unify(ns.parse_type("{float, 3, 4}"))

    def test_unify_list(self):
        self.assertEqual(
            ns.parse_type("[int]").unify(ns.parse_type("[int]")),
            dict(),
        )

    def test_unify_list_var(self):
        self.assertEqual(
            ns.parse_type("#a").unify(ns.parse_type("[int]")),
            dict(a=ns.parse_type("[int]")),
        )
        self.assertEqual(
            ns.parse_type("[#a]").unify(ns.parse_type("[int]")),
            dict(a=ns.parse_type("int")),
        )
        self.assertEqual(
            ns.parse_type("[#a]").unify(ns.parse_type("[[int]]")),
            dict(a=ns.parse_type("[int]")),
        )

    def test_unify_list_fail(self):
        with self.assertRaises(ns.UnificationError):
            ns.parse_type("[int]").unify(ns.parse_type("[float]"))

    def test_unify_arrow(self):
        self.assertEqual(
            ns.parse_type("(int, float) -> bool").unify(
                ns.parse_type("(int, float) -> bool")
            ),
            dict(),
        )

    def test_unify_arrow_var(self):
        self.assertEqual(
            ns.parse_type("#a").unify(ns.parse_type("(int, float) -> bool")),
            dict(a=ns.parse_type("(int, float) -> bool")),
        )
        self.assertEqual(
            ns.parse_type("(#a, float) -> bool").unify(
                ns.parse_type("(int, float) -> bool")
            ),
            dict(a=ns.parse_type("int")),
        )
        self.assertEqual(
            ns.parse_type("(#a, #a) -> bool").unify(
                ns.parse_type("(int, int) -> bool")
            ),
            dict(a=ns.parse_type("int")),
        )

    def test_unify_arrow_fail(self):
        with self.assertRaises(ns.UnificationError):
            ns.parse_type("(int, float) -> bool").unify(
                ns.parse_type("(int, int) -> bool")
            )

        with self.assertRaises(ns.UnificationError):
            ns.parse_type("(int, float) -> bool").unify(
                ns.parse_type("(float, int) -> bool")
            )

        with self.assertRaises(ns.UnificationError):
            ns.parse_type("(int, float) -> bool").unify(
                ns.parse_type("(#a, #a) -> bool")
            )

        with self.assertRaises(ns.UnificationError):
            ns.parse_type("int -> bool").unify(ns.parse_type("#a -> #a"))
