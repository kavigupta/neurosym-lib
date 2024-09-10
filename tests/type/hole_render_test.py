import unittest

import neurosym as ns


class TestHoleRender(unittest.TestCase):
    def test_empty_environment(self):
        self.assertEqual(
            ns.Hole(
                1234,
                ns.TypeWithEnvironment(ns.parse_type("x -> y"), ns.Environment.empty()),
            ).__to_pair__(False),
            "??::<x -> y>",
        )

    def test_permissive_environment(self):
        self.assertEqual(
            ns.Hole(
                1234,
                ns.TypeWithEnvironment(
                    ns.parse_type("x -> y"), ns.PermissiveEnvironmment()
                ),
            ).__to_pair__(False),
            "??::<x -> y|*>",
        )

    def test_single_element_environment(self):
        self.assertEqual(
            ns.Hole(
                1234,
                ns.TypeWithEnvironment(
                    ns.parse_type("x -> y"),
                    ns.Environment.empty().child(ns.parse_type("z")),
                ),
            ).__to_pair__(False),
            "??::<x -> y|0=z>",
        )

    def test_multiple_element_environment(self):
        self.assertEqual(
            ns.Hole(
                1234,
                ns.TypeWithEnvironment(
                    ns.parse_type("x -> y"),
                    ns.Environment.empty()
                    .child(ns.parse_type("a1"), ns.parse_type("a2"))
                    .child(ns.parse_type("b1"), ns.parse_type("b2")),
                ),
            ).__to_pair__(False),
            "??::<x -> y|0=b2,1=b1,2=a2,3=a1>",
        )
