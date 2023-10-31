import unittest

from neurosym.types.type_string_repr import TypeDefiner


class TestAllRules(unittest.TestCase):
    def test_basic_arith(self):
        t = TypeDefiner()
        self.maxDiff = None
        self.assertEqual(
            {
                t("i"): [
                    ("+", [t("i"), t("i")]),
                    ("1", []),
                ]
            },
            ns.examples.basic_arith_dsl().all_rules(
                t("i"), care_about_variables=False, type_depth_limit=float("inf")
            ),
        )
