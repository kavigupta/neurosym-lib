import unittest

from neurosym.examples.basic_arith import basic_arith_dsl
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
            basic_arith_dsl().all_rules(t("i"), care_about_variables=False),
        )
