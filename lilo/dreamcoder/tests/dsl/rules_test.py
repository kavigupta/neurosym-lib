import unittest

import neurosym as ns


class TestAllRules(unittest.TestCase):
    def test_basic_arith(self):
        t = ns.TypeDefiner()
        self.maxDiff = None
        self.assertEqual(
            {
                t("i"): [
                    ("+", [t("i"), t("i")]),
                    ("1", []),
                ]
            },
            ns.examples.basic_arith_dsl().all_rules(
                care_about_variables=False, valid_root_types=[t("i")]
            ),
        )
