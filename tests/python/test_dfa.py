import json
import unittest

import neurosym as ns

refresh = False


class TestDFARegression(unittest.TestCase):
    def test_python_dfa(self):
        if refresh:
            with open("test_data/dfa.json", "w") as f:
                json.dump(ns.python_dfa(), f, indent=2)
        else:
            with open("test_data/dfa.json") as f:
                dfa = json.load(f)
            self.assertEqual(
                ns.python_dfa(),
                dfa,
                "The DFA does not match the expected output. Set refresh=True to update the test data.",
            )

    def test_refresh(self):
        self.assertTrue(not refresh, "You left it on refresh=True.")
