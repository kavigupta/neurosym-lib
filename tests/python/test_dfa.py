import json
import unittest

import neurosym as ns


class TestDFARegression(unittest.TestCase):
    def test_python_dfa(self):
        with open("test_data/dfa.json") as f:
            dfa = json.load(f)
        self.assertEqual(ns.python_dfa(), dfa)
