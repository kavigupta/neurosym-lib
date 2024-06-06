import json
import unittest

from tutorial.process_tutorial import create_skeleton


class TestTutorial2(unittest.TestCase):
    def test_tutorial_2_solutions(self):
        with open("tutorial/tutorial2_discrete_solutions.ipynb") as f:
            solutions = json.load(f)
        with open("tutorial/tutorial2_discrete.ipynb") as f:
            notebook = json.load(f)

        self.assertEqual(create_skeleton(solutions), notebook)
