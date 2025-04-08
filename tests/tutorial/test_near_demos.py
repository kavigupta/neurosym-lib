import json
import unittest

from parameterized import parameterized

from tutorial.process_tutorial import create_skeleton

from .utils import execute_notebook


class TestNearDemos(unittest.TestCase):
    notebooks = [
        "tutorial/discrete_exercise_solutions.ipynb",
    ]

    @parameterized.expand([(path,) for path in notebooks])
    def test_notebook_run(self, path):
        """
        Checks that the notebook runs without errors
        """
        execute_notebook(path, suffix="print('Done')")

    @parameterized.expand([(path,) for path in notebooks])
    def test_notebook_skeleton(self, path):
        with open(path) as f:
            solutions = json.load(f)
        with open(path.replace("solutions", "skeleton")) as f:
            notebook = json.load(f)

        self.assertEqual(create_skeleton(solutions, path), notebook)
