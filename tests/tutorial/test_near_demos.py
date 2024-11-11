import json
import unittest
from parameterized import parameterized

import numpy as np

from tutorial.process_tutorial import create_skeleton

from .utils import execute_notebook


class TestNearDemos(unittest.TestCase):
    def test_near_demo_classification(self):
        result = execute_notebook(
            "tutorial/near_demo_classification.ipynb",
            suffix="import json; print('*' * 80); print(json.dumps(lin.weight.tolist()))",
        )

        *_, stars, weight, _ = result.split("\n")
        assert stars == "*" * 80
        weight = np.array(json.loads(weight))
        x, y = weight[1] - weight[0]
        ang = np.arctan2(y, x)
        ang = np.degrees(ang)

        # The angle should be between -80 and -100 degrees
        # representing an downward normal vector
        self.assertGreaterEqual(ang, -100)
        self.assertLessEqual(ang, -80)

    regression_notebooks = [
        "tutorial/near_demo_regression_solutions.ipynb",
        "tutorial/near_demo_regression_interface_solutions.ipynb",
    ]

    @parameterized.expand([(path,) for path in regression_notebooks])
    def test_near_demo_regression(self, path):
        """
        Counts the number of modules in the solution. Should be more than 1 as a linear
        layer won't be able to approximate a non-linear function
        """
        result = execute_notebook(
            path,
            suffix="import json; print('*' * 80); "
            + "print(json.dumps([len(module.contained_modules)]))",
        )

        *_, stars, len_module, _ = result.split("\n")
        assert stars == "*" * 80
        n_layers = json.loads(len_module)[0]
        self.assertGreaterEqual(n_layers, 1)

    @parameterized.expand([(path,) for path in regression_notebooks])
    def test_discrete_exercise_skeleton(self, path):
        with open(path) as f:
            solutions = json.load(f)
        with open(path.replace("solutions", "skeleton")) as f:
            notebook = json.load(f)

        self.assertEqual(create_skeleton(solutions, path), notebook)
