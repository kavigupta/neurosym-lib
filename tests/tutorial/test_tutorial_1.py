import json
import unittest

import numpy as np

from .utils import execute_notebook


class TestTutorial1(unittest.TestCase):
    def test_tutorial_1_classification(self):
        result = execute_notebook(
            "tutorial/tutorial1_classification.ipynb",
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

    def test_tutorial_1_regression(self):
        """
        Counts the number of modules in the solution. Should be more than 1 as a linear
        layer won't be able to approximate a non-linear function
        """
        result = execute_notebook(
            "tutorial/tutorial1_regression.ipynb",
            suffix="import json; print('*' * 80); "
            + "print(json.dumps([len(module.contained_modules)]))",
        )

        *_, stars, len_module, _ = result.split("\n")
        assert stars == "*" * 80
        n_layers = json.loads(len_module)[0]
        self.assertGreaterEqual(n_layers, 1)
