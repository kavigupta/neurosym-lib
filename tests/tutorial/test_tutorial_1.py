import json
import unittest

import numpy as np

from .utils import execute_notebook


class TestTutorial1(unittest.TestCase):
    def test_tutorial_1(self):
        result = execute_notebook(
            "tutorial/tutorial1.ipynb",
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
