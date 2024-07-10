import json
import unittest

from tutorial.process_tutorial import create_skeleton

from .utils import execute_notebook

validate = """
from permacache import stable_hash
def validate(t):
    vy = t[:, -1]
    vx = t[:, -2]
    [bounce_loc] = np.where((vy[1:] > 0) & (vy[:-1] < 0))
    if not bounce_loc.size:
        return "no ground bounces"
    if not (np.abs(vy[bounce_loc]) / np.abs(vy).max() > 0.25).mean() > 0.5:
        return "ground bounces not at max velocity"
    if not (np.sign(vx[bounce_loc]) == np.sign(vx[bounce_loc + 1])).mean() > 0.5:
        return "ground bounces don't mantain vx"
    return "success"

print("*" * 80)
print([validate(trajectory) for trajectory in trajectories], stable_hash(trajectories))
"""


class TestBouncingBallExercise(unittest.TestCase):
    def test_bouncing_ball_exercise_output(self):
        result = execute_notebook(
            "tutorial/bouncing_ball_exercise_solutions.ipynb",
            suffix=validate,
            cwd="tutorial",
        )
        *_, stars, validation, _ = result.split("\n")
        self.assertEqual(stars, "*" * 80)
        self.assertIn("success", validation)
        # determinism. For some reason this only works locally, so I'm commenting it out
        # self.assertIn(
        #     "508766982a85233fef5b04ee296e7cb64db494232901e4c5a8516b0482392e79",
        #     validation,
        # )

    def test_bouncing_ball_exercise_skeleton(self):
        self.maxDiff = None
        with open("tutorial/bouncing_ball_exercise_solutions.ipynb") as f:
            solutions = json.load(f)
        with open("tutorial/bouncing_ball_exercise_skeleton.ipynb") as f:
            notebook = json.load(f)

        self.assertEqual(
            create_skeleton(
                solutions, "tutorial/bouncing_ball_exercise_solutions.ipynb"
            ),
            notebook,
        )
