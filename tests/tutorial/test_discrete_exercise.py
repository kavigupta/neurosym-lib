import json
import unittest

from tests.tutorial.utils import execute_notebook
from tutorial.process_tutorial import create_skeleton


class TestDiscreteExercise(unittest.TestCase):
    def test_discrete_exercise_skeleton(self):
        with open("tutorial/discrete_exercise_solutions.ipynb") as f:
            solutions = json.load(f)
        with open("tutorial/discrete_exercise_skeleton.ipynb") as f:
            notebook = json.load(f)

        self.assertEqual(
            create_skeleton(solutions, "tutorial/discrete_exercise_solutions.ipynb"),
            notebook,
        )

    def test_discrete_exercise_skeleton_runnable(self):
        result = execute_notebook(
            "tutorial/discrete_exercise_solutions.ipynb",
            suffix="import json; print('*' * 80); print(json.dumps(abstraction_dsl.render()))",
        )
        *_, stars, actual, _ = result.split("\n")
        self.assertEqual(stars, "*" * 80)
        actual = json.loads(actual)
        expected = """
              0 :: () -> f
              1 :: () -> f
              2 :: () -> f
              + :: (f, f) -> f
              - :: (f, f) -> f
              * :: (f, f) -> f
             ** :: (f, f) -> f
              / :: (f, f) -> f
            sin :: f -> f
           sqrt :: f -> f
              < :: (f, f) -> b
            ite :: (b, f, f) -> f
            lam :: L<#body|#__lam_0> -> #__lam_0 -> #body
           $0 :: V<$0>
           $1 :: V<$1>
           $2 :: V<$2>
           $3 :: V<$3>
           __10 :: (f, f) -> f = (lam-abstr (#0 #1) (sin (/ #1 #0)))
           __20 :: (f, f) -> f = (lam-abstr (#0 #1) (ite (< #1 #0) (1) (0)))
           __30 :: () -> f = (lam-abstr () (+ (1) (2)))
           __40 :: f -> f = (lam-abstr (#0) (__10 (2) #0))
           __50 :: () -> f = (lam-abstr () (sqrt (2)))
        """
        actual_lines, expected_lines = [
            [line.strip() for line in text.split("\n") if line.strip()]
            for text in (actual, expected)
        ]
        # Base productions are ordered; abstractions may vary in order
        # due to compression being sensitive to the enumeration set.
        actual_base = [l for l in actual_lines if "lam-abstr" not in l]
        expected_base = [l for l in expected_lines if "lam-abstr" not in l]
        actual_abstr = {l.split(" :: ", 1)[1] for l in actual_lines if "lam-abstr" in l}
        expected_abstr = {
            l.split(" :: ", 1)[1] for l in expected_lines if "lam-abstr" in l
        }
        self.assertEqual(actual_base, expected_base)
        self.assertEqual(actual_abstr, expected_abstr)
