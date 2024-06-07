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
            lam :: L<#body|f> -> f -> #body
           $0_0 :: V<f@0>
           __10 :: (f, f) -> f = (lam-abstr (#0 #1) (sin (/ #1 #0)))
           __20 :: () -> f = (lam-abstr () (sqrt (2)))
           __30 :: (f, f) -> f -> f = (lam-abstr (#0 #1) (lam (- #1 (sin (** #0 ($0_0))))))
           __40 :: (f, f) -> f = (lam-abstr (#0 #1) (ite (< #1 #0) (1) (0)))
           __50 :: () -> f = (lam-abstr () (sin (1)))
        """
        actual_lines, expected_lines = [
            [line.strip() for line in text.split("\n") if line.strip()]
            for text in (actual, expected)
        ]
        self.assertEqual(actual_lines, expected_lines)
