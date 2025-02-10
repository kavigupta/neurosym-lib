import unittest

from neurosym.examples.simple_dreamcoder.experiment import (
    compute_learning_curve_for_default_experiment,
    run_all_experiments,
)


class SimpleDreamcoderRegressionTest(unittest.TestCase):
    def test_single_regression(self):
        seed = 0
        results_all = run_all_experiments("outputs/simple_dreamcoder")
        (compression_steps_by_iteration, count), *_ = results_all
        result = results_all[(compression_steps_by_iteration, count)][seed]
        result_current = compute_learning_curve_for_default_experiment(
            compression_steps_by_iteration=compression_steps_by_iteration,
            count=count,
            num_iterations=len(result["timings"]),
            seed=seed,
        )
        self.assertEqual(set(result.keys()), {"timings", "val_errors", "test_errors"})
        self.assertEqual(
            set(result_current.keys()), {"timings", "val_errors", "test_errors"}
        )
        self.assertEqual(len(result["timings"]), len(result_current["timings"]))
        for k in result:
            if k == "timings":
                continue
            self.assertEqual(result[k], result_current[k])
