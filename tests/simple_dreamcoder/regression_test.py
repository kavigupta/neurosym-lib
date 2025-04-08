import unittest

from parameterized import parameterized

from neurosym.examples import simple_dreamcoder


class SimpleDreamcoderRegressionTest(unittest.TestCase):
    @parameterized.expand(
        [
            (num_iterations, compression_steps_by_iteration)
            for num_iterations in [10, 20]
            for compression_steps_by_iteration in [1, 5]
        ]
    )
    def test_single_regression(self, num_iterations, compression_steps_by_iteration):
        kwargs = dict(
            compression_steps_by_iteration=compression_steps_by_iteration,
            count=100,
            num_iterations=num_iterations,
            seed=0,
        )
        result_s = (
            simple_dreamcoder.compute_and_save_learning_curve_for_default_experiment(
                "outputs/simple_dreamcoder_test", **kwargs
            )
        )
        timings_c, val_errors_c, test_errors_c = (
            simple_dreamcoder.compute_learning_curve_for_default_experiment(**kwargs)
        )
        print(result_s)
        self.assertEqual(result_s["val_errors"], val_errors_c)
        self.assertEqual(result_s["test_errors"], test_errors_c)
        self.assertEqual(len(result_s["timings"]), len(timings_c))
