import itertools
import json
import os
import time
from typing import List, Tuple

import numpy as np
import tqdm.auto as tqdm

from neurosym.dsl.dsl import DSL
from neurosym.examples.simple_dreamcoder.algorithm import (
    compute_best_fits_for_each,
    simple_dreamcoder,
)
from neurosym.examples.simple_dreamcoder.domain import example_dataset, example_dsl
from neurosym.utils.logging import log


def compute_learning_curve(
    dsl: DSL,
    xs_train: np.ndarray,
    values_train: np.ndarray,
    xs_test: np.ndarray,
    values_test: np.ndarray,
    *,
    compression_steps_by_iteration: int,
    count: int,
    num_iterations: int,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute the learning curve for the simple dreamcoder algorithm. This will run the algorithm
    for a fixed dataset and DSL, and return the timings, validation errors, and test errors.

    :param dsl: The DSL to use
    :param xs_train: The input sequences for the training set
    :param values_train: The target values for the training set
    :param xs_test: The input sequences for the test set
    :param values_test: The target values for the test set
    :param compression_steps_by_iteration: The number of compression steps to take at each iteration
    :param count: The number of programs to consider
    :param num_iterations: The number of iterations to run
    """
    timings = []
    val_errors = []
    test_errors = []
    for updated_dsl, dist_family, dist, _, error in tqdm.tqdm(
        itertools.islice(
            simple_dreamcoder(
                xs_train,
                values_train,
                dsl,
                compression_steps_by_iteration=compression_steps_by_iteration,
                count=count,
            ),
            num_iterations,
        ),
        total=num_iterations,
    ):
        test_error, _ = compute_best_fits_for_each(
            xs_test, values_test, updated_dsl, dist_family, dist, count=count
        )
        val_errors.append(error)
        test_errors.append(test_error)
        timings.append(time.time())
    return timings, val_errors, test_errors


def compute_learning_curve_for_default_experiment(
    *, compression_steps_by_iteration: int, count: int, seed: int, num_iterations=10
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute the learning curve for the simple dreamcoder algorithm. This will run the algorithm
    for a fixed dataset and DSL, and return the timings, validation errors, and test errors.

    :param compression_steps_by_iteration: The number of compression steps to take at each iteration
    :param count: The number of programs to consider
    :param seed: The random seed to use
    :param num_iterations: The number of iterations to run

    :return: A tuple of ```(timings, val_errors, test_errors)```, where ```timings``` is a list of
        the time at each iteration, ```val_errors``` is a list of the validation errors at each iteration,
        and ```test_errors``` is a list of the test errors at each iteration
    """
    dsl = example_dsl()
    xs_train, values_train = example_dataset(
        num_sequences=1000, len_sequences=100, seed=seed * 2
    )
    xs_test, values_test = example_dataset(
        num_sequences=100, len_sequences=100, seed=seed * 2 + 1
    )

    return compute_learning_curve(
        dsl,
        xs_train,
        values_train,
        xs_test,
        values_test,
        compression_steps_by_iteration=compression_steps_by_iteration,
        count=count,
        num_iterations=num_iterations,
    )


def compute_and_save_learning_curve_for_default_experiment(
    root_path: str,
    *,
    compression_steps_by_iteration: int,
    count: int,
    seed: int,
    num_iterations=10,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute the learning curve for the simple dreamcoder algorithm. This will run the algorithm
    for a fixed dataset and DSL, and return the timings, validation errors, and test errors.

    :param root_path: The root path to save the results
    :param compression_steps_by_iteration: The number of compression steps to take at each iteration
    :param count: The number of programs to consider
    :param seed: The random seed to use
    :param num_iterations: The number of iterations to run
    """
    path = os.path.join(
        root_path,
        f"learning_curve_{num_iterations}_{compression_steps_by_iteration}_{count}_{seed}.json",
    )
    os.makedirs(root_path, exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    log(f"Running experiment for path: {path}")
    timings, val_errors, test_errors = compute_learning_curve_for_default_experiment(
        compression_steps_by_iteration=compression_steps_by_iteration,
        count=count,
        num_iterations=num_iterations,
        seed=seed,
    )
    with open(path, "w") as f:
        json.dump(
            dict(timings=timings, val_errors=val_errors, test_errors=test_errors),
            f,
            indent=2,
        )
    return compute_and_save_learning_curve_for_default_experiment(
        root_path,
        compression_steps_by_iteration=compression_steps_by_iteration,
        count=count,
        num_iterations=num_iterations,
        seed=seed,
    )


def run_all_experiments(root_path: str):
    """
    Run all experiments for the simple dreamcoder algorithm. This will run the algorithm
    for different values of compression_steps_by_iteration and count, and save the results
    in the ``root_path``; if the results already exist, they will be loaded instead.

    :param root_path: The root path to save the results
    """
    results = {}
    for compression_steps_by_iteration in range(1 + 5):
        for count in [500]:
            res_compress_count = []
            for seed in range(10):
                res_compress_count.append(
                    compute_and_save_learning_curve_for_default_experiment(
                        root_path,
                        compression_steps_by_iteration=compression_steps_by_iteration,
                        count=count,
                        seed=seed,
                        num_iterations=100,
                    )
                )
                results[(compression_steps_by_iteration, count)] = res_compress_count
    return results


if __name__ == "__main__":
    run_all_experiments("outputs/simple_dreamcoder")
