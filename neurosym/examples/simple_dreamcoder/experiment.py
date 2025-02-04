import itertools
import json
import os
import time

import tqdm.auto as tqdm
from neurosym.examples.simple_dreamcoder.algorithm import (
    best_fits,
    evaluate_all_programs,
    iterate_algorithm,
)
from neurosym.examples.simple_dreamcoder.domain import example_dataset, example_dsl


def compute_learning_curve(
    dsl,
    xs_train,
    values_train,
    xs_test,
    values_test,
    *,
    compression_steps_by_iteration,
    count,
    num_iterations,
):
    timings = []
    val_errors = []
    test_errors = []
    for updated_dsl, dist_family, dist, _, error in tqdm.tqdm(
        itertools.islice(
            iterate_algorithm(
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
        best_programs_test = best_fits(
            xs_test, values_test, updated_dsl, dist_family, dist, count=count
        )
        test_error = (
            (
                (
                    evaluate_all_programs(xs_test, updated_dsl, best_programs_test)[1]
                    - values_test
                )
                ** 2
            )
            .sum(-1)
            .mean()
        )
        val_errors.append(error)
        test_errors.append(test_error)
        timings.append(time.time())
    return timings, val_errors, test_errors


def compute_learning_curve_for_default_experiment(
    *, compression_steps_by_iteration, count, seed, num_iterations=10
):
    dsl = example_dsl()
    xs_train, values_train = example_dataset(
        num_sequences=1000, len_sequences=20, seed=seed * 2
    )
    xs_test, values_test = example_dataset(
        num_sequences=100, len_sequences=20, seed=seed * 2 + 1
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
    *, compression_steps_by_iteration, count, seed, num_iterations=10
):
    root_path = "outputs/simple_dreamcoder"
    path = os.path.join(
        root_path,
        f"learning_curve_{num_iterations}_{compression_steps_by_iteration}_{count}_{seed}.json",
    )
    os.makedirs(root_path, exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
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
        compression_steps_by_iteration=compression_steps_by_iteration,
        count=count,
        num_iterations=num_iterations,
        seed=seed,
    )


def learning_curves_all():
    results = {}
    for compression_steps_by_iteration in [0, 1, 5]:
        for count in [5000]:
            res_compress_count = []
            for seed in range(3):
                res_compress_count.append(
                    compute_and_save_learning_curve_for_default_experiment(
                        compression_steps_by_iteration=compression_steps_by_iteration,
                        count=count,
                        seed=seed,
                        num_iterations=25,
                    )
                )
                results[(compression_steps_by_iteration, count)] = res_compress_count
    return results

if __name__ == "__main__":
    learning_curves_all()