import itertools

import numpy as np

from neurosym.compression.process_abstraction import multi_step_compression
from neurosym.program_dist.bigram import BigramProgramDistributionFamily
from neurosym.programs.s_expression_render import render_s_expression


def run_safely(f, x):
    # pylint: disable=broad-except
    try:
        y = f(x)
    except Exception:
        return None
    if np.isnan(y):
        return None
    return y


def evaluate_all_programs(xs, dsl, programs):
    # pylint: disable=broad-except
    filtered_programs, evaluations = [], []
    for prog in programs:
        try:
            actual_fn = dsl.compute(dsl.initialize(prog))
        except Exception:
            continue
        ys = []
        for inp in xs:
            y = run_safely(actual_fn, inp)
            if y is None or not -2 <= y <= 2:
                break
            ys.append(y)
        else:
            filtered_programs.append(prog)
            evaluations.append(ys)
    return filtered_programs, np.array(evaluations)


def best_fits(xs, values, dsl, family, dist, *, count=5000):
    programs = [prog for prog, _ in itertools.islice(family.enumerate(dist), count)]
    programs = sorted(programs, key=lambda x: len(render_s_expression(x)))
    filtered_programs, ys = evaluate_all_programs(xs, dsl, programs)
    errors = ((ys[None] - values[:, None]) ** 2).sum(-1)
    program_idxs = errors.argmin(1)
    return errors.min(1).mean(), [filtered_programs[i] for i in program_idxs]


def iterate_algorithm(
    xs, values, dsl, val_split=0.1, compression_steps_by_iteration=1, count=5000
):
    num_train = int(len(xs) * (1 - val_split))
    dist_family = BigramProgramDistributionFamily(dsl)
    dist = dist_family.uniform()
    while True:
        _, best_programs = best_fits(xs, values, dsl, dist_family, dist, count=count)
        error = (
            (
                (
                    evaluate_all_programs(xs, dsl, best_programs[num_train:])[1]
                    - values[num_train:]
                )
                ** 2
            )
            .sum(-1)
            .mean()
        )
        yield dsl, dist_family, dist, best_programs, error
        dsl, rewritten = multi_step_compression(
            dsl, best_programs[:num_train], compression_steps_by_iteration
        )
        dist_family = BigramProgramDistributionFamily(dsl)
        dist = dist_family.fit_distribution(rewritten).bound_minimum_likelihood(0.01)
