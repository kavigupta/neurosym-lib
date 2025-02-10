import itertools
from typing import Callable, List

import numpy as np

from neurosym.dsl.dsl import DSL
from neurosym.program_dist.distribution import (
    ProgramDistribution,
    ProgramDistributionFamily,
)
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression


def run_safely(f: Callable[[float], float], x: float):
    """
    Run a numeric function safely, returning None if it throws an exception or returns NaN.

    :param f: The function to run
    :param x: The input to the function

    :return: The output of the function, or None if it throws an exception or returns NaN
    """
    # pylint: disable=broad-except
    try:
        y = f(x)
    except Exception:
        return None
    if np.isnan(y):
        return None
    return y


def evaluate_all_programs(xs: np.ndarray, dsl: DSL, programs: List[SExpression]):
    """
    Evaluate the programs in the list on the sequence, returning only the ones that
    return valid values on all inputs.

    :param xs: The input sequence, a numpy array of shape (N,)
    :param dsl: The DSL to use
    :param programs: The programs to evaluate

    :return: A tuple of ```(filtered_programs, evaluations)```, where ```filtered_programs``` is a list of programs
        that returned valid values on all inputs, and ```evaluations``` is a matrix of shape ```(len(filtered_programs), N)```
    """
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


def compute_best_fits_for_each(
    xs: np.ndarray,
    values: np.ndarray,
    dsl: DSL,
    family: ProgramDistributionFamily,
    dist: ProgramDistribution,
    *,
    count: int = 5000
):
    """
    Compute the best fits for each sequence in the dataset. This is done by ensuring
    that the program is valid on all inputs, then computing the mean squared error
    between the program's output and each target value, computing the best program
    for each sequence.

    :param xs: The input sequences, a numpy array of shape ```(N,)```
    :param values: The target values, a numpy array of shape ```(num_sequences, N)```
    :param dsl: The DSL to use
    :param family: The family to use
    :param dist: The distribution to use
    :param count: The number of programs to consider

    :return: A tuple of ```(mean_error, best_programs)```, where ```mean_error``` is the mean squared error
        across all sequences, and ```best_programs``` is a list of the best program for each sequence
    """
    programs = [prog for prog, _ in itertools.islice(family.enumerate(dist), count)]
    programs = sorted(programs, key=lambda x: len(render_s_expression(x)))
    filtered_programs, ys = evaluate_all_programs(xs, dsl, programs)
    errors = ((ys[None] - values[:, None]) ** 2).sum(-1)
    program_idxs = errors.argmin(1)
    return errors.min(1).mean(), [filtered_programs[i] for i in program_idxs]
