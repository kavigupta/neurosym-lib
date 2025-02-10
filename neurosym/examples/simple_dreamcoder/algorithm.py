import itertools

import numpy as np

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
