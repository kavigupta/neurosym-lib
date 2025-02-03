import itertools
import numpy as np

from neurosym.compression.process_abstraction import multi_step_compression
from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.program_dist.bigram import BigramProgramDistributionFamily
from neurosym.programs.s_expression_render import render_s_expression


def example_dataset(num_sequences, len_sequences, *, slack=30, stride=5):
    xs = np.linspace(-10, 10, len_sequences * stride)
    values = (
        np.random.RandomState(1).rand(num_sequences, len_sequences * stride + slack) * 4
        - 2
    )
    values = np.mean(
        [values[:, i : i + len_sequences * stride] for i in range(slack)], axis=0
    )
    values *= np.sqrt(slack)
    values = values[:, ::stride]
    xs = xs[::stride]
    return xs, values


def example_dsl():
    dslf = DSLFactory()
    dslf.concrete("0", "() -> f", lambda: 0)
    dslf.concrete("1", "() -> f", lambda: 1)
    dslf.concrete("2", "() -> f", lambda: 2)
    dslf.concrete("+", "(f, f) -> f", lambda x, y: x + y)
    dslf.concrete("-", "(f, f) -> f", lambda x, y: x - y)
    # BEGIN SOLUTION "YOUR CODE HERE"
    dslf.concrete("*", "(f, f) -> f", lambda x, y: x * y)
    dslf.concrete("**", "(f, f) -> f", lambda x, y: x**y)
    dslf.concrete("/", "(f, f) -> f", lambda x, y: x / y)
    dslf.concrete("sin", "f -> f", np.sin)
    dslf.concrete("sqrt", "f -> f", np.sqrt)
    dslf.concrete("<", "(f, f) -> b", lambda x, y: x < y)
    dslf.concrete("ite", "(b, f, f) -> f", lambda cond, x, y: x if cond else y)
    # END SOLUTION
    dslf.lambdas()
    dslf.prune_to("f -> f")
    return dslf.finalize()


# TODO update the code in the discrete_exercise_solutioipynb to use these
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
            if y is None or not (-2 <= y <= 2):
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
    print("Mean error: ", errors.min(1).mean())
    return [filtered_programs[i] for i in program_idxs]


def full_algorithm(xs, values, dsl):
    dist_family = BigramProgramDistributionFamily(dsl)
    dist = dist_family.uniform()
    while True:
        best_programs = best_fits(xs, values, dsl, dist_family, dist, count=5000)
        abstraction_dsl, rewritten = multi_step_compression(dsl, best_programs, 5)
        abstraction_family = BigramProgramDistributionFamily(abstraction_dsl)
        abstraction_dist = abstraction_family.fit_distribution(
            rewritten
        ).bound_minimum_likelihood(0.01)
        dsl, dist_family, dist = abstraction_dsl, abstraction_family, abstraction_dist
        yield dsl, best_programs
