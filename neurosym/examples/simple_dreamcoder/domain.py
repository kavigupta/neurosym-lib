import numpy as np

from neurosym.dsl.dsl_factory import DSLFactory


def example_dataset(num_sequences, len_sequences, *, seed, slack=30, stride=5):
    xs = np.linspace(-10, 10, len_sequences * stride)
    values = (
        np.random.RandomState(seed).rand(num_sequences, len_sequences * stride + slack) * 4
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
