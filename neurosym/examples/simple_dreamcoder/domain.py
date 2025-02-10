import numpy as np


def example_dataset(num_sequences, len_sequences, *, seed, slack=30, stride=5):
    xs = np.linspace(-10, 10, len_sequences * stride)
    values = (
        np.random.RandomState(seed).rand(num_sequences, len_sequences * stride + slack)
        * 4
        - 2
    )
    values = np.mean(
        [values[:, i : i + len_sequences * stride] for i in range(slack)], axis=0
    )
    values *= np.sqrt(slack)
    values = values[:, ::stride]
    xs = xs[::stride]
    return xs, values
