import numpy as np

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.dsl.dsl_factory import DSLFactory


def add_variables_domain_datamodule(count, indices):
    """
    Datamodule for the "add variables" domain with shield functions. The task is to
    sum a subset of the input variables.

    :param count: The total number of input variables.
    :param indices: The indices of the variables to sum.
    """
    rng = np.random.default_rng(0)
    fn = lambda x: sum(x[:, [i]] for i in indices)
    x_train = rng.standard_normal((1000, count)).astype(np.float32)
    x_test = rng.standard_normal((200, count)).astype(np.float32)

    datamodule = DatasetWrapper(
        DatasetFromNpy(
            x_train,
            fn(x_train),
            0,
        ),
        DatasetFromNpy(
            x_test,
            fn(x_test),
            None,
        ),
    )
    return datamodule


def add_variables_domain_dsl(amount, is_vectorized=False, include_shield=True):
    """
    A basic DSL for the "add variables" domain with shield functions. The DSL includes a
    function to add two numbers, lambda abstractions, and optionally shield functions.

    :param amount: The number of input variables.
    :param is_vectorized: Whether to include a vectorized dispatch function.
    :param include_shield: Whether to include shield functions in the DSL.
    """
    dslf = DSLFactory(max_env_depth=amount + 1)

    main_output = f"({', '.join(['f'] * amount)}) -> f"
    vec_output = f"{{f, {amount}}} -> {{f, 1}}"

    dslf.production("+", "(f, f) -> f", lambda x, y: x + y)
    dslf.lambdas(
        include_shield=include_shield,
        max_type_depth=np.log2(amount) + 1,
        require_arities=[amount],
    )
    if is_vectorized:
        dslf.production(
            "dispatch",
            f"({main_output}) -> ({vec_output})",
            lambda f: lambda x: f(*(x[:, [i]] for i in range(x.shape[1]))),
        )
        dslf.prune_to(vec_output)
    else:
        dslf.prune_to(main_output)

    dsl = dslf.finalize()

    return dsl
