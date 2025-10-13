import numpy as np

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.dsl.dsl_factory import DSLFactory


def add_variables_domain_datamodule():
    rng = np.random.default_rng(0)
    fn = lambda x: x[:, [1]] + x[:, [2]] + x[:, [3]]
    x_train = rng.standard_normal((1000, 10)).astype(np.float32)
    x_test = rng.standard_normal((200, 10)).astype(np.float32)

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


def basic_drop_dsl(amount, is_vectorized=False, include_drops=True):
    dslf = DSLFactory(max_env_depth=amount + 1)

    dslf.typedef("f1", "{f, 1}")

    main_output = f"({', '.join(['$f1'] * amount)}) -> $f1"
    vec_output = f"{{f, {amount}}} -> $f1"

    dslf.production("+", "($f1, $f1) -> $f1", lambda x, y: x + y)
    dslf.lambdas(
        include_drops=include_drops,
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
