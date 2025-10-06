import numpy as np
import torch
from torch import nn

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.interface import NEAR
from neurosym.search.bounded_astar import bounded_astar
from neurosym.types.type_string_repr import TypeDefiner


def get_combinator_dsl(nesting):
    """
    A simple DSL that exists for debugging whether the heuristic NEAR search works.

    The idea is that you have a DSL where the only way to create a correct program is
    (correct_5 (correct_4 (correct_3 (correct_2 (correct_1 (terminal)))))) and if you
    instead do (wrong_5 (correct_4 (correct_3 (correct_2 (correct_1 (terminal))))))
    or replace any of the correct_i with wrong_i, the program will not work.

    To accomplish this, we need to ensure that (1) nesting is actually enforced and (2)
    that the wrong_i functions make it so that no matter what happens below them, the
    program will not work.

    To accomplish (1), we need to make sure that each level of nesting has a different
    tensor size. So, terminal will return a tensor of size 1, correct_1 will return a
    tensor of size 2, correct_2 will return a tensor of size 3, and so on.

    To accomplish (2), we set our target function to just be the identity function (but
    repeating the last element of the tensor). For the wrong_i functions, we will just
    return a tensor of all zeros.
    """
    dslf = DSLFactory()
    dslf.production(
        "terminal",
        "() -> {f, 1} -> {f, 2}",
        lambda lin: lin,
        dict(lin=lambda: nn.Linear(1, 2)),
    )

    for i in range(2, nesting + 1):
        typ = "({f, 1} -> {f, %s}) -> {f, 1} -> {f, %s}" % (i, i + 1)
        dslf.production(
            f"correct_{i}",
            typ,
            lambda f: lambda x: _expand_last_axis(f(x)),
        )
        dslf.production(
            f"wrong_{i}",
            typ,
            lambda f: lambda x: _expand_last_axis(f(x)) * 0,
        )
    dslf.prune_to("{f, 1} -> {f, %s}" % (nesting + 1))
    return dslf.finalize()


def get_variable_dsl(nesting):
    """
    Like get_combinator_dsl, but using lambdas and variables instead of combinators.

    Here, `terminal` is a function that goes {f, 1} -> {f, 2}, and each correct_i
    function is a function that goes {f, 1} -> {f, i + 1}. We have lambdas that create
    a variable
    """
    dslf = DSLFactory()
    dslf.production(
        "terminal",
        "{f, 1} -> {f, 2}",
        lambda x, *, lin: lin(x),
        dict(lin=lambda: nn.Linear(1, 2)),
    )

    for i in range(2, nesting + 1):
        typ = "{f, %s} -> {f, %s}" % (i, i + 1)
        dslf.production(f"correct_{i}", typ, _expand_last_axis)
        dslf.production(f"wrong_{i}", typ, lambda x: _expand_last_axis(x * 0))

    dslf.lambdas(max_arity=1)
    dslf.prune_to("{f, 1} -> {f, %s}" % (nesting + 1))
    return dslf.finalize()


def get_dataset(nesting, *, scale=100, train_size=1000, test_size=50):
    """
    Data for the debug_nested_dsl DSL.

    :param nesting: The nesting level of the DSL.
    :param scale: The scale of the data. By default fairly large, to make it easier to
        see if the search is working.
    :param train_size: The size of the training data.
    :param test_size: The size of the testing data.
    """
    rng = np.random.RandomState(0)
    xtrain, xtest = [
        scale * rng.randn(size, 1).astype(np.float32)
        for size in (train_size, test_size)
    ]
    ytrain, ytest = [np.repeat(x, nesting + 1, 1) for x in (xtrain, xtest)]
    return DatasetWrapper(
        DatasetFromNpy(xtrain, ytrain, None),
        DatasetFromNpy(xtest, ytest, None),
        batch_size=100,
    )


def run_near_on_dsl(nesting, dsl, neural_hole_filler, max_iterations=None):
    """
    Run NEAR on the given DSL, with the given neural modules.

    :param nesting: The nesting level of the DSL.
    :param dsl: The DSL to use.
    :param neural_hole_filler: The neural modules to use.
    :param max_iterations: The maximum number of iterations to run for.
    """
    interface = NEAR(
        max_depth=nesting * 10000,
        # lr hilariously high and n_epochs hilariously low but it's fine
        # for what we're investigating since the wrong_i simply do not work
        lr=0.01,
        n_epochs=50,
        accelerator="cpu",
    )
    interface.register_search_params(
        dsl=dsl,
        type_env=TypeDefiner(),
        neural_hole_filler=neural_hole_filler,
        search_strategy=bounded_astar,
        loss_callback=torch.nn.functional.mse_loss,
        validation_params=dict(progress_by_epoch=False),
    )
    return interface.fit(
        datamodule=get_dataset(nesting),
        program_signature="{f, 1} -> {f, %s}" % (nesting + 1),
        n_programs=1,
        validation_max_epochs=100,
        max_iterations=nesting * 4 if max_iterations is None else max_iterations,
    )


def _expand_last_axis(x):
    return torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
