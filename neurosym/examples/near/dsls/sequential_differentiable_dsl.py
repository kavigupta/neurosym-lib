import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory

from ..operations.basic import ite_torch
from ..operations.lists import fold_torch, map_torch


def example_rnn_dsl(input_size, output_size):
    """
    An example DSL for RNN-like sequence-to-sequence tasks.

    :param input_size: Size of the input vectors
    :param output_size: Size of the output vectors
    """
    dslf = DSLFactory(L=input_size, O=output_size, max_overall_depth=5)
    dslf.typedef("fL", "{f, $L}")

    dslf.production("add", "() -> ($fL, $fL) -> $fL", lambda: lambda x, y: x + y)
    dslf.production("mul", "() -> ($fL, $fL) -> $fL", lambda: lambda x, y: x * y)
    dslf.production(
        "fold", "((#a, #a) -> #a) -> [#a] -> #a", lambda f: lambda x: fold_torch(f, x)
    )
    dslf.production(
        "sum", "() -> $fL -> f", lambda: lambda x: torch.sum(x, dim=-1).unsqueeze(-1)
    )
    dslf.production(
        "linear",
        "() -> $fL -> $fL",
        lambda lin: lin,
        dict(lin=lambda: nn.Linear(input_size, input_size)),
    )
    dslf.production(
        "output",
        "(([$fL]) -> [$fL]) -> [$fL] -> [{f, $O}]",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(input_size, output_size)),
    )
    dslf.production(
        "ite",
        "(#a -> f, #a -> #a, #a -> #a) -> #a -> #a",
        ite_torch,
    )
    dslf.production(
        "map", "(#a -> #b) -> [#a] -> [#b]", lambda f: lambda x: map_torch(f, x)
    )

    dslf.production(
        "compose", "(#a -> #b, #b -> #c) -> #a -> #c", lambda f, g: lambda x: g(f(x))
    )

    dslf.prune_to("[{f, $L}] -> [{f, $O}]")
    return dslf.finalize()
