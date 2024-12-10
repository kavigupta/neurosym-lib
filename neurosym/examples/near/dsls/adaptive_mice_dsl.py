# pylint: disable=duplicate-code,cyclic-import
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory

from ..operations.aggregation import SymmetricMorletFilter, running_agg_torch
from ..operations.basic import ite_torch
from ..operations.lists import map_torch
from .simple_crim13_dsl import CRIM13_FEATURES, CRIM13_FULL_FEATURE_DIM


def adaptive_crim13_dsl(
    num_classes,
    hidden_dim=None,
):
    """
    A differentiable DSL for finding interpretable programs for mice behavior
    classification on the CRIM13 dataset.
    This DSL contains advanced aggregation functions such as convolution and morlet filters.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl_crim13.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    hidden_dim = num_classes if hidden_dim is None else hidden_dim
    seq_len = (
        13  # CRIM13 dataset has max sequence length of 13. Change for other datasets.
    )

    dslf = DSLFactory(
        input_size=CRIM13_FULL_FEATURE_DIM,
        output_size=num_classes,
        max_overall_depth=6,
        hidden_size=hidden_dim,
    )
    dslf.typedef("fO", "{f, $output_size}")
    dslf.typedef("fH", "{f, $hidden_size}")
    dslf.typedef("fI", "{f, $input_size}")

    for feature_name, feature_indices in CRIM13_FEATURES.items():
        dslf.parameterized(
            f"affine_{feature_name}",
            "() -> $fI -> $fH",
            lambda lin, feature_indices=feature_indices: lambda x: lin(
                x[..., feature_indices]
            ),
            parameters=dict(
                lin=lambda feature_indices=feature_indices: nn.Linear(
                    len(feature_indices), hidden_dim
                )
            ),
        )
        dslf.parameterized(
            f"affine_bool_{feature_name}",
            "() -> $fI -> {f, 1}",
            lambda lin, feature_indices=feature_indices: lambda x: lin(
                x[..., feature_indices]
            ),
            parameters=dict(
                lin=lambda feature_indices=feature_indices: nn.Linear(
                    len(feature_indices), 1
                )
            ),
        )

    dslf.concrete(
        "add",
        "(#a -> #b, #a -> #b) -> #a -> #b",
        lambda f1, f2: lambda x: f1(x) + f2(x),
    )
    dslf.concrete(
        "mul",
        "(#a -> #b, #a -> #b) -> #a -> #b",
        lambda f1, f2: lambda x: f1(x) * f2(x),
    )
    dslf.concrete(
        "running_avg_last5",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 4, lambda t: t),
    )
    dslf.concrete(
        "running_avg_last10",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 9, lambda t: t),
    )

    dslf.concrete(
        "running_avg_window5",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 2, lambda t: t + 2),
    )
    dslf.concrete(
        "running_avg_window11",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 5, lambda t: t + 5),
    )

    dslf.parameterized(
        f"convolve_3_len{seq_len}",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f, conv: lambda x: conv(f(x)).squeeze(),
        parameters=dict(
            conv=lambda: torch.nn.Conv1d(seq_len, 1, 3, padding=1, bias=False)
        ),
    )
    dslf.parameterized(
        f"convolve_5_len{seq_len}",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f, conv: lambda x: conv(f(x)).squeeze(),
        parameters=dict(
            conv=lambda: torch.nn.Conv1d(seq_len, 1, 5, padding=2, bias=False)
        ),
    )

    dslf.parameterized(
        "sym_morlet",
        "([#a] -> [#b]) -> [#a] -> #b",
        lambda f, filter: lambda x: filter(f(x)),
        parameters=dict(filter=lambda: SymmetricMorletFilter()),    # pylint: disable=unnecessary-lambda
    )

    if hidden_dim != num_classes:
        dslf.parameterized(
            "output",
            "(([$fI]) -> [$fH]) -> [$fI] -> $fO",
            lambda f, lin: lambda x: lin(f(x)).softmax(-1),
            dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
        )
    else:
        dslf.concrete(
            "output",
            "(([$fI]) -> [$fH]) -> [$fI] -> $fO",
            lambda f: lambda x: f(x).softmax(-1),
        )
    dslf.concrete(
        "ite",
        "(#a -> {f, 1},  #a -> #b, #a -> #b) -> #a -> #b",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),   # pylint: disable=unnecessary-lambda
    )
    dslf.concrete(
        "map", "(#a -> #b) -> [#a] -> [#b]", lambda f: lambda x: map_torch(f, x)
    )

    dslf.prune_to("[$fI] -> $fO")
    return dslf.finalize()
