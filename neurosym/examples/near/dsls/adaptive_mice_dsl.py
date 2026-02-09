# pylint: disable=duplicate-code,cyclic-import
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.types.type import ListType

from ..operations.aggregation import SymmetricMorletFilter, running_agg_torch
from ..operations.basic import ite_torch
from ..operations.lists import map_prefix_torch, map_torch
from .simple_calms21_dsl import CALMS21_FEATURES, CALMS21_FULL_FEATURE_DIM
from .simple_crim13_dsl import CRIM13_FEATURES, CRIM13_FULL_FEATURE_DIM


def adaptive_calms21_dsl(
    num_classes,
    hidden_dim=None,
):
    """
    A differentiable DSL for finding interpretable programs for mice behavior classification on the CALMS21 dataset.
    This DSL contains advanced aggregation functions such as convolution and morlet filters.
    Consult https://arxiv.org/abs/2104.02710 for more details.
    Consult https://neurosymbolic-learning.github.io/popl23tutorial/neurosymbolic_notebook3.html for a tutorial.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    return adaptive_mice_dsl_builder(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seq_len=100,  # CALMS21 dataset has a maximum sequence length of 100.
        features=CALMS21_FEATURES,
        full_feature_dim=CALMS21_FULL_FEATURE_DIM,
    )


def adaptive_crim13_dsl(
    num_classes,
    hidden_dim=None,
):
    """
    A differentiable DSL for finding interpretable programs for mice behavior classification on the CRIM13 dataset.
    This DSL contains advanced aggregation functions such as convolution and morlet filters.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl_crim13.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    return adaptive_mice_dsl_builder(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seq_len=13,  # CRIM13 dataset has a sequence length of 13.
        features=CRIM13_FEATURES,
        full_feature_dim=CRIM13_FULL_FEATURE_DIM,
    )


def adaptive_mice_dsl_builder(
    num_classes,
    features,
    full_feature_dim,
    seq_len,
    hidden_dim=None,
):
    """
    Builds a differentiable DSL for finding interpretable programs for mice behavior classification.

    :param num_classes: Number of behavior classes.
    :param features: A dictionary of feature names to feature indices.
    :param full_feature_dim: The full feature dimension.
    :param seq_len: The sequence length of the dataset.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    hidden_dim = num_classes if hidden_dim is None else hidden_dim

    dslf = DSLFactory(
        input_size=full_feature_dim,
        output_size=num_classes,
        max_overall_depth=6,
        hidden_size=hidden_dim,
    )
    dslf.typedef("fO", "{f, $output_size}")
    dslf.typedef("fH", "{f, $hidden_size}")
    dslf.typedef("fI", "{f, $input_size}")

    for feature_name, feature_indices in features.items():
        dslf.production(
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
        dslf.production(
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

    dslf.filtered_type_variable("affine_input", lambda x: not isinstance(x, ListType))

    dslf.production(
        "add",
        "(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
        lambda f1, f2: lambda x: f1(x) + f2(x),
    )
    dslf.production(
        "mul",
        "(%affine_input -> #b, %affine_input -> #b) -> %affine_input -> #b",
        lambda f1, f2: lambda x: f1(x) * f2(x),
    )

    dslf.production(
        "running_avg_last5",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 4, lambda t: t),
    )
    dslf.production(
        "running_avg_last10",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 9, lambda t: t),
    )

    dslf.production(
        "running_avg_window5",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 2, lambda t: t + 2),
    )
    dslf.production(
        "running_avg_window11",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 5, lambda t: t + 5),
    )

    dslf.production(
        f"convolve_3_len{seq_len}",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f, conv: lambda x: conv(f(x)).squeeze(),
        parameters=dict(
            conv=lambda: torch.nn.Conv1d(seq_len, 1, 3, padding=1, bias=False)
        ),
    )
    dslf.production(
        f"convolve_5_len{seq_len}",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f, conv: lambda x: conv(f(x)).squeeze(),
        parameters=dict(
            conv=lambda: torch.nn.Conv1d(seq_len, 1, 5, padding=2, bias=False)
        ),
    )

    # pylint: disable=unnecessary-lambda
    dslf.production(
        "sym_morlet",
        "([#a] -> [$fH]) -> [#a] -> $fH",
        lambda f, filter: lambda x: filter(f(x)),
        parameters=dict(filter=lambda: SymmetricMorletFilter()),
    )

    if hidden_dim != num_classes:
        dslf.production(
            "output",
            "(([$fI]) -> [$fH]) -> [$fI] -> [$fO]",
            lambda f, lin: lambda x: lin(f(x)).softmax(-1),
            dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
        )

    dslf.production(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        ite_torch,
    )
    # pylint: enable=unnecessary-lambda
    dslf.production(
        "map",
        "(#a -> #b) -> [#a] -> [#b]",
        lambda f: lambda x: map_torch(f, x),
    )
    dslf.production(
        "map_prefix",
        "([#a] -> #b) -> [#a] -> [#b]",
        lambda f: lambda x: map_prefix_torch(f, x),
    )

    dslf.prune_to("[$fI] -> $fO")
    return dslf.finalize()
