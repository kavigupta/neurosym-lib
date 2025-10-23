# pylint: disable=duplicate-code,cyclic-import
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory

from ..operations.aggregation import running_agg_torch
from ..operations.basic import ite_torch
from ..operations.lists import map_prefix_torch, map_torch
from neurosym.types.type import ListType

CRIM13_FEATURES = {
    "position": torch.LongTensor([0, 1, 2, 3]),
    "distance": torch.LongTensor([4]),
    "distance_change": torch.LongTensor([5]),
    "angle": torch.LongTensor([6, 7, 10]),
    "angle_change": torch.LongTensor([8, 9]),
    "velocity": torch.LongTensor([11, 12, 13, 14]),
    "acceleration": torch.LongTensor([15, 16, 17, 18]),
}

CRIM13_FULL_FEATURE_DIM = 19


def simple_crim13_dsl(num_classes, hidden_dim=None):
    """
    A differentiable DSL for finding interpretable programs for mice behavior
    classification on the CRIM13 dataset.
    Consult https://arxiv.org/abs/2007.12101 for more details.
    Consult https://github.com/trishullab/near/blob/master/near_code/dsl_crim13.py for the reference implementation.

    :param num_classes: Number of behavior classes.
    :param hidden_dim: Size of hidden dimension (if None, set to num_classes).
    """
    hidden_dim = num_classes if hidden_dim is None else hidden_dim

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

    dslf.filtered_type_variable(
        "affine_input", lambda x: not isinstance(x, ListType)
    )

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
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 4, lambda t: t),
    )
    dslf.production(
        "running_avg_last10",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 9, lambda t: t),
    )

    dslf.production(
        "running_avg_window5",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 2, lambda t: t + 2),
    )
    dslf.production(
        "running_avg_window11",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 5, lambda t: t + 5),
    )

    if hidden_dim != num_classes:
        dslf.production(
            "output",
            "(([$fI]) -> [$fH]) -> [$fI] -> [$fO]",
            lambda f, lin: lambda x: lin(f(x)).softmax(-1),
            dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
        )
    # pylint: disable=unnecessary-lambda
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

    dslf.prune_to("([$fI]) -> [$fO]")
    return dslf.finalize()
