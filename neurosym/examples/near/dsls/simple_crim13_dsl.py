# pylint: disable=duplicate-code,cyclic-import
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory

from ..operations.aggregation import running_agg_torch
from ..operations.basic import ite_torch

CRIM13_FEATURES = {
    "res_angle_head_body": torch.LongTensor(list(range(0, 2))),
    "axis_ratio": torch.LongTensor(list(range(2, 4))),
    "speed": torch.LongTensor(list(range(4, 6))),
    "acceleration": torch.LongTensor(list(range(6, 8))),
    "tangential_velocity": torch.LongTensor(list(range(8, 10))),
    "rel_angle_social": torch.LongTensor(list(range(10, 12))),
    "angle_between": torch.LongTensor(list(range(12, 13))),
    "facing_angle": torch.LongTensor(list(range(13, 15))),
    "overlap_bboxes": torch.LongTensor(list(range(15, 16))),
    "area_ellipse_ratio": torch.LongTensor(list(range(16, 17))),
    "min_res_nose_dist": torch.LongTensor(list(range(17, 18))),
}

CRIM13_FULL_FEATURE_DIM = 18


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
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 4, lambda t: t),
    )
    dslf.concrete(
        "running_avg_last10",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 9, lambda t: t),
    )

    dslf.concrete(
        "running_avg_window5",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 2, lambda t: t + 2),
    )
    dslf.concrete(
        "running_avg_window11",
        "(#a -> $fH) -> [#a] -> $fH",
        lambda f: lambda x: running_agg_torch(x, f, lambda t: t - 5, lambda t: t + 5),
    )

    if hidden_dim != num_classes:
        dslf.parameterized(
            "output",
            "(([$fI]) -> $fH) -> [$fI] -> $fO",
            lambda f, lin: lambda x: lin(f(x)).softmax(-1),
            dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
        )
    else:
        dslf.concrete(
            "output",
            "(([$fI]) -> $fH) -> [$fI] -> $fO",
            lambda f: lambda x: f(x).softmax(-1),
        )
    # pylint: disable=unnecessary-lambda
    dslf.concrete(
        "ite",
        "(#a -> {f, 1},  #a -> #b, #a -> #b) -> #a -> #b",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
    )
    # pylint: enable=unnecessary-lambda

    dslf.prune_to("([$fI]) -> $fO")
    return dslf.finalize()
