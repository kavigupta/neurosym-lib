# pylint: disable=duplicate-code,unnecessary-lambda
"""Attention-based DSL for ECG classification with ECGDeli features.

Each lambda variable corresponds to one channel (12 leads + 9 globals).
Each channel is a feature vector of size ``features_per_channel``.
[<-----lead1----->]...[<-----lead12----->][<--global feats->]
[<f1><f2>....<f14>]...[<f1><f2>.....<f14>][<g1><g2>.....<g9>]

"""

import numpy as np
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.shield import add_shield_productions
from neurosym.types.type import AtomicType, TensorType


def attention_ecg_dsl(
    num_channels,
    features_per_channel,
    num_classes,
    hidden_dim=16,
    max_overall_depth=6,
    use_shields=False,
    max_type_depth=None,
):
    """Build the attention ECG DSL.

    Each lambda variable corresponds to one channel (12 leads + 9 globals).
    Each channel is a feature vector of size ``features_per_channel``.

    :param num_channels: Number of input channels (leads + globals).
    :param features_per_channel: Feature vector size per channel.
    :param num_classes: Number of output classes.
    :param hidden_dim: Hidden dimension for affine projections.
    :param max_overall_depth: Maximum depth for DSL expansion.
    :param max_type_depth: Maximum depth for type expansion.
    """
    dslf = DSLFactory(
        I=features_per_channel,
        O=num_classes,
        H=hidden_dim,
        max_overall_depth=max_overall_depth,
        max_env_depth=num_channels,
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fHid", "{f, $H}")

    dslf.filtered_type_variable(
        "num",
        lambda x: isinstance(x, AtomicType)
        and x.name == "f"
        or isinstance(x, TensorType),
    )

    dslf.production("add", "(%num, %num) -> %num", lambda x, y: x + y)
    dslf.production("mul", "(%num, %num) -> %num", lambda x, y: x * y)
    dslf.production("sub", "(%num, %num) -> %num", lambda x, y: x - y)

    dslf.production(
        "ite",
        "({f, 1}, %num, %num) -> %num",
        lambda cond, x, y: torch.sigmoid(cond) * x + (1 - torch.sigmoid(cond)) * y,
    )

    dslf.production(
        "embed",
        "$fInp -> $fHid",
        lambda x, lin: lin(x),
        dict(lin=lambda: nn.Linear(features_per_channel, hidden_dim)),
    )

    dslf.production(
        "linear",
        "$fHid -> $fHid",
        lambda x, lin: lin(x),
        dict(lin=lambda: nn.Linear(hidden_dim, hidden_dim)),
    )
    dslf.production(
        "linear_bool",
        "$fHid -> {f, 1}",
        lambda x, lin: lin(x),
        dict(lin=lambda: nn.Linear(hidden_dim, 1)),
    )
    dslf.production(
        "output",
        "$fHid -> $fOut",
        lambda x, lin: lin(x),
        dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
    )
    dslf.lambdas(
        max_type_depth=(
            max_type_depth
            if max_type_depth is not None
            else np.log2(num_channels + 2) + 0.05
        )
    )
    if use_shields:
        add_shield_productions(dslf)
    dslf.prune_to(f"({'$fInp, ' * num_channels}) -> $fOut")
    return dslf.finalize()
