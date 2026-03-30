# pylint: disable=duplicate-code,unnecessary-lambda
"""Attention-based DSL for ECG classification with ECGDeli features.

Each lambda variable corresponds to one channel (12 leads + 9 globals).
Each channel is a feature vector of size ``features_per_channel``.
[<-----lead1----->]...[<-----lead12----->][<--global feats->]
[<f1><f2>....<f14>]...[<f1><f2>.....<f14>][<g1><g2>.....<g9>]

"""

import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.cost import ProgramEmbedding
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.types.type import ArrowType, AtomicType
from neurosym.examples.shield import add_shield_productions

class SoftChannelMask(nn.Module):
    """Learnable soft channel mask via sigmoid-activated logits.

    Produces a differentiable (0, 1) mask over leads that the
    AffineChannelAttention module can use as attention priors.
    """

    def __init__(self, num_channels: int = 12):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, environment=()):
        del environment
        return torch.sigmoid(self.logits).expand(*x.shape[:-1], -1)


class ChannelHoleFiller(NeuralHoleFiller):
    """Fills holes that produce channel-typed outputs with a SoftChannelMask."""

    def __init__(self, num_channels: int = 12):
        self.num_channels = num_channels

    def initialize_module(self, type_with_environment):
        typ = type_with_environment.typ

        def is_channel(t):
            return isinstance(t, AtomicType) and t.name == "channel"

        if isinstance(typ, ArrowType):
            out = typ.output_type
            if isinstance(out, ArrowType):
                out = out.output_type
            if is_channel(out):
                return SoftChannelMask(self.num_channels)
        if is_channel(typ):
            return SoftChannelMask(self.num_channels)
        return None


class _ChannelUnpackModule(nn.Module):
    """Wraps a TorchProgramModule to unpack (B, N, I) input into N separate (B, I) args."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, environment=()):
        return self.inner(*x.unbind(dim=1), environment=environment)


class ChannelUnpackEmbedding(ProgramEmbedding):
    """ProgramEmbedding that unpacks channelised input for multi-variable lambdas.

    The NEAR training loop passes a single tensor ``x`` of shape ``(B, N, I)``.
    This embedding wraps the program module so that ``x`` is unpacked into
    ``N`` separate ``(B, I)`` tensors, one per lambda variable.
    """

    def embed_program(self, program):
        return program

    def embed_initialized_program(self, program_module):
        return _ChannelUnpackModule(program_module)


def _guard_callables(fn, **kwargs):
    """Handle callable vs constant arguments for add/mul/sub."""
    if any(callable(value) for value in kwargs.values()):
        return lambda *args: fn(
            **{
                key: (value(*args) if callable(value) else value)
                for key, value in kwargs.items()
            }
        )
    return fn(**kwargs)


def _allow_only_non_channel_types(typ):
    """TODO: Rename
    Filter type variable to exclude channel types."""
    match typ:
        case ArrowType(input_type, output_type):
            return _allow_only_non_channel_types(
                input_type
            ) and _allow_only_non_channel_types(output_type)
        case AtomicType(type_name):
            return type_name not in ["channel", "feature"]
        case _:
            return True


def attention_ecg_dsl(
    num_channels,
    features_per_channel,
    num_classes,
    hidden_dim=16,
    max_overall_depth=6,
    use_shields=False,
):
    """Build the attention ECG DSL.

    Each lambda variable corresponds to one channel (12 leads + 9 globals).
    Each channel is a feature vector of size ``features_per_channel``.

    :param num_channels: Number of input channels (leads + globals).
    :param features_per_channel: Feature vector size per channel.
    :param num_classes: Number of output classes.
    :param hidden_dim: Hidden dimension for affine projections.
    :param max_overall_depth: Maximum depth for DSL expansion.
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

    dslf.filtered_type_variable("num", _allow_only_non_channel_types)

    dslf.production(
        "add",
        "(%num, %num) -> %num",
        lambda x, y: _guard_callables(fn=lambda x, y: x + y, x=x, y=y),
    )
    dslf.production(
        "mul",
        "(%num, %num) -> %num",
        lambda x, y: _guard_callables(fn=lambda x, y: x * y, x=x, y=y),
    )
    dslf.production(
        "sub",
        "(%num, %num) -> %num",
        lambda x, y: _guard_callables(fn=lambda x, y: x - y, x=x, y=y),
    )

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
    # Target type ($fInp * num_channels) -> $fOut has depth log2(num_channels+2).
    # With 21 channels that's ~4.52, so max_type_depth=5 is the minimum that works.
    dslf.lambdas(max_type_depth=5)
    if use_shields:
        add_shield_productions(dslf)
    dslf.prune_to(f"({'$fInp, ' * num_channels}) -> $fOut")
    return dslf.finalize()
