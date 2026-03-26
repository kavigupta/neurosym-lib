# pylint: disable=duplicate-code,unnecessary-lambda
"""Attention-based DSL for ECG classification with ECGDeli features.

Combines affine feature selectors with channel-aware attention over 12 ECG
leads.  Each feature group (amplitude, interval, ST, morphology, global) gets
both a flat affine selector and a masked-attention selector that respects lead
groupings.
[<-----lead1----->]...[<-----lead12----->][<--global feats->]
[<f1><f2>....<f14>]...[<f1><f2>.....<f14>][<g1><g2>.....<g9>]

"""

import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.cost import ProgramEmbedding
from neurosym.examples.near.neural_hole_filler import NeuralHoleFiller
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.types.type import ArrowType, AtomicType

from neurosym.datasets.ecg_data_example import (
    LEAD_NAMES,
    _GLOBAL_FEATURES
)

class AffineChannelAttention(nn.Module):
    """Masked attention over per-lead features within a feature group.

    Extracts per-lead features from the flat input vector using pre-computed
    column indices, applies a channel mask, computes attention-weighted
    combination, and projects through an affine layer.
    """

    def __init__(self, per_lead_indices, hidden_dim, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # per_lead_indices: dict[str, LongTensor] mapping lead name -> global col indices
        self.lead_names_ordered = list(per_lead_indices.keys())
        self.num_active_leads = len(self.lead_names_ordered)
        # Map lead names to lead index (0-11) for channel mask alignment
        lead_name_to_idx = {name: i for i, name in enumerate(LEAD_NAMES)}
        self.register_buffer(
            "lead_positions",
            torch.tensor(
                [lead_name_to_idx[n] for n in self.lead_names_ordered], dtype=torch.long
            ),
        )
        # Store per-lead column indices as a padded tensor
        index_lists = [per_lead_indices[n] for n in self.lead_names_ordered]
        self.features_per_lead = len(index_lists[0])
        # Register each lead's indices as a buffer
        stacked = torch.stack(index_lists)  # (num_leads, features_per_lead)
        self.register_buffer("col_indices", stacked)

        self.query = nn.Parameter(torch.randn(self.features_per_lead))
        self.projection = nn.Sequential(
            nn.Linear(self.features_per_lead, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, channel_mask):
        """Forward pass.

        Args:
            x: (B, input_dim) flat feature vector.
            channel_mask: (12,) or (B, 12) boolean/float mask over leads.
        """
        batch_size = x.shape[0]
        # Extract per-lead features: (B, num_active_leads, features_per_lead)
        per_lead = x[:, self.col_indices]  # fancy indexing

        # Get channel mask for our active leads
        if channel_mask.dim() == 1:
            channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1)
        active_mask = channel_mask[:, self.lead_positions]  # (B, num_active_leads)

        valid = active_mask > 0
        has_valid = valid.any(dim=-1, keepdim=True)
        valid = torch.where(has_valid, valid, torch.ones_like(valid))

        # Compute attention scores
        scores = torch.einsum("blf,f->bl", per_lead, self.query)
        masked_scores = scores.masked_fill(~valid, -1e9)
        attention = torch.softmax(masked_scores, dim=-1)

        # Mask prior weighting
        mask_prior = active_mask.clamp(min=0.0)
        prior_sum = mask_prior.sum(dim=-1, keepdim=True)
        mask_prior = torch.where(prior_sum > 0, mask_prior / prior_sum, attention)
        attention = attention * mask_prior
        att_sum = attention.sum(dim=-1, keepdim=True)
        attention = torch.where(
            att_sum > 0,
            attention / att_sum,
            torch.softmax(masked_scores, dim=-1),
        )

        # Weighted sum -> project
        context = torch.einsum("bl,blf->bf", attention, per_lead)
        return self.projection(context)


class AffineFeatureSelector(nn.Module):
    """Flat affine selector for a feature group (no channel awareness)."""

    def __init__(self, group_indices, hidden_dim, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("col_indices", group_indices)
        self.linear = nn.Sequential(
            nn.Linear(len(group_indices), hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(x[:, self.col_indices])


class AffineBoolSelector(nn.Module):
    """Affine selector that produces a scalar boolean condition."""

    def __init__(self, group_indices, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("col_indices", group_indices)
        self.linear = nn.Linear(len(group_indices), 1)

    def forward(self, x):
        return self.linear(x[:, self.col_indices])


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


def _channel_selector(channel_indices):
    """Create a function that returns a boolean mask for the given channel indices."""
    channel_indices = torch.as_tensor(channel_indices, dtype=torch.long)

    def _selector(x):
        mask = torch.zeros(
            *x.shape[:-1],
            len(LEAD_NAMES),
            device=x.device,
            dtype=torch.float32,
        )
        mask[..., channel_indices.to(x.device)] = 1.0
        return mask

    return _selector


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
        "output",
        "$fHid -> $fOut",
        lambda x, lin: lin(x),
        dict(lin=lambda: nn.Linear(hidden_dim, num_classes)),
    )
    # Target type ($fInp * num_channels) -> $fOut has depth log2(num_channels+2).
    # With 21 channels that's ~4.52, so max_type_depth=5 is the minimum that works.
    dslf.lambdas(max_type_depth=5)
    dslf.prune_to(f"({'$fInp, ' * num_channels}) -> $fOut")
    return dslf.finalize()
