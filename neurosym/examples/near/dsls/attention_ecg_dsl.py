# pylint: disable=duplicate-code
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.types.type import ArrowType, AtomicType

ECG_NUM_CHANNELS = 12
ECG_FEATURE_DIM = 6


class AttentionFeatureExtractor(nn.Module):
    """
    Soft-attention feature extractor for ECG data.

    Input: ``(N, 144)`` reshaped to ``(N, 12, 12)`` (12 channels x 12 values).
    A learnable query vector computes attention weights over channels via
    ``softmax(signal @ query)``, producing a context vector that is projected
    to the output feature dimension.
    """

    def __init__(self, input_dim=144, context_slice_dim=12, feature_dim=ECG_FEATURE_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.context_slice_dim = context_slice_dim
        self.feature_dim = feature_dim
        self.num_channels = input_dim // context_slice_dim

        self.query = nn.Parameter(torch.randn(self.context_slice_dim))
        self.projection = nn.Sequential(
            nn.Linear(self.context_slice_dim, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.num_channels, self.context_slice_dim)
        scores = torch.einsum("bkc,c->bk", x_reshaped, self.query)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.einsum("bk,bkc->bc", attn_weights, x_reshaped)
        return self.projection(context)


def _guard_callables(fn, **kwargs):
    if any(callable(value) for value in kwargs.values()):
        return lambda z: fn(
            **{
                key: (value(z) if callable(value) else value)
                for key, value in kwargs.items()
            }
        )
    return fn(**kwargs)


def _allow_only_non_channel_types(typ):
    match typ:
        case ArrowType(input_type, output_type):
            return _allow_only_non_channel_types(
                input_type
            ) and _allow_only_non_channel_types(output_type)
        case AtomicType(type_name):
            return type_name not in ["channel", "feature"]
        case _:
            return True


def attention_ecg_dsl(input_dim, num_classes, hidden_dim=None, max_overall_depth=6):
    """
    A soft-attention ECG DSL.

    Uses ``AttentionFeatureExtractor`` modules that learn to attend over
    the 12 ECG channels internally, replacing explicit channel selectors.

    :param input_dim: Number of input features (expected 12*6*2 = 144).
    :param num_classes: Number of output classes.
    :param hidden_dim: Unused, kept for API parity.
    :param max_overall_depth: Maximum depth for DSL expansion.
    """
    del hidden_dim
    feature_dim = ECG_FEATURE_DIM

    dslf = DSLFactory(
        I=input_dim, O=num_classes, F=feature_dim, max_overall_depth=max_overall_depth
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fFeat", "{f, $F}")

    dslf.production(
        "neural_interval",
        "() -> ($fInp) -> $fFeat",
        lambda lin: lambda x: lin(x),
        dict(lin=lambda: AttentionFeatureExtractor(input_dim=input_dim)),
    )

    dslf.production(
        "neural_amplitude",
        "() -> ($fInp) -> $fFeat",
        lambda lin: lambda x: lin(x),
        dict(lin=lambda: AttentionFeatureExtractor(input_dim=input_dim)),
    )

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
        "linear",
        "(($fInp) -> $fFeat) -> $fInp -> {f, 1}",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, 1)),
    )

    dslf.production(
        "output",
        "(($fInp) -> $fFeat) -> $fInp -> $fOut",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, num_classes)),
    )

    dslf.production(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        ite_torch,
    )

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize()
