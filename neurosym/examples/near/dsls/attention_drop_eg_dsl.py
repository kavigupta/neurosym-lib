# pylint: disable=duplicate-code
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.types.type import ArrowType, AtomicType

ECG_NUM_CHANNELS = 12
ECG_FEATURE_DIM = 6
ECG_VALUE_TYPES = ("interval", "amplitude")

# Group channels with similar clinical semantics.
ECG_CHANNEL_GROUPS = {
    "all": tuple(range(ECG_NUM_CHANNELS)),
    "limb": (0, 1, 2),
    "augmented_limb": (3, 4, 5),
    "precordial": (6, 7, 8, 9, 10, 11),
    "inferior": (1, 2, 5),
    "lateral": (0, 4, 10, 11),
    "septal": (6, 7),
    "anterior": (8, 9),
}


class MaskedAttentionFeatureExtractor(nn.Module):
    """
    Attention feature extractor with explicit channel masking.

    Input data is ``(B, 144)`` reshaped to ``(B, 12, 6, 2)``; the
    *value_type* index selects interval (0) or amplitude (1) features,
    yielding ``(B, 12, 6)``.  A learnable query vector computes attention
    scores over channels, masked so that excluded channels get ``-inf``
    before softmax.  An optional mask-prior further weights the attention.
    """

    def __init__(self, *, value_type: str, feature_dim: int = ECG_FEATURE_DIM):
        super().__init__()
        if value_type not in ECG_VALUE_TYPES:
            raise ValueError(f"Unknown ECG value type: {value_type}")
        self.value_type = value_type
        self.feature_dim = feature_dim
        self.query = nn.Parameter(torch.randn(ECG_FEATURE_DIM))
        self.projection = nn.Sequential(
            nn.Linear(ECG_FEATURE_DIM, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(
            -1, ECG_NUM_CHANNELS * ECG_FEATURE_DIM * len(ECG_VALUE_TYPES)
        )
        structured_x = flat_x.reshape(-1, ECG_NUM_CHANNELS, ECG_FEATURE_DIM, 2)
        type_idx = ECG_VALUE_TYPES.index(self.value_type)
        signal = structured_x[:, :, :, type_idx]  # [B, 12, 6]

        channel_mask = channel_mask.reshape(-1, ECG_NUM_CHANNELS)
        valid = channel_mask > 0
        has_valid = valid.any(dim=-1, keepdim=True)
        valid = torch.where(has_valid, valid, torch.ones_like(valid))

        scores = torch.einsum("bkc,c->bk", signal, self.query)
        masked_scores = scores.masked_fill(~valid, -1e9)
        attention = torch.softmax(masked_scores, dim=-1)

        mask_prior = channel_mask.clamp(min=0.0)
        prior_sum = mask_prior.sum(dim=-1, keepdim=True)
        mask_prior = torch.where(prior_sum > 0, mask_prior / prior_sum, attention)

        attention = attention * mask_prior
        attention_sum = attention.sum(dim=-1, keepdim=True)
        attention = torch.where(
            attention_sum > 0,
            attention / attention_sum,
            torch.softmax(masked_scores, dim=-1),
        )

        context = torch.einsum("bk,bkc->bc", attention, signal)
        return self.projection(context)


def _channel_selector(channel_indices):
    channel_indices = torch.as_tensor(channel_indices, dtype=torch.long)

    def _selector(x):
        mask = torch.zeros(
            *x.shape[:-1],
            ECG_NUM_CHANNELS,
            device=x.device,
            dtype=torch.float32,
        )
        mask[..., channel_indices.to(x.device)] = 1.0
        return mask

    return _selector


def _drop_variables(base_mask, variables_to_drop):
    return torch.clamp(base_mask - variables_to_drop, min=0.0)


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


def attention_drop_eg_dsl(input_dim, num_classes, hidden_dim=None, max_overall_depth=6):
    """
    An attention-based ECG DSL with explicit channel masking and ``drop_variables``.

    Combines channel/group selectors with masked-attention feature extractors
    so that program search can discover which ECG leads matter.

    :param input_dim: Number of input features (expected 12*6*2 = 144).
    :param num_classes: Number of output classes.
    :param hidden_dim: Unused, kept for API parity.
    :param max_overall_depth: Maximum depth for DSL expansion.
    """
    del hidden_dim

    dslf = DSLFactory(
        I=input_dim,
        O=num_classes,
        F=ECG_FEATURE_DIM,
        max_overall_depth=max_overall_depth,
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fFeat", "{f, $F}")

    for channel_idx in range(ECG_NUM_CHANNELS):
        dslf.production(
            f"channel_{channel_idx}",
            "() -> () -> channel",
            lambda channel_idx=channel_idx: _channel_selector((channel_idx,)),
        )

    for group_name, channel_indices in ECG_CHANNEL_GROUPS.items():
        dslf.production(
            f"channel_group_{group_name}",
            "() -> () -> channel",
            lambda channel_indices=channel_indices: _channel_selector(channel_indices),
        )

    dslf.production(
        "drop_variables",
        "(() -> channel, () -> channel) -> () -> channel",
        lambda channel_selector, variables_to_drop: lambda x: _drop_variables(
            channel_selector(x), variables_to_drop(x)
        ),
    )

    dslf.production(
        "attention_interval",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda channel_selector, extractor: lambda x: extractor(x, channel_selector(x)),
        dict(
            extractor=lambda: MaskedAttentionFeatureExtractor(value_type="interval")
        ),
    )
    dslf.production(
        "attention_amplitude",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda channel_selector, extractor: lambda x: extractor(x, channel_selector(x)),
        dict(
            extractor=lambda: MaskedAttentionFeatureExtractor(value_type="amplitude")
        ),
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
        dict(lin=lambda: nn.Linear(ECG_FEATURE_DIM, 1)),
    )
    dslf.production(
        "output",
        "(($fInp) -> $fFeat) -> $fInp -> $fOut",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(ECG_FEATURE_DIM, num_classes)),
    )

    dslf.production(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        ite_torch,
    )

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize()


def attention_drop_ecg_dsl(
    input_dim, num_classes, hidden_dim=None, max_overall_depth=6
):
    """Alias with corrected ECG spelling."""
    return attention_drop_eg_dsl(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        max_overall_depth=max_overall_depth,
    )
