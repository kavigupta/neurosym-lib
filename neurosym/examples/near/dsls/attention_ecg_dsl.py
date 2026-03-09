import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.types.type import ArrowType, AtomicType


class AttentionFeatureExtractor(nn.Module):
    def __init__(self, input_dim=144, context_slice_dim=12, feature_dim=6):
        """
        :param input_dim: Flattened input dimension (12 leads * 6 timepoints * 2 values) = 144
        :param context_slice_dim: Dimension of a single channel's data (6*2=12)
        :param feature_dim: Dimension of the output feature (e.g. 6)
        """
        super().__init__()
        self.input_dim = input_dim
        self.context_slice_dim = context_slice_dim
        self.feature_dim = feature_dim
        self.num_channels = input_dim // context_slice_dim # 12

        # Learnable attention query vector
        self.query = nn.Parameter(torch.randn(self.context_slice_dim))

        # Transformation for the extracted feature
        self.projection = nn.Sequential(
            nn.Linear(self.context_slice_dim, self.feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [Batch, 144]
        batch_size = x.shape[0]

        # Reshape to [Batch, Channels, SliceDim] -> [Batch, 12, 12]
        x_reshaped = x.view(batch_size, self.num_channels, self.context_slice_dim)

        # Compute attention scores: Dot product of query with each channel slice
        # query shape: [12] -> [1, 1, 12]
        # x_reshaped: [B, 12, 12]
        # scores: [B, 12]
        scores = torch.einsum('bkc,c->bk', x_reshaped, self.query)

        # Attention weights
        attn_weights = torch.softmax(scores, dim=1) # [B, 12]

        # Weighted sum of channels (Soft Attention)
        # context: [B, 12]
        context = torch.einsum('bk,bkc->bc', attn_weights, x_reshaped)

        # Project to feature space
        return self.projection(context)


def attention_ecg_dsl(input_dim, num_classes, hidden_dim=None, max_overall_depth=6):
    """
    An Attention-based differentiable ECG DSL.
    Replaces discrete channel selection with soft attention.

    :param input_dim: Number of input features (expected 12*6*2 = 144).
    :param num_classes: Number of output classes.
    :param hidden_dim: Unused, kept for API parity.
    :param max_overall_depth: Maximum depth for DSL expansion.
    """
    del hidden_dim
    feature_dim = 6
    dslf = DSLFactory(
        I=input_dim, O=num_classes, F=feature_dim, max_overall_depth=max_overall_depth
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fFeat", "{f, $F}")

    # [NEW] Parameterized Extractor with internal Attention
    # Replaces explicit 'channel' selection with internal soft attention
    def debug_interval(lin):
        def inner(u):
            # print(f"DEBUG: neural_interval called with u={u}")
            def core(x):
                # print(f"DEBUG: neural_interval core called with type(x)={type(x)}")
                res = lin(x)
                # print(f"DEBUG: neural_interval returning type(res)={type(res)}")
                return res
            return core
        return inner

    dslf.parameterized(
        "neural_interval",
        "() -> ($fInp) -> $fFeat",
        lambda lin: lambda x: lin(x),
        dict(lin=lambda: AttentionFeatureExtractor(input_dim=input_dim)),
    )

    dslf.parameterized(
        "neural_amplitude",
        "() -> ($fInp) -> $fFeat",
        lambda lin: lambda x: lin(x),
        dict(lin=lambda: AttentionFeatureExtractor(input_dim=input_dim)),
    )

    def guard_callables(fn, **kwargs):
        if any(callable(value) for value in kwargs.values()):
            return lambda z: fn(
                **{
                    key: (value(z) if callable(value) else value)
                    for key, value in kwargs.items()
                }
            )
        return fn(**kwargs)

    def filter_constants(x):
        match x:
            case ArrowType(a, b):
                return filter_constants(a) and filter_constants(b)
            case AtomicType(a):
                return a not in ["channel", "feature"]
            case _:
                return True

    dslf.filtered_type_variable("num", filter_constants)
    dslf.concrete(
        "add",
        "(%num, %num) -> %num",
        lambda x, y: guard_callables(fn=lambda x, y: x + y, x=x, y=y),
    )
    dslf.concrete(
        "mul",
        "(%num, %num) -> %num",
        lambda x, y: guard_callables(fn=lambda x, y: x * y, x=x, y=y),
    )



    dslf.parameterized(
        "linear",
        "(($fInp) -> $fFeat) -> $fInp -> {f, 1}",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, 1)),
    )

    dslf.parameterized(
        "output",
        "(($fInp) -> $fFeat) -> $fInp -> $fOut",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, num_classes)),
    )

    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        ite_torch,
    )

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize()
