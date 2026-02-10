# pylint: disable=duplicate-code,cyclic-import
import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.types.type import ArrowType, AtomicType


def _subset_selector_all_feat(x, channel, typ):
    x = x.reshape(-1, 12, 6, 2)
    typ_idx = torch.full(
        size=(x.shape[0],), fill_value=(0 if typ == "interval" else 1), device=x.device
    )
    channel_mask = channel(x.reshape(-1, 144))  # [B, 12]
    masked_x = (x * channel_mask[..., None, None]).sum(1)
    return masked_x[torch.arange(x.shape[0]), :, typ_idx]


def simple_ecg_dsl(input_dim, num_classes, hidden_dim=None, max_overall_depth=6):
    """
    A differentiable ECG DSL that mirrors the channel/interval DSL in ecg_demo.

    :param input_dim: Number of input features (expected 12*6*2 = 144).
    :param num_classes: Number of output classes (or labels).
    :param hidden_dim: Unused, kept for API parity with other DSLs.
    :param max_overall_depth: Maximum depth for DSL expansion.
    """
    feature_dim = 6
    dslf = DSLFactory(
        I=input_dim, O=num_classes, F=feature_dim, max_overall_depth=max_overall_depth
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fFeat", "{f, $F}")

    for i in range(12):
        dslf.concrete(
            f"channel_{i}",
            "() -> () -> channel",
            lambda: lambda x: torch.nn.functional.one_hot(
                torch.full(tuple(x.shape[:-1]), i, device=x.device, dtype=torch.long),
                num_classes=12,
            ),
        )

    dslf.concrete(
        "select_interval",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda ch: lambda x: _subset_selector_all_feat(x, ch, "interval"),
    )

    dslf.concrete(
        "select_amplitude",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda ch: lambda x: _subset_selector_all_feat(x, ch, "amplitude"),
    )

    def guard_callables(fn, **kwargs):
        is_callable = [callable(kwargs[k]) for k in kwargs]
        if any(is_callable):
            return lambda z: fn(
                **{
                    k: (kwargs[k](z) if is_callable[i] else kwargs[k])
                    for i, k in enumerate(kwargs)
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

    dslf.filtered_type_variable("num", lambda x: filter_constants(x))
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
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
    )

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize()
