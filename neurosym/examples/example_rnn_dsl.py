"""
RNN example


('list', 'atom') : [dsl.FoldFunction, dsl.SimpleITE],
('atom', 'atom') : [dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE,LinearAffine]
# 
# Tfloat_Tfloat_add :: BinOp
# Tfloat_Tfloat_mul :: BinOp
# Linear_c :: (Tfloat -> Tfloat)
# fold :: BinOp -> list[tensor[float]] -> tensor[float]
# map :: (Tfloat -> Tfloat) -> list_Tfloat -> list_Tfloat
# Tlist_float_ITE :: (list_Tfloat_bool) -> (list_Tfloat_Tfloat) -> (list_Tfloat_Tfloat) -> (list_Tfloat_Tfloat)
# Tfloat_ITE :: (Tfloat_bool) -> (BinOp) -> (BinOp)
"""
import torch
import torch.nn as nn
from neurosym.dsl.dsl_factory import DSLFactory

from neurosym.operations.basic import ite_torch
from neurosym.operations.lists import fold_torch, map_torch


def example_rnn_dsl(length, out_length):
    dslf = DSLFactory(L=length, O=out_length, max_overall_depth=5)
    dslf.typedef("fL", "{f, $L}")

    dslf.concrete(
        "Tfloat_Tfloat_add", "() -> ($fL, $fL) -> $fL", lambda: lambda x, y: x + y
    )
    dslf.concrete(
        "Tfloat_Tfloat_mul", "() -> ($fL, $fL) -> $fL", lambda: lambda x, y: x * y
    )
    dslf.concrete(
        "fold",
        "(($fL, $fL) -> $fL) -> ([$fL]) -> $fL",
        lambda f: lambda x: fold_torch(f, x),
    )
    dslf.concrete(
        "Sum",
        "() -> ($fL) -> f",
        lambda: lambda x: torch.sum(x, dim=-1).unsqueeze(-1),
    )
    dslf.parameterized(
        "Linear_c",
        "() -> ($fL) -> $fL",
        lambda linear: linear,
        dict(linear=lambda: nn.Linear(length, length)),
    )
    dslf.parameterized(
        "output",
        "(([$fL]) -> [$fL]) -> ([$fL]) -> [{f, $O}]",
        lambda f, linear: lambda x: linear(f(x)),
        dict(linear=lambda: nn.Linear(length, out_length)),
    )
    # This is causing an issue. The inner program (f) has a output shape of out_length.
    # The type system is not filtering out programs that have the wrong output dim.
    dslf.concrete(
        "Tfloat_ITE",
        "(([$fL]) -> f, ([$fL]) -> [$fL], ([$fL]) -> [$fL]) -> ([$fL]) -> [$fL]",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
    )
    dslf.concrete(
        "Map",
        "(($fL) -> $fL) -> ([$fL]) -> [$fL]",
        lambda f: lambda x: map_torch(f, x),
    )

    dslf.concrete(
        "compose",
        "(#a -> #b, #b -> #c) -> #a -> #c",
        lambda f, g: lambda x: f(g(x)),
    )

    dslf.prune_to("[{f, $L}] -> [{f, $O}]")
    return dslf.finalize()
