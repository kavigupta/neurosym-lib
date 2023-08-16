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

from neurosym.types.type_string_repr import TypeDefiner
from ..dsl.production import ConcreteProduction, ParameterizedProduction
from ..dsl.dsl import DSL
from ..types.type_signature import ConcreteTypeSignature
from neurosym.operations.basic import ite_torch
from neurosym.operations.lists import fold_torch, map_torch


def example_rnn_dsl(length, out_length):
    t = TypeDefiner(L=length, O=out_length)
    t.typedef("fL", "{f, $L}")

    return DSL(
        [
            ConcreteProduction(
                "Tfloat_Tfloat_add",
                ConcreteTypeSignature([], t("($fL, $fL) -> $fL")),
                lambda: lambda x, y: x + y,
            ),
            ConcreteProduction(
                "Tfloat_Tfloat_mul",
                ConcreteTypeSignature([], t("($fL, $fL) -> $fL")),
                lambda: lambda x, y: x * y,
            ),
            ConcreteProduction(
                "fold",
                ConcreteTypeSignature([t("($fL, $fL) -> $fL")], t("([$fL]) -> $fL")),
                lambda f: lambda x: fold_torch(f, x),
            ),
            ConcreteProduction(
                "Sum",
                ConcreteTypeSignature([], t("($fL) -> f")),
                lambda: lambda x: torch.sum(x, dim=-1).unsqueeze(-1),
            ),
            ParameterizedProduction(
                "Linear_c",
                ConcreteTypeSignature([], t("($fL) -> $fL")),
                lambda linear: linear,
                dict(linear=lambda: nn.Linear(length, length)),
            ),
            ParameterizedProduction(
                "output",
                ConcreteTypeSignature(
                    [t("([$fL]) -> [$fL]")], t("([$fL]) -> [{f, $O}]")
                ),
                lambda f, linear: lambda x: linear(f(x)),
                dict(linear=lambda: nn.Linear(length, out_length)),
            ),
            ConcreteProduction(
                "Tlist_float_ITE",
                ConcreteTypeSignature(
                    [
                        t("([$fL]) -> f"),
                        t("([$fL]) -> [$fL]"),
                        t("([$fL]) -> [$fL]"),
                    ],
                    t("([$fL]) -> [$fL]"),
                ),
                lambda cond, fx, fy: ite_torch(cond, fx, fy),
            ),
            ConcreteProduction(
                "Map",
                ConcreteTypeSignature([t("($fL) -> $fL")], t("([$fL]) -> [$fL]")),
                lambda f: lambda x: map_torch(f, x),
            ),
            ConcreteProduction(
                "list_Tfloat_list_Tfloat_compose",
                ConcreteTypeSignature(
                    [t("([$fL]) -> [$fL]"), t("([$fL]) -> [$fL]")],
                    t("([$fL]) -> [$fL]"),
                ),
                lambda f, g: lambda x: f(g(x)),
            ),
            ConcreteProduction(
                "List_Tfloat_Tfloat_bool_compose",
                ConcreteTypeSignature(
                    [t("([$fL]) -> $fL"), t("($fL) -> f")], t("([$fL]) -> f")
                ),
                lambda f, g: lambda x: f(g(x)),
            ),
        ]
    )
