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
from ..dsl.production import ConcreteProduction, ParameterizedProduction
from ..dsl.dsl import DSL
from ..types.type import AtomicType, ListType
from ..types.type_signature import ConcreteTypeSignature
from neurosym.operations.basic import ite_torch
from neurosym.operations.lists import fold_torch, map_torch

linear_obj = AtomicType("Tfloat_Tfloat")
binop_Tfloat = AtomicType("BinOp")
list_Tfloat_bool = AtomicType("list_Tfloat_bool")
list_Tfloat_Tfloat = AtomicType("list_Tfloat_Tfloat")
list_Tfloat_list_Tfloat = AtomicType("list_Tfloat_list_Tfloat")
Tfloat_bool = AtomicType("Tfloat_bool")
outTfloat = AtomicType("outTfloat")
list_Tfloat_list_outTfloat = AtomicType("list_Tfloat_list_outTfloat")

example_rnn_dsl = lambda length, out_length: DSL(
    [
        ConcreteProduction(
            "Tfloat_Tfloat_add",
            ConcreteTypeSignature([], binop_Tfloat),
            lambda: lambda x, y: x + y,
        ),
        ConcreteProduction(
            "Tfloat_Tfloat_mul",
            ConcreteTypeSignature([], binop_Tfloat),
            lambda: lambda x, y: x * y,
        ),
        ConcreteProduction(
            "fold",
            ConcreteTypeSignature([binop_Tfloat], list_Tfloat_Tfloat),
            lambda f: lambda x: fold_torch(f, x),
        ),
        ConcreteProduction(
            "Sum",
            ConcreteTypeSignature([], Tfloat_bool),
            lambda: lambda x: torch.sum(x, dim=-1).unsqueeze(-1),
        ),
        ParameterizedProduction(
            "Linear_c",
            ConcreteTypeSignature([], linear_obj),
            lambda linear: linear,
            dict(linear=lambda: nn.Linear(length, length)),
        ),
        ParameterizedProduction(
            "output",
            ConcreteTypeSignature([list_Tfloat_list_Tfloat], list_Tfloat_list_outTfloat),
            lambda f, linear: lambda x: linear(f(x)),
            dict(linear=lambda: nn.Linear(length, out_length)),
        ),
        ConcreteProduction(
            "Tlist_float_ITE",
            ConcreteTypeSignature(
                [list_Tfloat_bool, list_Tfloat_Tfloat, list_Tfloat_Tfloat],
                list_Tfloat_Tfloat,
            ),
            lambda cond, fx, fy: ite_torch(cond, fx, fy),
        ),
        ConcreteProduction(
            "Map",
            ConcreteTypeSignature([linear_obj], list_Tfloat_list_Tfloat),
            lambda f: lambda x: map_torch(f, x),
        ),
        ConcreteProduction(
            "list_Tfloat_list_Tfloat_compose",
            ConcreteTypeSignature(
                [list_Tfloat_list_Tfloat, list_Tfloat_list_Tfloat],
                list_Tfloat_list_Tfloat,
            ),
            lambda f, g: lambda x: f(g(x)),
        ),
        ConcreteProduction(
            "List_Tfloat_Tfloat_bool_compose",
            ConcreteTypeSignature([list_Tfloat_Tfloat, Tfloat_bool], list_Tfloat_bool),
            lambda f, g: lambda x: f(g(x)),
        ),
    ]
)
