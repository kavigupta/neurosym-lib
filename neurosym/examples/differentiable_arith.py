"""
DSL of the form:

# one :: float
# ones :: tensor[float]
# int_int_add :: float -> float -> float
# Tint_int_add :: tensor[float] -> float -> tensor[float]
# Tint_Tint_add :: tensor[float] -> tensor[float] -> tensor[float]
# Function signature for Tint_linear
# Tint_linear :: tensor[float] -> tensor[float]
"""
import torch
from ..dsl.production import ConcreteProduction
from ..dsl.dsl import DSL
from ..types.type import AtomicType, ListType
from ..types.type_signature import ConcreteTypeSignature


# int_type = AtomicType("int")
float_type = AtomicType("float")
list_float_type = ListType(float_type)

#@TODO: How to integrate length?
# 1. DSL object has length built in.
# 2. ....
INPUT_LENGHT = 10

differentiable_arith_dsl = DSL(
    [
        ConcreteProduction(
            "one",
            ConcreteTypeSignature([], float_type),
            lambda: torch.tensor(1.0),
        ),
        ConcreteProduction(
            "ones",
            ConcreteTypeSignature([], list_float_type),
            lambda: torch.ones(INPUT_LENGHT),
        ),
        ConcreteProduction(
            "int_int_add",
            ConcreteTypeSignature([float_type, float_type], float_type),
            lambda x, y: x + y,
        ),
        ConcreteProduction(
            "Tint_int_add",
            ConcreteTypeSignature([list_float_type, float_type], list_float_type),
            lambda x, y: x + y,
        ),
        ConcreteProduction(
            "Tint_Tint_add",
            ConcreteTypeSignature([list_float_type, list_float_type], list_float_type),
            lambda x, y: x + y,
        ),
        ConcreteProduction(
            "Tint_linear",
            ConcreteTypeSignature([list_float_type], list_float_type),
            lambda x: x #@TODO: How to integrate linear?
        )
    ]
)





