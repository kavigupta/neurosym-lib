"""
DSL of the form:

# one :: float
# ones :: tensor[float]
# int_int_add :: float -> float -> float
# Tint_int_add :: tensor[float] -> float -> tensor[float]
# Tint_Tint_add :: tensor[float] -> tensor[float] -> tensor[float]
# Function signature for Tint_linear
# Tint_linear :: LinearObj -> tensor[float] -> tensor[float]
# Linear_c :: LinearQbj
"""
import torch
import torch.nn as nn
from ..dsl.production import ConcreteProduction, ParameterizedProduction
from ..dsl.dsl import DSL
from ..types.type import AtomicType, ListType
from ..types.type_signature import ConcreteTypeSignature


# int_type = AtomicType("int")
float_type = AtomicType("float")
linear_obj_type = AtomicType("LinearObj")
list_float_type = ListType(float_type)


def differentiable_arith_dsl(length):
    return DSL([ConcreteProduction("one", ConcreteTypeSignature([], float_type), lambda: torch.tensor(1.0)), ConcreteProduction("ones", ConcreteTypeSignature([], list_float_type), lambda: torch.ones(length)), ConcreteProduction("int_int_add", ConcreteTypeSignature([float_type, float_type], float_type), lambda x, y: x + y), ConcreteProduction("Tint_int_add", ConcreteTypeSignature([list_float_type, float_type], list_float_type), lambda x, y: x + y), ConcreteProduction("Tint_Tint_add", ConcreteTypeSignature([list_float_type, list_float_type], list_float_type), lambda x, y: x + y), ConcreteProduction("Tint_linear", ConcreteTypeSignature([linear_obj_type, list_float_type], list_float_type), lambda f, x: f(x)), ParameterizedProduction("Linear_c", ConcreteTypeSignature([], linear_obj_type), lambda linear: linear, dict(linear=lambda: nn.Linear(length, length)))])
