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

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.models.constant import ConstantConfig, Constant


def simple_constants_dsl(length):
    constant_dslf = DSLFactory(L=length, max_overall_depth=5)
    constant_dslf.typedef("fL", "{f, $L}")
    constant_dslf.concrete("ones", "() -> $fL", lambda: torch.ones(length))
    constant_dslf.concrete("add", "() -> ($fL, $fL) -> $fL", lambda: lambda x, y: x + y)
    constant_dslf.parameterized(
        "constant",
        "() -> $fL",
        lambda const: const,
        dict(
            const=lambda: Constant(
                ConstantConfig(model_name="constant", size=length, init="random")
            )
        ),
    )
    # constant_dslf.prune_to("() -> $fL")
    return constant_dslf.finalize()
