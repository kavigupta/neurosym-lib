import torch
from torch import nn

from neurosym.dsl.dsl_factory import DSLFactory


def differentiable_arith_dsl(length):
    """
    A simple differentiable DSL for arithmetic operations on tensors.

    :param length: Length of the tensors to operate on
    """
    dslf = DSLFactory(L=length)
    dslf.concrete("one", "() -> f", lambda: torch.tensor(1.0))
    dslf.concrete("ones", "() -> {f, $L}", lambda: torch.ones(length))
    dslf.concrete("int_int_add", "(f, f) -> f", lambda x, y: x + y)
    dslf.concrete("Tint_int_add", "({f, $L}, f) -> {f, $L}", lambda x, y: x + y)
    dslf.concrete("Tint_Tint_add", "({f, $L}, {f, $L}) -> {f, $L}", lambda x, y: x + y)
    dslf.concrete(
        "app_Tint", "({f, $L} -> {f, $L}, {f, $L}) -> {f, $L}", lambda f, x: f(x)
    )
    dslf.parameterized(
        "Linear_c",
        "() -> {f, $L} -> {f, $L}",
        lambda linear: linear,
        dict(linear=lambda: nn.Linear(length, length)),
    )
    return dslf.finalize()
