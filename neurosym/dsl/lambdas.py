from dataclasses import dataclass
from typing import List

from neurosym.dsl.abstraction import AbstractionIndexParameter, AbstractionParameter
from neurosym.dsl.dsl import DSL
from neurosym.dsl.production import LambdaProduction, VariableProduction
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.types.type_signature import LambdaTypeSignature


@dataclass
class LambdaFunction:
    dsl: DSL
    body: InitializedSExpression
    typ: LambdaTypeSignature

    @classmethod
    def of(cls, dsl, body, typ):
        return cls(dsl, body, typ)

    def __call__(self, *args):
        assert len(args) == self.typ.function_arity()
        body = replace_variables_with_args(self.dsl, self.body)
        # Reverse the arguments because we want to inject them from the right.
        body = inject(self.dsl, body, args[::-1])
        return self.dsl.compute(body)


def replace_variables_with_args(dsl: DSL, expr: InitializedSExpression):
    """
    Replace variable nodes with AbstractionIndexParameter nodes.
    """
    if isinstance(expr, AbstractionParameter):
        return expr
    if isinstance(expr, AbstractionIndexParameter):
        return expr
    assert isinstance(expr, InitializedSExpression)
    prod = dsl.get_production(expr.symbol)
    if isinstance(prod, VariableProduction):
        return AbstractionIndexParameter(prod.index_in_env)
    return InitializedSExpression(
        expr.symbol,
        [replace_variables_with_args(dsl, child) for child in expr.children],
        expr.state,
    )


def inject(dsl: DSL, expr: InitializedSExpression, args: List[object], frame_shift=0):
    """
    Inject the arguments into the expression, and frame shift the indices.
    """
    if isinstance(expr, AbstractionParameter):
        return expr
    if isinstance(expr, AbstractionIndexParameter):
        assert expr.index - frame_shift < len(args)
        if expr.index - frame_shift >= 0:
            return AbstractionParameter(args[expr.index - frame_shift])
        return expr
    prod = dsl.get_production(expr.symbol)
    children = expr.children
    if isinstance(prod, LambdaProduction):
        frame_shift += prod.type_signature().function_arity()
    return InitializedSExpression(
        expr.symbol,
        [inject(dsl, child, args, frame_shift=frame_shift) for child in children],
        expr.state,
    )
