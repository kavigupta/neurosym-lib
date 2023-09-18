from dataclasses import dataclass
from typing import Dict

from neurosym.programs.s_expression_render import render_s_expression

from .production import FunctionLikeProduction
from ..types.type_signature import TypeSignature
from ..programs.s_expression import InitializedSExpression, SExpression


@dataclass
class AbstractionIndexParameter:
    """
    Represents a parameter of an abstraction.
    Used in SExpressions and InitializedSExpressions.
    """

    index: int

    def apply(self, inputs):
        """
        Map the parameter to a value.
        """
        return AbstractionParameter(inputs[self.index])

    def __initialize__(self, dsl):
        return self

    def __to_pair__(self, for_stitch):
        return f"#{self.index}"


@dataclass
class AbstractionParameter:
    """
    Represents a parameter of an abstraction, with an initialized value.

    Used in InitializedSExpressions.
    """

    value: object

    def __compute_value__(self, dsl):
        return self.value


@dataclass
class AbstractionProduction(FunctionLikeProduction):
    """
    Represents an abstraction in a DSL.
    """

    _symbol: str
    _type_signature: TypeSignature
    _body: SExpression

    def base_symbol(self):
        return self._symbol

    def get_index(self):
        return None

    def with_index(self, index):
        raise NotImplementedError

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        return dict(
            body=dsl.initialize(self._body),
        )

    def evaluate(self, dsl, state, inputs):
        initialized_body = state["body"]
        return dsl.compute(with_index_parameters(initialized_body, inputs))

    def render_as_lambda(self):
        body = render_s_expression(self._body, False)
        arguments = " ".join(
            render_s_expression(AbstractionIndexParameter(i), False)
            for i in range(self._type_signature.arity())
        )
        return f"(lam ({arguments}) {body})"

    def render(self):
        return f"{self._symbol:>15} :: {self._type_signature.render()} = {self.render_as_lambda()}"


def with_index_parameters(ise: InitializedSExpression, inputs: tuple):
    """
    With the given inputs, replace all AbstractionIndexParameters in the given
    InitializedSExpression with AbstractionParameters.

    Args:
        ise: The InitializedSExpression.
        inputs: The inputs.

    Returns a new InitializedSExpression.
    """
    if isinstance(ise, AbstractionIndexParameter):
        return ise.apply(inputs)
    return InitializedSExpression(
        ise.symbol,
        tuple(with_index_parameters(x, inputs) for x in ise.children),
        ise.state,
    )
