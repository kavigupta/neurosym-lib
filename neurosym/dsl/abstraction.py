from dataclasses import dataclass
from typing import Dict

from .production import Production
from ..types.type_signature import TypeSignature
from ..programs.s_expression import InitializedSExpression, SExpression


@dataclass
class AbstractionIndexParameter:
    index: int

    def apply(self, inputs):
        return AbstractionParameter(inputs[self.index])

    def __initialize__(self, dsl):
        return self


@dataclass
class AbstractionParameter:
    value: object

    def __compute_value__(self, dsl):
        return self.value


@dataclass
class AbstractionProduction(Production):
    _symbol: str
    _type_signature: TypeSignature
    _body: SExpression

    def symbol(self):
        return self._symbol

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        return dict(
            body=dsl.initialize(self._body),
        )

    def compute_on_pytorch(self, dsl, state, inputs):
        initialized_body = state["body"]
        print(initialized_body)
        return dsl.compute_on_pytorch(with_index_parameters(initialized_body, inputs))


def with_index_parameters(ise: InitializedSExpression, inputs: tuple):
    if isinstance(ise, AbstractionIndexParameter):
        return ise.apply(inputs)
    return InitializedSExpression(
        ise.symbol,
        tuple(with_index_parameters(x, inputs) for x in ise.children),
        ise.state,
    )
