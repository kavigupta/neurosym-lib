from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict


from ..types.type_signature import TypeSignature


class Production(ABC):
    """
    Represents a production rule in a simple s-expression grammar.

    Has a symbol, a type signature, and a function that represents the fn represented.
    """

    @abstractmethod
    def symbol(self):
        pass

    @abstractmethod
    def type_signature(self) -> TypeSignature:
        """
        Return a TypeSignature object representing the type signature of this production
        """

    @abstractmethod
    def initialize(self, dsl) -> Dict[str, object]:
        """
        Return some state that this production might need to compute its function.
        E.g., for a neural network production, this might be the weights of the network.
        """

    @abstractmethod
    def compute_on_pytorch(self, dsl, state, *inputs):
        """
        Return the resulting pytorch expression of computing this function on the inputs
            Takes in the state of the production, which is the result of initialize().

        Effectively a form of denotation semantics.
        """

    @abstractmethod
    def render(self) -> str:
        """
        Render this production as a string.
        """


@dataclass
class ConcreteProduction(Production):
    _symbol: str
    _type_signature: TypeSignature
    _compute_on_pytorch: Callable[..., object]

    def symbol(self):
        return self._symbol

    def type_signature(self) -> TypeSignature:
        return self._type_signature

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {}

    def compute_on_pytorch(self, dsl, state, inputs):
        del dsl
        assert state == {}
        try:
            return self._compute_on_pytorch(*inputs)
        except TypeError:
            raise TypeError(
                f"Error computing {self._symbol} on inputs {inputs} with state {state}"
            )

    def render(self):
        return f"{self._symbol:>15} :: {self._type_signature.render()}"


@dataclass
class ParameterizedProduction(ConcreteProduction):
    _initialize: Dict[str, Callable[[], object]]

    def initialize(self, dsl) -> Dict[str, object]:
        del dsl
        return {k: v() for k, v in self._initialize.items()}

    def compute_on_pytorch(self, dsl, state, inputs):
        del dsl
        return self._compute_on_pytorch(*inputs, **state)

    def render(self):
        lhs = f"{self._symbol}[{', '.join(self._initialize)}]"
        return f"{lhs:>15} :: {self._type_signature.render()}"
