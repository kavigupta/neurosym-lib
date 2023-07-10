from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from ..types.type_signature import TypeSignature


class Production(ABC):
    """
    Represents a production rule in a simple s-expression grammar.

    Has a symbol, a type signature, and a function that represents the function represented.
    """

    @abstractmethod
    def symbol(self):
        pass

    @abstractmethod
    def type_signature(self) -> TypeSignature:
        """
        Return a TypeSignature object representing the type signature of this production.
        """

    @abstractmethod
    def compute_on_pytorch(self, *inputs):
        """
        Return the resulting pytorch expression of computing this function on the inputs.

        Effectively a form of denotation semantics.
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

    def compute_on_pytorch(self, *inputs):
        return self._compute_on_pytorch(*inputs)

class ParameterizedProduction(Production):
    pass