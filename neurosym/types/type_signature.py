from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from neurosym.types.type import Type


class TypeSignature(ABC):
    """
    Represents a type signature, which is a function converting back and
        forth between types (outputs) and lists of types (inputs).
    """

    @abstractmethod
    def unify_return(self, type: Type) -> List[Type]:
        """
        Returns a list of types, one for each of the arguments, or None
        if the type cannot be unified.
        """

    @abstractmethod
    def unify_arguments(self, types: List[Type]) -> Type:
        """
        Returns the return type of the function, or None if the types
        cannot be unified.
        """

    @abstractmethod
    def arity(self) -> int:
        """
        Returns the arity of the function, i.e., the number of arguments.
        """


def expanded_type_signature(type_signature: TypeSignature) -> TypeSignature:
    """
    Expands the type signature into a concrete type signature.
    """
    # TODO (MB) implement this
    raise NotImplementedError


@dataclass
class ConcreteTypeSignature(TypeSignature):
    """
    Represents a concrete type signature, where the return type is known and the
    arguments are known.
    """

    arguments: List[Type]
    return_type: Type

    def unify_return(self, type: Type) -> List[Type]:
        if type == self.return_type:
            return self.arguments
        else:
            return None

    def unify_arguments(self, types: List[Type]) -> Type:
        if list(types) == list(self.arguments):
            return self.return_type
        else:
            return None

    def arity(self) -> int:
        return len(self.arguments)
