from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from neurosym.types.type import Type


class TypeSignature(ABC):
    """
    Wrapper around DreamCoder's Implementation of Hindley-Milner Type Inference
    """

    @abstractmethod
    def unify_return(self, type: Type) -> List[List[Type]]:
        """
        Returns a list of lists of types, where each list of types is a possible
        expansion for the given type signature.
        """


class HindleyMilnerTypeSignature(TypeSignature):
    """
    Wrapper around DreamCoder's Implementation of Hindley-Milner Type Inference
    """

    def unify_return(self, type: Type) -> List[List[Type]]:
        # TODO(MB) implement this
        raise NotImplementedError


@dataclass
class ConcreteTypeSignature(TypeSignature):
    """
    Represents a concrete type signature, where the return type is known and the
    arguments are known.
    """
    arguments: List[Type]
    return_type: Type

    def unify_return(self, type: Type) -> List[List[Type]]:
        if type == self.return_type:
            return [self.arguments]
        else:
            return []
