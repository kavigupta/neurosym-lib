from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from neurosym.types.type import ArrowType, Type
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment


class TypeSignature(ABC):
    """
    Represents a type signature, which is a function converting back and
        forth between types (outputs) and lists of types (inputs).
    """

    @abstractmethod
    def unify_return(self, type: TypeWithEnvironment) -> List[TypeWithEnvironment]:
        """
        Returns a list of types, one for each of the arguments, or None
        if the type cannot be unified.
        """

    @abstractmethod
    def unify_arguments(self, twes: List[TypeWithEnvironment]) -> TypeWithEnvironment:
        """
        Returns the return type of the function, or None if the types
        cannot be unified.
        """

    @abstractmethod
    def arity(self) -> int:
        """
        Returns the arity of the function, i.e., the number of arguments.
        """

    @abstractmethod
    def render(self) -> str:
        """
        Render this type signature as a string.
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

    @classmethod
    def from_type(cls, type: Type) -> "ConcreteTypeSignature":
        assert isinstance(type, ArrowType)
        return cls(list(type.input_type), type.output_type)

    def unify_return(self, twe: TypeWithEnvironment) -> List[TypeWithEnvironment]:
        if twe.typ == self.return_type:
            return [TypeWithEnvironment(t, twe.env) for t in self.arguments]
        else:
            return None

    def unify_arguments(self, twes: List[TypeWithEnvironment]) -> TypeWithEnvironment:
        types = [x.typ for x in twes]
        envs = [x.env for x in twes]
        env = envs[0] if envs else Environment.empty()
        assert all(envs[0] == env for env in envs)
        if list(types) == list(self.arguments):
            return TypeWithEnvironment(self.return_type, env)
        else:
            return None

    def arity(self) -> int:
        return len(self.arguments)

    def astype(self) -> Type:
        return ArrowType(tuple(self.arguments), self.return_type)

    def render(self) -> str:
        from neurosym.types.type_string_repr import render_type

        return render_type(self.astype())
