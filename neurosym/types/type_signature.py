"""
Type signatures are used to represent the types of functions in the
DSL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from frozendict import frozendict
from torch import NoneType

from neurosym.types.type import ArrowType, Type, TypeVariable, UnificationError
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment


class TypeSignature(ABC):
    """
    Represents a type signature, which is a function converting back and
    forth between types (outputs) and lists of types (inputs).
    """

    @abstractmethod
    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        """
        Returns a list of types, one for each of the arguments, or None
        if the type cannot be unified.
        """

    @abstractmethod
    def return_type_template(self) -> Type:
        """
        Returns a template for the return type, with type variables.

        This can be an over-approximation of the actual return type,
        but it must not be an under-approximation.
        """

    @abstractmethod
    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
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

    def required_env_index(self) -> Union[int, None]:
        """
        Return the environment index this signature requires, or None if it
        does not depend on a specific environment slot. Used to prune
        productions that reference unreachable environment indices.
        """
        return None


@dataclass
class FunctionTypeSignature(TypeSignature):
    """
    Represents a concrete type signature, where the return type is known and the
    arguments are known. :py:class:`neurosym.TypeVariable` can be used
    in the arguments and return type, as long as any variable that appears
    in the arguments also appears in the return type (and vice versa).

    :param arguments: The types of the arguments.
    :param return_type: The type of the return value.
    """

    arguments: List[Type]
    return_type: Type

    @classmethod
    def from_type(cls, typ: Type) -> "FunctionTypeSignature":
        """
        Create a function type signature from a type, which must be an arrow type.
        """
        assert isinstance(typ, ArrowType)
        return cls(list(typ.input_type), typ.output_type)

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        try:
            mapping = twe.typ.unify(self.return_type)
        except UnificationError:
            return None
        return [
            TypeWithEnvironment(t.subst_type_vars(mapping), twe.env)
            for t in self.arguments
        ]

    def return_type_template(self) -> Type:
        return self.return_type

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        types = [x.typ for x in twes]
        envs = [x.env for x in twes]
        env = StrictEnvironment.merge_all(*envs)
        mapping = {}
        try:
            for t1, t2 in zip(self.arguments, types):
                for_arg = t1.unify(t2)
                for k, v in for_arg.items():
                    if k in mapping:
                        if mapping[k] != v:
                            return None
                    else:
                        mapping[k] = v
        except UnificationError:
            return None
        return TypeWithEnvironment(self.return_type.subst_type_vars(mapping), env)

    def arity(self) -> int:
        return len(self.arguments)

    def astype(self) -> Type:
        """
        Convert this type signature to an arrow type, which is the
        type of the function that this signature represents.
        """
        return ArrowType(tuple(self.arguments), self.return_type)

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return render_type(self.astype())


@dataclass
class LambdaTypeSignature(TypeSignature):
    """
    Represents the type signature of the lambda production. This is a production that
    matches any function type with the given input types, and returns the output type
    of the matching function type, but with with the input types placed in the environment.

    :param input_types: The types of the arguments to the lambda.
    """

    input_types: List[ArrowType]

    def arity(self) -> int:
        # just the body
        return 1

    def function_arity(self) -> int:
        """
        Get the arity of the function represented by the lambda,
        i.e., the number of arguments.
        """
        return len(self.input_types)

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        body = TypeVariable("body")
        input_types = ";".join(render_type(x) for x in self.input_types)
        lambda_type = f"L<{render_type(body)}|{input_types}>"

        return f"{lambda_type} -> {render_type(ArrowType(self.input_types, body))}"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if not isinstance(twe.typ, ArrowType):
            return None
        if twe.typ.input_type != self.input_types:
            return None
        return [
            TypeWithEnvironment(
                twe.typ.output_type,
                twe.env.child(*self.input_types),
            )
        ]

    def return_type_template(self) -> Type:
        return ArrowType(self.input_types, TypeVariable("body"))

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 1:
            return None
        parent = twes[0].env.parent(self.input_types)
        return TypeWithEnvironment(ArrowType(self.input_types, twes[0].typ), parent)


@dataclass
class VariableTypeSignature(TypeSignature):
    """
    Represents the type signature of a variable production.

    This is a type signature where the return type is known, but the
    environment must contain the given type at the given index in
    order to be valid.

    :param variable_type: The type of the variable.
    :param index_in_env: The index of the variable in the environment.
    """

    variable_type: Type
    index_in_env: int

    def required_env_index(self) -> Union[int, None]:
        return self.index_in_env

    def arity(self) -> int:
        # leaf
        return 0

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return f"V<{render_type(self.variable_type)}@{self.index_in_env}>"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if twe.typ != self.variable_type:
            return None
        if not twe.env.contains_type_at(self.variable_type, self.index_in_env):
            return None
        return []

    def return_type_template(self) -> Type:
        return self.variable_type

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 0:
            return None
        return TypeWithEnvironment(
            self.variable_type,
            StrictEnvironment(frozendict({self.index_in_env: self.variable_type})),
        )
