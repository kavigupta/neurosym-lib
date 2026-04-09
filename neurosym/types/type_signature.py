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
        # Apply substitution to each child env before merging, so that
        # polymorphic variable types get resolved consistently.
        if mapping:
            envs = [env.subst_type_vars(mapping) for env in envs]
        env = StrictEnvironment.merge_all(*envs)
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
    Represents the type signature of the lambda production. This is a polymorphic
    production that matches any function type with the given arity, and returns
    the output type of the matching function type, with the input types placed
    in the environment.

    :param function_arity: The number of arguments to the lambda.
    """

    function_arity: int

    def arity(self) -> int:
        # just the body
        return 1

    def _type_vars(self):
        return tuple(TypeVariable(f"__lam_{i}") for i in range(self.function_arity))

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        body = TypeVariable("body")
        tvs = self._type_vars()
        input_types = ";".join(render_type(x) for x in tvs)
        lambda_type = f"L<{render_type(body)}|{input_types}>"

        return f"{lambda_type} -> {render_type(ArrowType(tvs, body))}"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if not isinstance(twe.typ, ArrowType):
            return None
        if len(twe.typ.input_type) != self.function_arity:
            return None
        return [
            TypeWithEnvironment(
                twe.typ.output_type,
                twe.env.child(*twe.typ.input_type),
            )
        ]

    def return_type_template(self) -> Type:
        return ArrowType(self._type_vars(), TypeVariable("body"))

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 1:
            return None
        env = twes[0].env
        # First try the fast path: all lambda arg slots are present.
        parent_types = env.parent_types(self.function_arity)
        if parent_types is not None:
            parent = env.parent(parent_types)
            return TypeWithEnvironment(ArrowType(parent_types, twes[0].typ), parent)
        # Slow path: some lambda arg slots are unused. Fill with type variables
        # and strip the lambda entries from the environment manually.
        if not isinstance(env, StrictEnvironment):
            return None
        # pylint: disable=protected-access
        input_types = tuple(
            env._elements.get(i, TypeVariable(f"__lam_unused_{i}"))
            for i in range(self.function_arity)
        )
        # Build parent env: shift indices down by function_arity, dropping 0..arity-1
        parent_elements = {}
        for idx, typ in env._elements.items():
            if idx >= self.function_arity:
                parent_elements[idx - self.function_arity] = typ
        parent = StrictEnvironment(frozendict(parent_elements))
        return TypeWithEnvironment(ArrowType(input_types, twes[0].typ), parent)


@dataclass
class VariableTypeSignature(TypeSignature):
    """
    Represents the type signature of a variable production.

    This is a polymorphic type signature that matches any type present
    at the given index in the environment.

    :param index_in_env: The index of the variable in the environment.
    """

    index_in_env: int

    def required_env_index(self) -> Union[int, None]:
        return self.index_in_env

    def arity(self) -> int:
        # leaf
        return 0

    def render(self) -> str:
        return f"V<${self.index_in_env}>"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if isinstance(twe.typ, TypeVariable):
            # A type variable can be any type; match if the index exists
            if (
                isinstance(twe.env, StrictEnvironment)
                and self.index_in_env
                in twe.env._elements  # pylint: disable=protected-access
            ):
                return []
            # In a permissive environment, type variables always match
            if not isinstance(twe.env, StrictEnvironment):
                return []
            return None
        if not twe.env.contains_type_at(twe.typ, self.index_in_env):
            return None
        return []

    def return_type_template(self) -> Type:
        return TypeVariable(f"__var_{self.index_in_env}")

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 0:
            return None
        typ = TypeVariable(f"__var_{self.index_in_env}")
        return TypeWithEnvironment(
            typ,
            StrictEnvironment(frozendict({self.index_in_env: typ})),
        )
