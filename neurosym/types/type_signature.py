"""
Type signatures are used to represent the types of functions in the
DSL.
"""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Union

from frozendict import frozendict
from torch import NoneType

from neurosym.types.type import (
    ArrowType,
    GenericTypeVariable,
    Type,
    TypeVariable,
    UnificationError,
)
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment

_fresh_type_variable_counter = itertools.count()


def _instantiate_type_scheme(types: List[Type]) -> List[Type]:
    """Rename every type variable in ``types`` to a globally-fresh name,
    consistently across all the given types.

    This implements let-polymorphism: each use of a polymorphic production is an
    independent instance of its type scheme, so its variables must not be
    conflated with identically-named variables coming from sibling subterms. For
    example, ``index :: (i, [#T]) -> #T`` and ``++ :: ([#T], [#T]) -> [#T]`` each
    declare a variable literally named ``#T``, but the two are unrelated. Without
    fresh instantiation, composing them (``(++ (index 0 empty) empty)``) forces
    the single shared ``#T`` into contradictory bindings and unification wrongly
    fails.

    Variable subclasses (e.g. filtered variables) and their attributes are
    preserved.
    """
    subst = {}
    for typ in types:
        for node in typ.walk_type_nodes():
            if isinstance(node, GenericTypeVariable) and node.name not in subst:
                fresh_name = f"_fresh_{next(_fresh_type_variable_counter)}"
                subst[node.name] = replace(node, name=fresh_name)
    if not subst:
        return list(types)
    return [typ.subst_type_vars(subst) for typ in types]


def _unify_sequentially(pairs) -> dict:
    """Unify a sequence of ``(expected, actual)`` type pairs, threading a single
    substitution through and composing it left to right.

    Applying the accumulated substitution to each pair before unifying it keeps
    the accumulated substitution's domain disjoint from each new binding's
    domain, so composition reduces to "apply the new bindings to the old ones,
    then take the union". This is what lets a variable that is constrained by two
    different arguments (e.g. both arguments of ``++``) be unified transitively,
    rather than reported as a spurious conflict.

    :raises UnificationError: if the pairs cannot be unified (including a failed
        occurs check).
    """
    subst = {}
    for expected, actual in pairs:
        new = expected.subst_type_vars(subst).unify(actual.subst_type_vars(subst))
        for name, value in new.items():
            if name in value.get_type_vars() and not (
                isinstance(value, GenericTypeVariable) and value.name == name
            ):
                raise UnificationError(f"occurs check failed: {name} in {value}")
        subst = {name: value.subst_type_vars(new) for name, value in subst.items()}
        subst.update(new)
    return subst


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
        # Instantiate this signature's type scheme with fresh variables so its
        # polymorphic variables are independent of identically-named variables in
        # the argument types (see _instantiate_type_scheme).
        *arguments, return_type = _instantiate_type_scheme(
            list(self.arguments) + [self.return_type]
        )
        try:
            mapping = _unify_sequentially(zip(arguments, types))
        except UnificationError:
            return None
        # Apply substitution to each child env before merging, so that
        # polymorphic variable types get resolved consistently.
        if mapping:
            envs = [env.subst_type_vars(mapping) for env in envs]
        env = StrictEnvironment.merge_all(*envs)
        return TypeWithEnvironment(return_type.subst_type_vars(mapping), env)

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
        # Extract top arity slots, filling missing (unused) slots with
        # type variable placeholders.
        top_types, parent = twes[0].env.unwind_top(self.function_arity)
        if top_types is None:
            return None
        input_types = tuple(
            t if t is not None else TypeVariable(f"__lam_unused_{i}")
            for i, t in enumerate(top_types)
        )
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
            if twe.env.has_index(self.index_in_env):
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
