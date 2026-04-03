"""
Type signatures are used to represent the types of functions in the
DSL.
"""

import itertools
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from frozendict import frozendict
from torch import NoneType

from neurosym.types.type import (
    ArrowType,
    FilteredTypeVariable,
    Type,
    TypeVariable,
    UnificationError,
)
from neurosym.types.type_with_environment import StrictEnvironment, TypeWithEnvironment
from neurosym.utils.documentation import internal_only


@internal_only
def resolve_type(typ: Type, bindings: Dict[str, Type]) -> Type:
    """
    Transitively resolve type variables using the given bindings.

    Follows chains like ``_w1 -> _f1 -> int`` until a fixed point is reached
    (either a concrete type or an unbound variable).
    """
    for _ in range(100):
        resolved = typ.subst_type_vars(bindings)
        if resolved == typ:
            return typ
        typ = resolved
    return typ


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

    @internal_only
    def alpha_rename(self) -> Tuple["FunctionTypeSignature", Dict[str, str]]:
        """
        Return a copy of this signature with all type variables replaced by
        fresh names. Preserves FilteredTypeVariable filters.

        Returns ``(fresh_signature, rename_map)`` where ``rename_map`` maps
        old variable names to new variable names.
        """
        all_vars = {}
        for arg in self.arguments:
            all_vars.update(arg.collect_type_var_objects())
        all_vars.update(self.return_type.collect_type_var_objects())

        if not all_vars:
            return self, {}

        rename_map = {}
        subst = {}
        for name, var_obj in all_vars.items():
            fresh_name = f"_{uuid.uuid4().hex}"
            rename_map[name] = fresh_name
            if isinstance(var_obj, FilteredTypeVariable):
                subst[name] = FilteredTypeVariable(fresh_name, var_obj.type_filter)
            else:
                subst[name] = TypeVariable(fresh_name)

        fresh_args = [arg.subst_type_vars(subst) for arg in self.arguments]
        fresh_return = self.return_type.subst_type_vars(subst)
        return FunctionTypeSignature(fresh_args, fresh_return), rename_map

    def render(self) -> str:
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return render_type(self.astype())


@dataclass
class LambdaTypeSignature(TypeSignature):
    """
    Represents the type signature of a lambda production. Matches any function
    type (ArrowType) with the given number of input arguments. The actual input
    types are extracted from the query type at unification time.

    :param num_args: The number of arguments the lambda takes.
    """

    num_args: int

    def arity(self) -> int:
        # just the body
        return 1

    def function_arity(self) -> int:
        """
        Get the arity of the function represented by the lambda,
        i.e., the number of arguments.
        """
        return self.num_args

    def render(self) -> str:
        args = ";".join(f"#_arg{i}" for i in range(self.num_args))
        return f"L<#body|{args}> -> ({args}) -> #body"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if not isinstance(twe.typ, ArrowType):
            return None
        if len(twe.typ.input_type) != self.num_args:
            return None
        return [
            TypeWithEnvironment(
                twe.typ.output_type,
                twe.env.child(*twe.typ.input_type),
            )
        ]

    def return_type_template(self) -> Type:
        # Wildcard: matches any ArrowType. The arity check in unify_return
        # handles filtering.
        return TypeVariable("_lambda_return")

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 1:
            return None
        # Reconstruct the arrow type from the body type and environment.
        # child(*input_types) inserts in order, each at index 0, so
        # the last type ends up at index 0. We read them back in reverse
        # index order to recover the original input_types ordering.
        env = twes[0].env
        if not isinstance(env, StrictEnvironment):
            return None
        # pylint: disable=protected-access
        input_types = []
        for i in reversed(range(self.num_args)):
            input_types.append(
                env._elements.get(i, TypeVariable(f"_lam_arg{i}"))
            )
        parent = env.parent(tuple(input_types))
        return TypeWithEnvironment(ArrowType(tuple(input_types), twes[0].typ), parent)


@dataclass
class VariableTypeSignature(TypeSignature):
    """
    Represents the type signature of a variable production.

    Matches any type that is present in the environment at the given index.

    :param index_in_env: The index of the variable in the environment.
    """

    index_in_env: int

    def arity(self) -> int:
        # leaf
        return 0

    def render(self) -> str:
        return f"V<@{self.index_in_env}>"

    def unify_return(
        self, twe: TypeWithEnvironment
    ) -> Union[List[TypeWithEnvironment], NoneType]:
        if not isinstance(twe.env, StrictEnvironment):
            # PermissiveEnvironment: always matches
            return []
        # pylint: disable=protected-access
        if self.index_in_env not in twe.env._elements:
            return None
        env_type = twe.env._elements[self.index_in_env]
        try:
            twe.typ.unify(env_type)
        except UnificationError:
            return None
        return []

    def return_type_template(self) -> Type:
        # Wildcard: matches any type. The environment check in unify_return
        # handles filtering.
        return TypeVariable("_var_return")

    def unify_arguments(
        self, twes: List[TypeWithEnvironment]
    ) -> Union[TypeWithEnvironment, NoneType]:
        if len(twes) != 0:
            return None
        # Without a fixed type, return a type variable. The caller must
        # resolve it from context (e.g., the environment).
        var = TypeVariable(f"_var{self.index_in_env}")
        return TypeWithEnvironment(
            var,
            StrictEnvironment(frozendict({self.index_in_env: var})),
        )


def bottom_up_enumerate_types(
    terminals: List[Type],
    constructors: List[Tuple[int, Callable]],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
):
    """
    Returns a list of all possible expansions of the given terminals
    using the given constructors.

    :param terminals: The atomic types to use.
    :param constructors: The constructors to use.
    :param max_expansion_steps: The maximum number of times to apply a constructor.
    :param max_overall_depth: The maximum depth of the resulting types.
    """
    assert (
        min(max_expansion_steps, max_overall_depth) < np.inf
    ), "must specify either max_expansion_steps or max_overall_depth"
    current_with_depth = [(t, t.depth, 0) for t in terminals]
    current_with_depth = [
        (t, d, n)
        for t, d, n in current_with_depth
        if d <= max_overall_depth and n <= max_expansion_steps
    ]
    overall = set()
    while True:
        overall.update(current_with_depth)
        current_with_depth = []
        for arity, fn in constructors:
            additional_depth = np.log2(arity + 1)
            will_work = [
                (t, d, n)
                for t, d, n in overall
                if d + additional_depth <= max_overall_depth
                and n + 1 <= max_expansion_steps
            ]
            for subentities in itertools.product(will_work, repeat=arity):
                types = [t for t, _, _ in subentities]
                depth = max((d for _, d, _ in subentities)) + additional_depth
                steps = max((n for _, _, n in subentities)) + 1
                assert depth <= max_overall_depth
                assert steps <= max_expansion_steps
                res = (fn(*types), depth, steps)
                if res in overall:
                    continue
                current_with_depth.append(res)
        if len(current_with_depth) == 0:
            break
    return sorted([t for t, _, _ in overall if t.depth < max_overall_depth], key=str)


def type_expansions(
    sig: Type,
    terminals: List[Type],
    constructors: List[Tuple[int, Callable]],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
    exclude_variables=(),
):
    """
    Returns a list of all possible expansions of the given type, where the
    variables in exclude_variables are not expanded.

    This is useful for expanding a type signature, where some of the type
    variables should not be expanded.

    :param sig: The type to expand.
    :param terminals: The atomic types to use.
    :param constructors: The constructors to use.
    :param max_expansion_steps: The maximum number of times to apply a constructor.
    :param max_overall_depth: The maximum depth of the resulting types.
    :param exclude_variables: The type variables to exclude from expansion.

    :return: A list of all possible expansions of the given type.
    """
    # pylint: disable=cyclic-import
    from neurosym.types.type_string_repr import render_type

    depth_by_var = sig.max_depth_per_type_variable()
    for var in exclude_variables:
        if var not in depth_by_var:
            raise ValueError(
                f"Variable {var} not in type signature {render_type(sig)}, "
                + f"cannot exclude. Valid options: {sorted(depth_by_var)}"
            )
        del depth_by_var[var]
    enumerations_by_var = {
        ty_var: bottom_up_enumerate_types(
            terminals,
            constructors,
            max_expansion_steps,
            max_overall_depth - depth_by_var[ty_var],
        )
        for ty_var in depth_by_var.keys()
    }
    variables = list(depth_by_var.keys())
    enumerations = [enumerations_by_var[ty_var] for ty_var in variables]
    for types in product(*enumerations):
        remap = dict(zip(variables, types))
        yield sig.subst_type_vars(remap)


def _all_available_types(types: List[Type]):
    available_types = set()
    for typ in types:
        for t in typ.walk_type_nodes():
            available_types.add(t)
    return sorted(available_types, key=str)


def _type_universe(types: List[Type], no_zeroadic=False):
    """
    Produce a type universe from the given types.

    :param types: The types to use.
    :param no_zeroadic: If True, do not include zero-arity constructors.

    :return: A tuple of ``(atomic_types, constructors)``, where ``atomic_types``
        represents the atomic types in the universe, and ``constructors``
        is a list of tuples of the form ``(arity, constructor)``, where
        constructor is a function that takes ``arity`` types and
        produces a new type.
    """
    available_types = _all_available_types(types)

    constructors = []
    for t in available_types:
        if isinstance(t, TypeVariable):
            continue
        if no_zeroadic and isinstance(t, ArrowType) and len(t.input_type) == 0:
            continue
        tv = t.get_type_vars()
        constructors.append(
            (len(tv), lambda *args, t=t, tv=tv: t.subst_type_vars(dict(zip(tv, args))))
        )

    return [c() for count, c in constructors if count == 0], [
        (count, c) for count, c in constructors if count > 0
    ]
