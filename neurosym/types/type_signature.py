"""
Type signatures are used to represent the types of functions in the
DSL.
"""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Tuple, Union

import numpy as np
from frozendict import frozendict
from torch import NoneType

from neurosym.types.type import (
    ArrowType,
    ListType,
    Type,
    TypeVariable,
    UnificationError,
)
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment


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
        env = Environment.merge_all(*envs)
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
            Environment(frozendict({self.index_in_env: self.variable_type})),
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
    return sorted([t for t, _, _ in overall], key=str)


def _signature_expansions(
    sig: FunctionTypeSignature,
    terminals: List[Type],
    constructors: List[Tuple[int, Callable]],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
):
    """
    Returns a list of all possible expansions of the given type signature.

    Any type variables that appear in both the arguments and return type
        will be kept, while any other type variables will be expanded.
    """
    variables_in_arguments = {
        var for arg in sig.arguments for var in arg.max_depth_per_type_variable()
    }
    variables_in_return = set(sig.return_type.max_depth_per_type_variable())
    exclude_variables = variables_in_arguments & variables_in_return
    return type_expansions(
        sig.astype(),
        terminals,
        constructors,
        max_expansion_steps,
        max_overall_depth,
        exclude_variables,
    )


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


def _type_universe(types: List[Type], require_arity_up_to=None, no_zeroadic=False):
    """
    Produce a type universe from the given types.

    :param types: The types to use.
    :param require_arity_up_to: If specified, include all constructors up to this arity,
        even if they are not present in the types.
    :param no_zeroadic: If True, do not include zero-arity constructors.

    :return: A tuple of ``(atomic_types, constructors)``, where ``atomic_types``
        represents the atomic types in the universe, and ``constructors``
        is a list of tuples of the form ``(arity, constructor)``, where
        constructor is a function that takes ``arity`` types and
        produces a new type.
    """
    atomic_types = set()
    num_arrow_args = set()
    has_list = False
    for typ in types:
        for t in typ.walk_type_nodes():
            if t.is_atomic():
                atomic_types.add(t)
            if isinstance(t, ArrowType):
                num_arrow_args.add(len(t.input_type))
            if isinstance(t, ListType):
                has_list = True
    atomic_types = sorted(atomic_types, key=str)
    if require_arity_up_to is not None:
        num_arrow_args |= set(range(1, require_arity_up_to + 1))
    if no_zeroadic:
        num_arrow_args -= {0}
    num_arrow_args = sorted(num_arrow_args)

    constructors = []
    if has_list:
        constructors.append((1, ListType))
    for n in num_arrow_args:
        constructors.append(
            (
                n + 1,
                lambda *args: ArrowType(tuple(args[:-1]), args[-1]),
            )
        )
    return atomic_types, constructors
