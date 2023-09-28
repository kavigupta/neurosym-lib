from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
from typing import Callable, List, Tuple
import frozendict

from neurosym.types.type import (
    ArrowType,
    Type,
    ListType,
    TypeVariable,
    UnificationError,
)
from neurosym.types.type_with_environment import Environment, TypeWithEnvironment
from itertools import product
import numpy as np


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
        try:
            mapping = twe.typ.unify(self.return_type)
        except UnificationError:
            return None
        return [
            TypeWithEnvironment(t.subst_type_vars(mapping), twe.env)
            for t in self.arguments
        ]

    def unify_arguments(self, twes: List[TypeWithEnvironment]) -> TypeWithEnvironment:
        types = [x.typ for x in twes]
        envs = [x.env for x in twes]
        env = envs[0] if envs else Environment.empty()
        assert all(envs[0] == env for env in envs)
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
        return ArrowType(tuple(self.arguments), self.return_type)

    def render(self) -> str:
        from neurosym.types.type_string_repr import render_type

        return render_type(self.astype())


@dataclass
class LambdaTypeSignature(TypeSignature):
    """
    Represents the type signature of the lambda production.
    """

    input_types: List[ArrowType]

    def arity(self) -> int:
        # just the body
        return 1

    def function_arity(self) -> int:
        return len(self.input_types)

    def render(self) -> str:
        from neurosym.types.type_string_repr import render_type

        body = TypeVariable("body")
        input_types = ";".join(render_type(x) for x in self.input_types)
        lambda_type = f"L<{render_type(body)}|{input_types}>"

        return f"{lambda_type} -> {render_type(ArrowType(self.input_types, body))}"

    def unify_return(self, twe: TypeWithEnvironment) -> List[TypeWithEnvironment]:
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

    def unify_arguments(self, twes: List[TypeWithEnvironment]) -> TypeWithEnvironment:
        if len(twes) != 1:
            return None
        parent = twes[0].env.parent(*self.input_type)
        return TypeWithEnvironment(ArrowType(self.input_types, twes[0].typ), parent)


@dataclass
class VariableTypeSignature(TypeSignature):
    """
    Represents the type signature of variable production.
    """

    variable_type: Type
    index_in_env: int

    def arity(self) -> int:
        # leaf
        return 0

    def render(self) -> str:
        from neurosym.types.type_string_repr import render_type

        return f"V<{render_type(self.variable_type)}@{self.index_in_env}>"

    def unify_return(self, twe: TypeWithEnvironment) -> List[TypeWithEnvironment]:
        if twe.typ != self.variable_type:
            return None
        if not twe.env.contains_type_at(self.variable_type, self.index_in_env):
            return None
        return []

    def unify_arguments(self, twes: List[TypeWithEnvironment]) -> TypeWithEnvironment:
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
    """
    assert (
        min(max_expansion_steps, max_overall_depth) < np.inf
    ), "must specify either max_expansion_steps or max_overall_depth"
    current_with_depth = [(t, t.depth(), 0) for t in terminals]
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
                depth = max([d for _, d, _ in subentities]) + additional_depth
                steps = max([n for _, _, n in subentities]) + 1
                assert depth <= max_overall_depth
                assert steps <= max_expansion_steps
                new_type = fn(*types)
                res = (new_type, depth, steps)
                if res in overall:
                    continue
                current_with_depth.append(res)
        if len(current_with_depth) == 0:
            break
    return sorted([t for t, _, _ in overall], key=str)


def signature_expansions(
    sig: ConcreteTypeSignature,
    terminals: List[Type],
    constructors: List[Tuple[int, Callable]],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
):
    variables_in_arguments = {
        var for arg in sig.arguments for var in arg.max_depth_per_type_variable()
    }
    variables_in_return = set(sig.return_type.max_depth_per_type_variable())
    exclude_variables = variables_in_arguments & variables_in_return
    return expansions(
        sig.astype(),
        terminals,
        constructors,
        max_expansion_steps,
        max_overall_depth,
        exclude_variables,
    )


def expansions(
    sig: Type,
    terminals: List[Type],
    constructors: List[Tuple[int, Callable]],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
    exclude_variables=(),
):
    from neurosym.types.type_string_repr import render_type

    depth_by_var = sig.max_depth_per_type_variable()
    for var in exclude_variables:
        if var not in depth_by_var:
            raise ValueError(
                f"Variable {var} not in type signature {render_type(sig)}, cannot exclude. Valid options: {sorted(depth_by_var)}"
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


def type_universe(types: List[Type], require_arity_up_to=None):
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
