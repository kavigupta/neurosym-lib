from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from torch import ListType
from neurosym.types.type import ArrowType, Type, TypeVariable
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


def expansions(
    sig: Type,
    expand_to: List[Type],
    max_expansion_steps=np.inf,
    max_overall_depth=np.inf,
):
    assert isinstance(sig, Type)
    assert (
        min(max_expansion_steps, max_overall_depth) < np.inf
    ), "must specify either max_expansion_steps or max_overall_depth"

    if sig.depth() > max_overall_depth or max_expansion_steps == 0:
        return

    if not sig.has_type_vars():
        yield sig
        return
    ty_vars = sorted(sig.get_type_vars())
    for type_assignment in product(expand_to, repeat=len(ty_vars)):
        substitution = {}
        for ty_var, ty in zip(ty_vars, type_assignment):
            if ty.has_type_vars():
                # if the type to expand with itself has type vars like `[#a]`
                # then we replace these with a name unique to this outer `ty_var`
                # We do this replacement with another substitution
                inner_substitution = {}
                for inner_ty_var in sorted(ty.get_type_vars()):
                    fresh = ty_var + "_" + inner_ty_var
                    assert (
                        fresh not in ty_vars
                    ), f"fresh type variable name is already in use: {fresh}"
                    inner_substitution[inner_ty_var] = TypeVariable(fresh)
                ty = ty.subst_type_vars(inner_substitution)
            substitution[ty_var] = ty
        new_sig = sig.subst_type_vars(substitution)
        if new_sig.has_type_vars():
            # print("recursing at ", new_sig.depth())
            # print("recursing with", new_sig.render())
            yield from expansions(
                new_sig,
                expand_to,
                max_expansion_steps=max_expansion_steps - 1,
                max_overall_depth=max_overall_depth,
            )
        else:
            if new_sig.depth() <= max_overall_depth:
                yield new_sig


def type_universe(types: List[Type]):
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
    num_arrow_args = sorted(num_arrow_args)
    expand_to = []

    fresh_var = 0

    def fresh():
        nonlocal fresh_var
        fresh_var += 1
        return f"FRESH{fresh_var}"

    if has_list:
        expand_to.append(ListType(TypeVariable(fresh())))
    for n in num_arrow_args:
        args = [TypeVariable(fresh()) for _ in range(n)]
        expand_to.append(ArrowType(tuple(args), TypeVariable(fresh())))
    for t in atomic_types:
        expand_to.append(t)
    return expand_to


@dataclass
class LambdaTypeSignature(TypeSignature):
    """
    Represents the type signature of the (lamN ...) production.

    Does not implement the unify_return method, since that would
        require enumerating all the possible argument types.
    """
