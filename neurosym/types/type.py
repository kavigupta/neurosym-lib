import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np


class Type(ABC):
    def node_summary(self):
        """
        Return a summary of this node, excluding children. This is useful
        for caching in certain contexts, such as the TreeTrie.
        """
        summary_dict = {
            "_type": self.__class__.__name__,
            **self.inherent_parameters(),
        }
        summary_dict = dict(sorted(summary_dict.items()))
        return str(summary_dict)

    @abstractmethod
    def inherent_parameters(self):
        """
        Return a list of inherent parameters of the type, i.e., all parameters not associated
        with the children.
        """
        raise NotImplementedError

    @abstractmethod
    def children(self):
        """
        Iterate over direct children and yield each.
        """
        raise NotImplementedError

    @abstractmethod
    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        """
        Unify this type with another type. Returns a dictionary of substitutions.

        :param other: the other type to unify with.
        :param already_tried_other_direction: then we have already tried to unify
            the other direction and failed, so we should not try again.

        :raises UnificationError: if the types cannot be unified.
        """
        raise NotImplementedError

    @abstractmethod
    def subst_type_vars(self, subst: Dict[str, "Type"]):
        """
        Substitute type variables in the type according to the substitution.
        """
        raise NotImplementedError

    def is_atomic(self):
        """
        Return True if the type is atomic.
        """
        return len(list(self.walk_type_nodes())) == 1 and not isinstance(
            self, TypeVariable
        )

    def depth_of_type_variables(self) -> List[Tuple[str, float]]:
        """
        Return a list of tuples of the form (type_variable, depth) where depth is the
        depth of the type variable in the type tree.
        """
        node_height = np.log2(1 + len(list(self.children())))
        return [
            (tv, depth + node_height)
            for child in self.children()
            for tv, depth in child.depth_of_type_variables()
        ]

    def max_depth_per_type_variable(self) -> Dict[str, float]:
        """
        Return a dictionary mapping type variables to their maximum depth in the type
        tree.
        """
        result = {}
        for tv, depth in self.depth_of_type_variables():
            result[tv] = max(result.get(tv, 0), depth)
        return result

    def get_type_vars(self):
        """
        Return a set of all type variables in the type.
        """
        return sorted({tv for tv, _ in self.depth_of_type_variables()})

    def has_type_vars(self):
        """
        Return True if the type has type variables.
        """
        return len(self.get_type_vars()) > 0

    @cached_property
    def depth(self):
        """
        Return the depth of the type tree.
        """
        children = [child.depth for child in self.children()]
        depth = max(children + [0]) + np.log2(len(children) + 1)
        return depth

    def walk_type_nodes(self) -> Iterator["Type"]:
        """
        Walk the type tree and yield all children nodes.
        """
        yield self
        for child in self.children():
            yield from child.walk_type_nodes()


class UnificationError(Exception):
    """
    Raised when two types cannot be unified.
    """


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    """
    An atomic type is a type that cannot be further decomposed. It is a leaf node in the
    type tree. Examples of atomic types are integers, floats, and strings. These types
    are rendered as strings of their name, e.g., ``AtomicType("f")`` is rendered as ``"f"``.

    :field name: the name of the atomic type.
    """

    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def inherent_parameters(self):
        return dict(name=self.name)

    def children(self):
        yield from []

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, AtomicType):
            if self.name != other.name:
                raise UnificationError(f"{self} != {other}")
            return {}
        if already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        return other.unify(self, already_tried_other_direction=True)

    def subst_type_vars(self, subst: Dict[str, Type]):
        return self


@dataclass(frozen=True, eq=True)
class TensorType(Type):
    """
    A tensor is a type that represents a tensor of a given shape. These types are rendered
    as strings of the form ``{dtype, *shape}``, e.g., ``TensorType(ns.AtomicType("f"), (3, 4))`` is
    rendered as ``"{f, 3, 4}"``.

    :field dtype: the data type of the tensor (usually ``ns.AtomicType("f")``).
    :field shape: the shape of the tensor.
    """

    dtype: Type
    shape: Tuple[int]

    def inherent_parameters(self):
        return dict(dtype=self.dtype, shape=self.shape)

    def children(self):
        yield from []

    def subst_type_vars(self, subst: Dict[str, Type]):
        del subst
        assert not self.has_type_vars()
        return self

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, TensorType):
            if self.shape != other.shape:
                raise UnificationError(f"{self} != {other}")
            return self.dtype.unify(other.dtype)
        if already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        return other.unify(self, already_tried_other_direction=True)


@dataclass(frozen=True, eq=True)
class ListType(Type):
    """
    Represents a list of elements of a given type. These types are rendered as strings of
    the form ``[element_type]``, e.g., ``ListType(ns.AtomicType("f"))`` is rendered as ``"[f]"``.

    :field element_type: the type of the elements in the list.
    """

    element_type: Type

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield self.element_type

    def subst_type_vars(self, subst: Dict[str, Type]):
        return ListType(self.element_type.subst_type_vars(subst))

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if isinstance(other, ListType):
            return self.element_type.unify(other.element_type)
        if already_tried_other_direction:
            raise UnificationError(f"{self} != {other}")
        return other.unify(self, already_tried_other_direction=True)


@dataclass(frozen=True, eq=True)
class ArrowType(Type):
    """
    An arrow type represents a function type. It is a type that takes a tuple of input
    types and returns an output type. These types are rendered as strings of the form
    ``(input_type1, input_type2, ...) -> (output_type)``, e.g.,
    ``ArrowType((ns.AtomicType("f"), ns.AtomicType("f")), ns.AtomicType("f"))`` is rendered
    as ``"(f, f) -> f"``.

    :field input_type: a tuple of input types.
    :field output_type: the output type.
    """

    input_type: Tuple[Type]
    output_type: Type

    def __post_init__(self):
        assert isinstance(self.input_type, tuple), "input_type must be a tuple"

    def subst_type_vars(self, subst: Dict[str, Type]):
        return ArrowType(
            tuple(t.subst_type_vars(subst) for t in self.input_type),
            self.output_type.subst_type_vars(subst),
        )

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield from self.input_type
        yield self.output_type

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if not isinstance(other, ArrowType):
            if already_tried_other_direction:
                raise UnificationError(f"{self} != {other}")
            return other.unify(self, already_tried_other_direction=True)
        if len(self.input_type) != len(other.input_type):
            raise UnificationError(f"{self} != {other}")
        individuals = []
        for t1, t2 in zip(self.input_type, other.input_type):
            individuals.append(t1.unify(t2))
        individuals.append(self.output_type.unify(other.output_type))
        subst = {}
        for individual in individuals:
            for k, v in individual.items():
                if k in subst:
                    if subst[k] != v:
                        raise UnificationError(f"{k} is unified to {subst[k]} and {v}")
                else:
                    subst[k] = v
        return subst


@dataclass(frozen=True, eq=True)
class GenericTypeVariable(Type):
    """
    Base class for type variables. See ``TypeVariable`` and ``FilteredTypeVariable`` for
    concrete implementations and documentation
    """

    name: str

    @classmethod
    def fresh(cls):
        return cls(f"_{uuid.uuid4().hex}")

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def depth_of_type_variables(self) -> List[Tuple[str, int]]:
        return [(self.name, 0)]

    def subst_type_vars(self, subst: Dict[str, Type]):
        return subst.get(self.name, self)

    def inherent_parameters(self):
        return dict()

    def children(self):
        yield from []

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        return {self.name: other}


class TypeVariable(GenericTypeVariable):
    """
    A type variable is a type that can be unified with any other type, but must be
    unified with the same type throughout the type tree. Type variables are used to
    represent generic types in the type system. These types are rendered as strings of
    the form ``#name``, e.g., ``TypeVariable("x")`` is rendered as ``#x``.

    :field name: the name of the type variable.
    """


@dataclass(frozen=True, eq=True)
class FilteredTypeVariable(GenericTypeVariable):
    """
    A filtered type variable is a type variable (see ``TypeVariable``) that can only be
    unified with types that satisfy a given predicate. These types are rendered as
    strings of the form ``%name``, e.g., ``FilteredTypeVariable("x", lambda t: isinstance(t, ns.AtomicType))``
    is rendered as ``%x``.

    :field name: the name of the type variable.
    :field type_filter: a predicate that takes a type and returns True if the type can be unifie
        with the type variable.
    """

    type_filter: Callable[[Type], bool]

    def unify(
        self, other: "Type", already_tried_other_direction=False
    ) -> Dict[str, "Type"]:
        if self.type_filter(other):
            return {self.name: other}
        raise UnificationError(f"{self} does not match {other}")


float_t = AtomicType("f")
