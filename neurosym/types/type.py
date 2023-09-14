from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


class Type(ABC):
    @abstractmethod
    def walk_type_nodes(self):
        """
        Walk the type tree and yield all nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def children(self):
        """
        Iterate over direct children and yield each.
        """
        raise NotImplementedError

    def is_atomic(self):
        """
        Return True if the type is atomic.
        """
        return len(list(self.walk_type_nodes())) == 1 and not isinstance(
            self, TypeVariable
        )

    def depth(self):
        """
        Return the depth of the type tree.
        """
        children = [child.depth() for child in self.children()]
        return max(children + [0]) + np.log2(len(children) + 1)


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def walk_type_nodes(self):
        yield self

    def children(self):
        yield from []

    def has_type_vars(self):
        return False

    def get_type_vars(self):
        return set()

    def subst_type_vars(self, subst: Dict[str, Type]):
        return self


@dataclass(frozen=True, eq=True)
class TensorType(Type):
    """
    A tensor is a type that represents a tensor of a given shape.
    """

    dtype: Type
    shape: Tuple[int]

    def walk_type_nodes(self):
        # is atomic. we do not consider the dtype and shape as nodes.
        yield self

    def children(self):
        yield from []

    def has_type_vars(self):
        assert not self.dtype.has_type_vars()
        return False

    def get_type_vars(self):
        assert not self.dtype.has_type_vars()
        return set()

    def subst_type_vars(self, subst: Dict[str, Type]):
        assert not self.has_type_vars()
        return self


@dataclass(frozen=True, eq=True)
class ListType(Type):
    """
    A list type is a type of the form [t] where t is a type.
    """

    element_type: Type

    def walk_type_nodes(self):
        yield self
        yield from self.element_type.walk_type_nodes()

    def children(self):
        yield self.element_type

    def has_type_vars(self):
        return self.element_type.has_type_vars()

    def get_type_vars(self):
        return self.element_type.get_type_vars()

    def subst_type_vars(self, subst: Dict[str, Type]):
        if not self.has_type_vars():
            return self
        return ListType(self.element_type.subst_type_vars(subst))


@dataclass(frozen=True, eq=True)
class ArrowType(Type):
    """
    An arrow type is a type of the form t1 -> t2 where t1 and t2 are types.
    """

    input_type: Tuple[Type]
    output_type: Type

    def __post_init__(self):
        assert isinstance(self.input_type, tuple), "input_type must be a tuple"

    def has_type_vars(self):
        return (
            any(t.has_type_vars() for t in self.input_type)
            or self.output_type.has_type_vars()
        )

    def get_type_vars(self):
        return self.output_type.get_type_vars() | set().union(
            *[t.get_type_vars() for t in self.input_type]
        )

    def subst_type_vars(self, subst: Dict[str, Type]):
        if not self.has_type_vars():
            return self
        return ArrowType(
            tuple(t.subst_type_vars(subst) for t in self.input_type),
            self.output_type.subst_type_vars(subst),
        )

    def walk_type_nodes(self):
        yield self
        for t in self.input_type:
            yield from t.walk_type_nodes()
        yield from self.output_type.walk_type_nodes()

    def children(self):
        yield from self.input_type
        yield self.output_type


@dataclass(frozen=True, eq=True)
class TypeVariable(Type):
    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def get_type_vars(self):
        return {self.name}

    def has_type_vars(self):
        return True

    def subst_type_vars(self, subst: Dict[str, Type]):
        return subst.get(self.name, self)

    def walk_type_nodes(self):
        yield self

    def children(self):
        yield from []


float_t = AtomicType("f")
