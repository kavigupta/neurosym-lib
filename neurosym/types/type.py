from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict
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

    def depth_of_type_variables(self) -> List[Tuple[str, int]]:
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

    def depth_of_type_variables(self) -> List[Tuple[str, int]]:
        return [(self.name, 0)]

    def subst_type_vars(self, subst: Dict[str, Type]):
        return subst.get(self.name, self)

    def walk_type_nodes(self):
        yield self

    def children(self):
        yield from []


float_t = AtomicType("f")
