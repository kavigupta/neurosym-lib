from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


class Type(ABC):
    @abstractmethod
    def walk_type_nodes(self):
        """
        Walk the type tree and yield all nodes.
        """

    def is_atomic(self):
        """
        Return True if the type is atomic.
        """
        return len(list(self.walk_type_nodes())) == 1


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"

    def walk_type_nodes(self):
        yield self


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


@dataclass(frozen=True, eq=True)
class ListType(Type):
    """
    A list type is a type of the form [t] where t is a type.
    """

    element_type: Type

    def walk_type_nodes(self):
        yield self
        yield from self.element_type.walk_type_nodes()


@dataclass(frozen=True, eq=True)
class ArrowType(Type):
    """
    An arrow type is a type of the form t1 -> t2 where t1 and t2 are types.
    """

    input_type: Tuple[Type]
    output_type: Type

    def __post_init__(self):
        assert isinstance(self.input_type, tuple), "input_type must be a tuple"

    def walk_type_nodes(self):
        yield self
        for t in self.input_type:
            yield from t.walk_type_nodes()
        yield from self.output_type.walk_type_nodes()


float_t = AtomicType("f")
