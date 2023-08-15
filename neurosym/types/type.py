from abc import ABC
from dataclasses import dataclass
from typing import Tuple


class Type(ABC):
    pass


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str

    def __post_init__(self):
        assert self.name.isidentifier(), f"{self.name} is not a valid identifier"


@dataclass(frozen=True, eq=True)
class TensorType(Type):
    """
    A tensor is a type that represents a tensor of a given shape.
    """

    dtype: Type
    shape: Tuple[int]


@dataclass(frozen=True, eq=True)
class ListType(Type):
    """
    A list type is a type of the form [t] where t is a type.
    """

    element_type: Type


@dataclass(frozen=True, eq=True)
class ArrowType(Type):
    """
    An arrow type is a type of the form t1 -> t2 where t1 and t2 are types.
    """

    input_type: Tuple[Type]
    output_type: Type

    def __post_init__(self):
        assert isinstance(self.input_type, tuple), "input_type must be a tuple"


float_t = AtomicType("f")
