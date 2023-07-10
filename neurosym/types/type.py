from abc import ABC
from dataclasses import dataclass


class Type(ABC):
    pass


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str


# A list type is a type of the form [t] where t is a type.
@dataclass(frozen=True, eq=True)
class ListType(Type):
    element_type: Type
    