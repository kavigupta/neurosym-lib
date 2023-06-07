from abc import ABC
from dataclasses import dataclass


class Type(ABC):
    pass


@dataclass(frozen=True, eq=True)
class AtomicType(Type):
    name: str
