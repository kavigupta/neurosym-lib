from dataclasses import dataclass
from typing import Dict, Tuple

from frozendict import frozendict

from neurosym.types.type import Type


@dataclass(frozen=True, eq=True)
class Environment:
    """
    Represents a type environment containing the variables.
    """

    _elements: Dict[int, Type]  # actually a frozendict

    @classmethod
    def empty(cls):
        return cls(frozendict())

    def child(self, *new_typs: Tuple[Type]):
        result = {i + len(new_typs) : typ for i, typ in self._elements.items()}
        for i, new_typ in enumerate(new_typs):
            result[i] = new_typ
        return Environment(frozendict(result))

    def __len__(self):
        return max(self._elements) + 1

@dataclass(frozen=True, eq=True)
class TypeWithEnvironment:
    """
    Represents a type in a given environment.
    """

    typ: Type
    env: Environment
