from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        new_typs = new_typs[
            ::-1
        ]  # reverse, so that the first element is the first index
        result = {i + len(new_typs): typ for i, typ in self._elements.items()}
        for i, new_typ in enumerate(new_typs):
            result[i] = new_typ
        return Environment(frozendict(result))

    def parent(self, new_types):
        new_types = new_types[
            ::-1
        ]  # reverse, so that the first element is the first index
        for i, new_typ in enumerate(new_types):
            if i in self._elements:
                assert self._elements[i] == new_typ
        result = {
            i - len(new_types): typ
            for i, typ in self._elements.items()
            if i >= len(new_types)
        }
        return Environment(frozendict(result))

    def merge(self, other: "Environment"):
        """
        Merge two environments.
        """
        result = dict(self._elements)
        # pylint: disable=protected-access
        for i, typ in other._elements.items():
            if i in result:
                assert result[i] == typ
            else:
                result[i] = typ
        return Environment(frozendict(result))

    @classmethod
    def merge_all(cls, *environments: List["Environment"]):
        """
        Merge a list of environments.
        """
        result = {}
        for env in environments:
            # pylint: disable=protected-access
            result.update(env._elements)
        return Environment(frozendict(result))

    def contains_type_at(self, typ: Type, index: int):
        return index in self._elements and self._elements[index] == typ

    def __len__(self):
        return 0 if not self._elements else max(self._elements) + 1

    def short_repr(self):
        return ",".join(
            f"{i}={self._elements[i].short_repr()}" for i in sorted(self._elements)
        )


@dataclass(frozen=True, eq=True)
class PermissiveEnvironmment:
    def child(self, *new_types: Tuple[Type]):
        del new_types
        return self

    def parent(self, new_types):
        del new_types
        return self

    def contains_type_at(self, typ: Type, index: int):
        del typ, index
        return True

    def __len__(self):
        return 0


@dataclass(frozen=True, eq=True)
class TypeWithEnvironment:
    """
    Represents a type in a given environment.
    """

    typ: Type
    env: Environment
