from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Tuple

from frozendict import frozendict

from neurosym.types.type import Type


@dataclass(frozen=True, eq=True)
class Environment:
    """
    Represents a type environment containing a stack of types.
    """

    _elements: Dict[int, Type]  # actually a frozendict

    @classmethod
    def empty(cls):
        """
        The empty environment.
        """
        return cls(frozendict())

    def child(self, *new_typs: Tuple[Type]) -> "Environment":
        """
        Add the given types to the top of the environment,
        in reverse order. This environment is not modified.

        :param new_typs: The types to add.
        """
        new_typs = new_typs[
            ::-1
        ]  # reverse, so that the first element is the first index
        result = {i + len(new_typs): typ for i, typ in self._elements.items()}
        for i, new_typ in enumerate(new_typs):
            result[i] = new_typ
        return Environment(frozendict(result))

    def parent(self, new_types) -> "Environment":
        """
        Assert that the given types are at the top of the environment,
        and remove them. This environment is not modified.

        :param new_types: The types to remove.
        """
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
        Merge two environments, asserting that they agree on the same indices.
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
        Merge a list of environments, doing so in order.
        """
        result = {}
        for env in environments:
            # pylint: disable=protected-access
            result.update(env._elements)
        return Environment(frozendict(result))

    def contains_type_at(self, typ: Type, index: int) -> bool:
        """
        Returns whether the environment contains the given type at the given index.
        """
        return index in self._elements and self._elements[index] == typ

    def __len__(self):
        """
        The number of elements in the environment. This includes "skipped" indices.
        """
        return 0 if not self._elements else max(self._elements) + 1

    def short_repr(self):
        """
        Produce a short representation of the environment.
        """
        return ",".join(
            f"{i}={self._elements[i].short_repr()}" for i in sorted(self._elements)
        )

    @cached_property
    def unique_hash(self):
        """
        Compute a unique hash for this environment.
        """
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return "E", tuple(
            sorted((i, render_type(typ)) for i, typ in self._elements.items())
        )


@dataclass(frozen=True, eq=True)
class PermissiveEnvironmment:
    """
    Like Environment, but allows any type at any index.
    """

    unique_hash = "P"

    def child(self, *new_types: Tuple[Type]):
        """
        Just return self, since any types are allowed.
        """
        del new_types
        return self

    def parent(self, new_types):
        """
        Just return self, since any types are allowed.
        """
        del new_types
        return self

    def contains_type_at(self, typ: Type, index: int):
        """
        Just return True, since any types are allowed.
        """
        del typ, index
        return True

    def __len__(self):
        """
        The number of elements in this environment is 0, just a placeholder.
        """
        return 0


@dataclass(frozen=True, eq=True)
class TypeWithEnvironment:
    """
    Represents a type in a given environment.

    :param typ: The type.
    :param env: The environment.
    """

    typ: Type
    env: Environment

    @cached_property
    def unique_hash(self):
        """
        Return a unique hash for this object.
        """
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return render_type(self.typ), self.env.unique_hash
