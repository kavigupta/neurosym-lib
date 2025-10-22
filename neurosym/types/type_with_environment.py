from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple

from frozendict import frozendict

from neurosym.types.type import Type


@dataclass(frozen=True, eq=True)
class StrictEnvironment:
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

    def child(self, *new_typs: Tuple[Type]) -> "StrictEnvironment":
        """
        Add the given types to the top of the environment,
        in reverse order. This environment is not modified.

        :param new_typs: The types to add.
        """
        env = self
        # insert the elements in regular order so that the last element ends up at 0
        for typ in new_typs:
            env = env.attempt_insert(0, typ)
            assert env is not None
        return env

    def parent(self, new_types) -> "StrictEnvironment":
        """
        Assert that the given types are at the top of the environment,
        and remove them. This environment is not modified.

        :param new_types: The types to remove.
        """
        env = self
        # delete the last type first, since it starts at index 0
        for typ in new_types[::-1]:
            if len(env) == 0:
                # This is fine, there's just some missing stuff above
                return env
            env = env.attempt_remove(0, typ)
            assert (
                env is not None
            ), f"Could not remove type {typ} from environment {self}"
        return env

    def attempt_insert(
        self, index: int, typ: Optional[Type] = None
    ) -> "StrictEnvironment | None":
        """
        Attempt to insert the given type at the given index, shifting other types
        up by one. If the index is beyond the current length of the environment,
        this will fail and return None.
        """
        if index > len(self):
            return None
        result = {}
        for i, existing_typ in self._elements.items():
            if i < index:
                result[i] = existing_typ
            else:
                result[i + 1] = existing_typ
        if typ is not None:
            result[index] = typ
        return StrictEnvironment(frozendict(result))

    def attempt_remove(
        self, index: int, typ: Optional[Type] = None
    ) -> "StrictEnvironment | None":
        """
        Attempt to remove the given type at the given index, shifting other types
        down by one. If the index is beyond the current length of the environment,
        this will fail and return None.
        """
        if index >= len(self):
            return None
        if index in self._elements and (
            typ is not None and self._elements[index] != typ
        ):
            return None
        result = {}
        for i, existing_typ in self._elements.items():
            if i < index:
                result[i] = existing_typ
            elif i > index:
                result[i - 1] = existing_typ
        return StrictEnvironment(frozendict(result))

    def merge(self, other: "StrictEnvironment"):
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
        return StrictEnvironment(frozendict(result))

    @classmethod
    def merge_all(cls, *environments: List["StrictEnvironment"]):
        """
        Merge a list of environments, doing so in order.
        """
        result = {}
        for env in environments:
            # pylint: disable=protected-access
            result.update(env._elements)
        return StrictEnvironment(frozendict(result))

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
        from neurosym.types.type_string_repr import render_type

        return ",".join(
            f"{i}={render_type(self._elements[i])}" for i in sorted(self._elements)
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

    def __getitem__(self, index):
        """
        Get the type at the given index.
        """
        return self._elements[index]


@dataclass(frozen=True, eq=True)
class PermissiveEnvironmment:
    """
    Like StrictEnvironment, but allows any type at any index.
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

    def attempt_insert(self, index: int):
        """
        Just return self, since any types are allowed.
        """
        del index
        return self

    def attempt_remove(self, index: int):
        """
        Just return self, since any types are allowed.
        """
        del index
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

    def short_repr(self):
        """
        Produce a short representation of the environment.
        """
        return "*"


@dataclass(frozen=True, eq=True)
class TypeWithEnvironment:
    """
    Represents a type in a given environment.

    :param typ: The type.
    :param env: The environment.
    """

    typ: Type
    env: StrictEnvironment

    @cached_property
    def unique_hash(self):
        """
        Return a unique hash for this object.
        """
        # pylint: disable=cyclic-import
        from neurosym.types.type_string_repr import render_type

        return render_type(self.typ), self.env.unique_hash
