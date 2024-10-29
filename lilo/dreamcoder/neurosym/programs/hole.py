from dataclasses import dataclass
from typing import Iterator, List

from ..types.type_with_environment import TypeWithEnvironment
from .s_expression import SExpression


@dataclass(eq=True, frozen=True)
class Hole:
    """
    Represents a hole in a program. A hole is a placeholder for a subexpression that is not yet known.

    :param id: The unique id of the hole.
    :param twe: The type and environment of the hole.
    """

    id: int
    twe: TypeWithEnvironment

    @classmethod
    def of(cls, twe: TypeWithEnvironment) -> "Hole":
        """
        Create a hole with the given type and environment.

        Automatically assigns a unique id to the hole.
        """
        assert isinstance(twe, TypeWithEnvironment)
        cls._id = getattr(cls, "_id", 0) + 1
        return cls(id=cls._id, twe=twe)

    def __to_pair__(self, for_stitch: bool) -> str:
        """
        Convert the hole to a string representation. Used in
        the rendering of SExpressions as strings.
        """
        from neurosym.types.type_string_repr import render_type

        del for_stitch
        env_short = self.twe.env.short_repr()
        if env_short:
            env_short = "|" + env_short
        return f"??::<{render_type(self.twe.typ)}{env_short}>"


def _replace_holes(
    program: SExpression, holes: List[Hole], hole_replacements: List[SExpression]
) -> SExpression:
    """
    Replace the holes in node with the hole_replacements in the given SExpression.

    :param program: the SExpression to replace holes in
    :param holes: the holes to replace
    :param hole_replacements: the replacements for the holes
    """

    assert len(holes) == len(hole_replacements)
    if isinstance(program, Hole):
        if program not in holes:
            return program
        return hole_replacements[holes.index(program)]
    return SExpression(
        program.symbol,
        tuple(
            _replace_holes(child, holes, hole_replacements)
            for child in program.children
        ),
    )


def _all_holes(program: SExpression) -> Iterator[Hole]:
    """
    Yield all holes in the given SExpression.

    :param program: the SExpression to find holes in
    """
    if isinstance(program, Hole):
        yield program
    else:
        for child in program.children:
            yield from _all_holes(child)
