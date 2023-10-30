from dataclasses import dataclass
from typing import List

from ..types.type_with_environment import TypeWithEnvironment
from .s_expression import SExpression


@dataclass(eq=True, frozen=True)
class Hole:
    @classmethod
    def of(cls, twe: TypeWithEnvironment) -> "Hole":
        assert isinstance(twe, TypeWithEnvironment)
        cls._id = getattr(cls, "_id", 0) + 1
        return cls(id=cls._id, twe=twe)

    id: int
    twe: TypeWithEnvironment

    def __to_pair__(self, for_stitch: bool) -> str:
        from neurosym.types.type_string_repr import render_type

        del for_stitch
        env_short = self.twe.env.short_repr()
        if env_short:
            env_short = "|" + env_short
        return f"??::<{render_type(self.twe.typ)}{env_short}>"


def replace_holes(
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
            replace_holes(child, holes, hole_replacements) for child in program.children
        ),
    )


def all_holes(program: SExpression) -> List[Hole]:
    """
    Yield all holes in the given SExpression.

    :param program: the SExpression to find holes in
    """
    if isinstance(program, Hole):
        yield program
    else:
        for child in program.children:
            yield from all_holes(child)
