from dataclasses import dataclass
from typing import Iterable

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import NearStructuralCost
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression


@dataclass
class MinimalStepsNearStructuralCostWithDrops(NearStructuralCost):
    """
    Structural cost that counts the minimal number of steps needed to fill
    each hole in a program.
    """

    symbol_costs: dict[str, int]

    def all_holes(self, model: SExpression) -> Iterable[Hole]:
        if isinstance(model, Hole):
            yield model
            return
        for child in model.children:
            yield from self.all_holes(child)

    def compute_structural_cost(self, model: SExpression, dsl: DSL) -> float:
        sizes_twes = [len(hole.twe.env) for hole in self.all_holes(model)]
        if not sizes_twes:
            return 0
        return sum(sizes_twes) / len(sizes_twes) + len(sizes_twes)
