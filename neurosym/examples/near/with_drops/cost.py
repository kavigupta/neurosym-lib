from dataclasses import dataclass

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import (
    MinimalStepsNearStructuralCost,
    PerNodeNearStructuralCost,
)
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression


@dataclass
class MinimalStepsNearStructuralCostWithDrops(PerNodeNearStructuralCost):
    """
    Structural cost that counts the minimal number of steps needed to fill
    each hole in a program.
    """

    symbol_costs: dict[str, int]

    def extra_node_cost(self, node: SExpression, dsl: DSL) -> float:
        del dsl
        if not isinstance(node, Hole):
            return 0
        return len(node.twe.env)

    def compute_node_cost(self, node: SExpression, dsl: DSL) -> float:
        return MinimalStepsNearStructuralCost.compute_node_cost(
            self, node, dsl
        ) + self.extra_node_cost(node, dsl)
