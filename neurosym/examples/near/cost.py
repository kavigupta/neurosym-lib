from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import (
    InitializedSExpression,
    SExpression,
    is_initialized_s_expression,
)
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.utils.logging import log


class StructuralCost(ABC):
    """
    Compute a structural cost for a program. This is used to guide the search towards
    more completed programs.
    """

    @abstractmethod
    def compute_structural_cost(self, model: SExpression) -> float:
        """
        Computes the structural cost of a partial program.
        """


class NearValidationHeuristic(ABC):
    """
    Compute a validation cost for a program. This is used to guide the search towards
    more accurate programs.
    """

    @abstractmethod
    def train_and_compute_cost(
        self, model: InitializedSExpression
    ) -> Tuple[InitializedSExpression, float]:
        """
        Train a model and compute the validation cost. This is allowed to
        mutate the original model.
        """

    def compute_cost(self, model: InitializedSExpression) -> float:
        """
        Compute the validation cost of a model.
        """
        _, val_loss = self.train_and_compute_cost(model)
        return val_loss


@dataclass
class NearCost:
    """
    Near cost function that combines a structural cost and a validation heuristic.
    """

    structural_cost: StructuralCost
    validation_heuristic: NearValidationHeuristic
    structural_cost_weight: float = 0.5
    error_loss: float = 10000

    def compute_cost(self, model: InitializedSExpression) -> float:
        """
        Compute the cost of a model.
        """
        val_loss = self.validation_heuristic.compute_cost(model)
        struct_cost = self.structural_cost.compute_structural_cost(model.uninitialize())
        return (
            1 - self.structural_cost_weight
        ) * val_loss + self.structural_cost_weight * struct_cost

    def __call__(self, node: DSLSearchNode) -> float:
        """
        Compute the cost of a model.
        """
        program = node.program
        if not isinstance(program, InitializedSExpression):
            try:
                program = node.dsl.initialize(program)
            except PartialProgramNotFoundError:
                log(f"Partial Program not found for {render_s_expression(program)}")
                # TODO
                return self.error_loss
            except UninitializableProgramError as e:
                log(e.message)
                return self.error_loss
        assert is_initialized_s_expression(program), type(program)
        return self.compute_cost(program)


class NumberHolesStructuralCost(StructuralCost):
    def compute_structural_cost(self, model: SExpression) -> float:
        cost = 0
        if isinstance(model, Hole):
            cost += 1
            return cost
        for child in model.children:
            cost += self.compute_structural_cost(child)
        return cost


class UninitializableProgramError(Exception):
    """
    UninitializableProgramError is raised when a program cannot be
    initialized due to either an inability to fill a hole in a partial program
    or when a program has no parameters.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message
