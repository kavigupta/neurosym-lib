from abc import ABC, abstractmethod
from dataclasses import dataclass

from neurosym.dsl.dsl import DSL
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


class NearStructuralCost(ABC):
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
    def compute_cost(self, dsl: DSL, model: InitializedSExpression) -> float:
        """
        Train a model and compute the validation cost. This mutates the model
        to train it

        :param dsl: The DSL to use for training.
        :param model: The model to train. Will be mutated in place.

        :returns: The validation loss as a `float`.
        """


@dataclass
class NearCost:
    """
    Near cost function that combines a structural cost and a validation heuristic.
    """

    structural_cost: NearStructuralCost
    validation_heuristic: NearValidationHeuristic
    structural_cost_weight: float = 0.5
    error_loss: float = 10000

    def compute_cost(self, dsl: DSL, model: InitializedSExpression) -> float:
        """
        Compute the cost of a model. Calling this will mutate the `model`.

        :param dsl: The DSL to use for training.
        :param model: The model to train. Will be mutated in place.
        """
        try:
            val_loss = self.validation_heuristic.compute_cost(dsl, model)
        except UninitializableProgramError as e:
            log(e.message)
            return self.error_loss
        struct_cost = self.structural_cost.compute_structural_cost(model.uninitialize())
        return (
            1 - self.structural_cost_weight
        ) * val_loss + self.structural_cost_weight * struct_cost

    def __call__(
        self,
        node: DSLSearchNode | SExpression | InitializedSExpression,
        dsl: DSL | None = None,
    ) -> float:
        """
        Compute the cost of a node or program. Will initialize the program if it is not
        already initialized, but if it is initialized, it will mutate the program.

        :param node: The node or program to compute the cost of.
        :param dsl: The DSL to use for training. Only required if `node` is not
            a `DSLSearchNode`.

        :returns: The cost as a `float`.
        """
        if isinstance(node, DSLSearchNode):
            program = node.program
            assert (
                dsl is None or dsl == node.dsl
            ), "If node is a DSLSearchNode, dsl must be None or equal to node.dsl"
            dsl = node.dsl
        else:
            assert (
                dsl is not None
            ), "If node is not a DSLSearchNode, dsl must be provided"
            program = node
        if not is_initialized_s_expression(program):
            try:
                program = dsl.initialize(program)
            except PartialProgramNotFoundError:
                log(f"Partial Program not found for {render_s_expression(program)}")
                return self.error_loss
            except UninitializableProgramError as e:
                log(e.message)
                return self.error_loss
        assert is_initialized_s_expression(program), type(program)
        return self.compute_cost(dsl, program)


class PerHoleNearStructuralCost(NearStructuralCost):
    """
    Structural cost is run on each hole in the program, then summed.
    """

    @abstractmethod
    def compute_hole_cost(self, hole: Hole) -> float:
        """
        Compute the cost of a hole.
        """

    def compute_structural_cost(self, model: SExpression) -> float:
        if isinstance(model, Hole):
            return self.compute_hole_cost(model)
        cost = 0
        for child in model.children:
            cost += self.compute_structural_cost(child)
        return cost


class NumberHolesNearStructuralCost(PerHoleNearStructuralCost):
    """
    Structural cost that counts the number of holes in a program.
    """

    def compute_hole_cost(self, hole: Hole) -> float:
        return 1


class UninitializableProgramError(Exception):
    """
    UninitializableProgramError is raised when a program cannot be
    initialized due to either an inability to fill a hole in a partial program
    or when a program has no parameters.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message
