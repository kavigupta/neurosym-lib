from abc import ABC, abstractmethod
from dataclasses import dataclass

from neurosym.dsl.dsl import DSL
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
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
    def compute_structural_cost(self, model: SExpression, dsl: DSL) -> float:
        """
        Computes the structural cost of a partial program.
        """


class NearValidationHeuristic(ABC):
    """
    Compute a validation cost for a program. This is used to guide the search towards
    more accurate programs.
    """

    @abstractmethod
    def compute_cost(
        self, dsl: DSL, model: InitializedSExpression, embedding: "ProgramEmbedding"
    ) -> float:
        """
        Train a model and compute the validation cost. This mutates the model
        to train it

        :param dsl: The DSL to use for training.
        :param model: The model to train. Will be mutated in place.
        :param embedding: The embedding to use for the model.

        :returns: The validation loss as a `float`.
        """


class ProgramEmbedding(ABC):
    """
    A class that embeds a program within a larger framework.
    """

    @abstractmethod
    def embed_program(self, program: SExpression) -> SExpression:
        """
        Embeds a program into a larger program.

        :param program: The program to embed.
        :returns: The embedded program.
        """

    @abstractmethod
    def embed_initialized_program(
        self, program: TorchProgramModule
    ) -> TorchProgramModule:
        """
        Embeds a program into a neural model.

        :param program: The program to embed.
        :returns: The neural model.
        """


class IdentityProgramEmbedding(ProgramEmbedding):
    """
    An embedding that does nothing.
    """

    def embed_program(self, program: SExpression) -> SExpression:
        return program

    def embed_initialized_program(
        self, program: TorchProgramModule
    ) -> TorchProgramModule:
        return program


@dataclass
class NearCost:
    """
    Near cost function that combines a structural cost and a validation heuristic.
    """

    structural_cost: NearStructuralCost
    validation_heuristic: NearValidationHeuristic
    structural_cost_weight: float = 0.5
    error_loss: float = 10000
    embedding: ProgramEmbedding = IdentityProgramEmbedding()

    def __post_init__(self):
        assert isinstance(
            self.embedding, ProgramEmbedding
        ), f"embedding must be a ProgramEmbedding, but was {self.embedding}"

    def compute_cost(self, dsl: DSL, model: InitializedSExpression) -> float:
        """
        Compute the cost of a model. Calling this will mutate the `model`.

        :param dsl: The DSL to use for training.
        :param model: The model to train. Will be mutated in place.
        """
        try:
            val_loss = self.validation_heuristic.compute_cost(
                dsl, model, self.embedding
            )
        except UninitializableProgramError as e:
            log(e.message)
            return self.error_loss
        struct_cost = self.structural_cost.compute_structural_cost(
            self.embedding.embed_program(model.uninitialize()), dsl
        )
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


class PerNodeNearStructuralCost(NearStructuralCost):
    """
    Structural cost is run on each hole in the program, then summed.
    """

    @abstractmethod
    def compute_node_cost(self, node: SExpression, dsl: DSL) -> float:
        """
        Compute the cost of a hole.
        """

    def compute_structural_cost(self, model: SExpression, dsl: DSL) -> float:
        cost = self.compute_node_cost(model, dsl)
        if isinstance(model, Hole):
            return cost
        for child in model.children:
            cost += self.compute_structural_cost(child, dsl)
        return cost


class NumberHolesNearStructuralCost(PerNodeNearStructuralCost):
    """
    Structural cost that counts the number of holes in a program.
    """

    def compute_node_cost(self, node: SExpression, dsl: DSL) -> float:
        if isinstance(node, Hole):
            return 1
        return 0


@dataclass
class MinimalStepsNearStructuralCost(PerNodeNearStructuralCost):
    """
    Structural cost that counts the minimal number of steps needed to fill
    each hole in a program.
    """

    symbol_costs: dict[str, int]

    def compute_node_cost(self, node: SExpression, dsl: DSL) -> float:
        if not isinstance(node, Hole):
            return max(self.symbol_costs.get(node.symbol, 0) - 1, 0)
        result = dsl.minimal_term_size_for_type(
            node.twe, symbol_costs=self.symbol_costs
        )
        return result


class UninitializableProgramError(Exception):
    """
    UninitializableProgramError is raised when a program cannot be
    initialized due to either an inability to fill a hole in a partial program
    or when a program has no parameters.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message
