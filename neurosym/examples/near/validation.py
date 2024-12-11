from typing import Tuple, Union

import torch

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.methods.base_trainer import schedule_optimizer
from neurosym.examples.near.methods.near_example_trainer import NEARTrainerConfig
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


class ValidationCost:
    """
    A class that computes the validation cost of a program using a neural DSL.
    This is epsilon-admissible heuristic: https://arxiv.org/abs/2007.12101

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param error_loss: The loss to return if the program is invalid.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param structural_cost_weight: Linearly interpolates b/w structural cost and validation loss.
        The scale of the validation cost (float) and structural_cost (int) can
        vary so it's important to tune this for each new problem.
    :param kwargs: Additional arguments to pass to the trainer.
    """

    def __init__(
        self,
        *,
        neural_dsl: DSL,
        trainer_cfg: NEARTrainerConfig,
        datamodule: DatasetWrapper,
        error_loss=10000,
        progress_by_epoch=False,
        structural_cost_weight=0.5,
    ):
        self.neural_dsl = neural_dsl
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.error_loss = error_loss
        self.structural_cost_weight = structural_cost_weight
        self.progress_by_epoch = progress_by_epoch

    def structural_cost(self, program: SExpression) -> int:
        """
        Recursively calculates the structural cost of a program.
        Approximated as the total number of Holes.

        :param program: The program to compute the structural cost for.

        :returns: An integer structural cost of the program.
        """
        cost = 0
        if isinstance(program, Hole):
            cost += 1
            return cost
        for child in program.children:
            cost += self.structural_cost(child)
        return cost

    def __call__(
        self, node: DSLSearchNode | SExpression, n_epochs: int = None
    ) -> float:
        """
        Trains a partial program. Returns validation cost after training.

        :param node: The partial program DSLSearchNode to compute the score for.

        :returns: The validation loss as a `float`.
        """
        if isinstance(node, DSLSearchNode):
            program = node.program
        else:
            program = node
        try:
            log(f"Training {render_s_expression(program)}")
            _, val_loss = self.validate_model(program=program, n_epochs=n_epochs)
        except UninitializableProgramError as e:
            log(e.message)
            return self.error_loss

        return (
            1 - self.structural_cost_weight
        ) * val_loss + self.structural_cost_weight * self.structural_cost(
            program=program
        )

    def validate_model(
        self, program: SExpression, n_epochs: Union[int, None] = None
    ) -> Tuple[TorchProgramModule, float]:
        """
        Initializes a TorchProgramModule and trains it. Returns the trained module, and the
        validation loss.

        :param program: The program to validate.

        :returns: A tuple containing the trained TorchProgramModule and the validation loss.
        """
        module, val_loss = self._fit_trainer(program, n_epochs=n_epochs)
        return module, val_loss

    def _fit_trainer(self, program, *, n_epochs):
        program_module, model = self.program_to_module(program)

        val_loss = _train_model(
            model, self.datamodule, n_epochs=n_epochs, trainer_cfg=self.trainer_cfg
        )

        return program_module, val_loss

    def program_to_module(
        self, program: SExpression
    ) -> Tuple[TorchProgramModule, torch.nn.Module]:
        """
        Convert a program to a TorchProgramModule, which can then be trained.
        This can be overriden in subclasses to provide custom behavior, e.g.,
        integrating the program module into a larger model.

        :param program: The program to convert.
        :returns: A tuple containing the TorchProgramModule and the full Torch model to train.
            These should share weights, so that training the model also trains the program.
        """
        try:
            model = TorchProgramModule(
                self.neural_dsl, self.neural_dsl.initialize(program)
            )
        except PartialProgramNotFoundError as e:
            raise UninitializableProgramError(
                f"Partial Program not found for {render_s_expression(program)}"
            ) from e

        if len(list(model.parameters())) == 0:
            raise UninitializableProgramError(
                f"No parameters in program {render_s_expression(program)}"
            )
        return model, model


class UninitializableProgramError(Exception):
    """
    UninitializableProgramError is raised when a program cannot be
    initialized due to either an inability to fill a hole in a partial program
    or when a program has no parameters.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def _train_model(model, datamodule, *, n_epochs, trainer_cfg: NEARTrainerConfig):
    if n_epochs is None:
        n_epochs = trainer_cfg.n_epochs
    optimizer, schedulers = schedule_optimizer(
        trainer_cfg.optimizer(
            model.parameters(), lr=trainer_cfg.lr, weight_decay=trainer_cfg.weight_decay
        ),
        trainer_cfg.scheduler,
        len(datamodule.train),
        n_epochs,
    )
    torch.manual_seed(trainer_cfg.seed)
    model = model.train()
    model = model.to(trainer_cfg.accelerator)
    for _ in range(n_epochs):
        for batch in datamodule.train_dataloader():
            batch = {k: v.to(trainer_cfg.accelerator) for k, v in batch.items()}
            x, y = batch["inputs"], batch["outputs"]
            optimizer.zero_grad()
            loss = trainer_cfg.loss_callback(model(x, environment=()), y)
            loss.backward()
            optimizer.step()
            for scheduler in schedulers:
                scheduler.step()

    model = model.eval()
    val_loss_sum = 0
    val_loss_count = 0
    for batch in datamodule.val_dataloader():
        batch = {k: v.to(trainer_cfg.accelerator) for k, v in batch.items()}
        x, y = batch["inputs"], batch["outputs"]
        with torch.no_grad():
            val_loss_sum += trainer_cfg.loss_callback(
                model(x, environment=()), y
            ).item()
        val_loss_count += 1
    model = model.train().cpu()
    return val_loss_sum / val_loss_count
