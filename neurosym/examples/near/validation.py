from typing import Tuple

import torch

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import (
    IdentityProgramEmbedding,
    MinimalStepsNearStructuralCost,
    NearCost,
    NearValidationHeuristic,
    ProgramEmbedding,
    UninitializableProgramError,
)
from neurosym.examples.near.methods.base_trainer import schedule_optimizer
from neurosym.examples.near.methods.near_example_trainer import NEARTrainerConfig
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


class ValidationCost(NearValidationHeuristic):
    """
    A class that computes the validation cost of a program using a neural DSL.
    This is epsilon-admissible heuristic: https://arxiv.org/abs/2007.12101

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param structural_cost_weight: Linearly interpolates b/w structural cost and validation loss.
        The scale of the validation cost (float) and structural_cost (int) can
        vary so it's important to tune this for each new problem.
    :param kwargs: Additional arguments to pass to the trainer.
    """

    def __init__(
        self,
        *,
        trainer_cfg: NEARTrainerConfig,
        datamodule: DatasetWrapper,
        progress_by_epoch=False,
        n_epochs=None,
    ):
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.progress_by_epoch = progress_by_epoch
        self.n_epochs = n_epochs

    def with_n_epochs(self, n_epochs: int) -> "ValidationCost":
        """
        Returns a new ValidationCost object with a different number of epochs.
        """
        return ValidationCost(
            trainer_cfg=self.trainer_cfg,
            datamodule=self.datamodule,
            progress_by_epoch=self.progress_by_epoch,
            n_epochs=n_epochs,
        )

    def compute_cost(
        self, dsl: DSL, model: InitializedSExpression, embedding: ProgramEmbedding
    ) -> Tuple[InitializedSExpression, float]:
        """
        Initializes a TorchProgramModule and trains it. Returns the trained module, and the
        validation loss.

        :param program: The program to validate.

        :returns: A tuple containing the trained TorchProgramModule and the validation loss.
        """
        log(f"Training {render_s_expression(model.uninitialize())}")

        model = self.program_to_module(dsl, model, embedding)

        val_loss = _train_model(
            model, self.datamodule, n_epochs=self.n_epochs, trainer_cfg=self.trainer_cfg
        )

        return val_loss

    def program_to_module(
        self, dsl: DSL, program: InitializedSExpression, embedding: ProgramEmbedding
    ) -> torch.nn.Module:
        """
        Convert a program to a TorchProgramModule, which can then be trained.
        This can be overriden in subclasses to provide custom behavior, e.g.,
        integrating the program module into a larger model.

        :param program: The program to convert.
        :returns: A tuple containing the TorchProgramModule and the full Torch model to train.
            These should share weights, so that training the model also trains the program.
        """
        program_module = TorchProgramModule(dsl, program)

        model = embedding.embed_initialized_program(program_module)

        if len([x for x in model.parameters() if x.requires_grad]) == 0:
            raise UninitializableProgramError(
                f"No parameters in program {render_s_expression(program)}"
            )
        return model


def default_near_cost(
    *,
    trainer_cfg: NEARTrainerConfig,
    datamodule: DatasetWrapper,
    progress_by_epoch=False,
    embedding: ProgramEmbedding = IdentityProgramEmbedding(),
    structural_cost_weight: float = 0.5,
    symbol_costs=None,
    **kwargs,
):
    """
    Default NearCost. This is, by default, a 50/50 blend of structural cost and validation cost,
    with the given parameters.

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param embedding: The embedding to use.
    :param structural_cost_weight: Linearly interpolates b/w structural cost and validation loss.
    :param kwargs: Additional arguments to pass to the trainer.
    """
    return NearCost(
        structural_cost=MinimalStepsNearStructuralCost(symbol_costs=symbol_costs or {}),
        validation_heuristic=ValidationCost(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
            progress_by_epoch=progress_by_epoch,
            **kwargs,
        ),
        structural_cost_weight=structural_cost_weight,
        embedding=embedding,
    )


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
