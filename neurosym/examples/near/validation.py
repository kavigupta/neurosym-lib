from typing import List, Tuple

import torch
import tqdm.auto as tqdm

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.methods.base_trainer import schedule_optimizer
from neurosym.examples.near.methods.near_example_trainer import (
    NEARTrainerConfig,
    classification_mse_loss,
)
from neurosym.examples.near.models.torch_program_module import TorchProgramModule
from neurosym.examples.near.neural_dsl import PartialProgramNotFoundError
from neurosym.programs.hole import Hole
from neurosym.programs.s_expression import SExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.search_graph.dsl_search_node import DSLSearchNode
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


class _ProgressBar(pl.callbacks.Callback):
    """
    callback that updates a progress bar once per epoch
    """

    def __init__(self, num_epochs, progress_bar):
        self.num_epochs = num_epochs
        self.progress_bar = progress_bar

    def on_train_epoch_end(self, trainer, pl_module):
        self.progress_bar.update(1)
        # set train and val loss in progress bar
        self.progress_bar.set_postfix(
            train_loss=trainer.callback_metrics["train_loss"].item(),
            val_loss=trainer.callback_metrics["val_loss"].item(),
        )


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
    :param callbacks: Callbacks to use during training.
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
        callbacks: List[pl.callbacks.Callback] = (),
        **kwargs,
    ):
        self.neural_dsl = neural_dsl
        self.trainer_cfg = trainer_cfg
        self.datamodule = datamodule
        self.error_loss = error_loss
        self.structural_cost_weight = structural_cost_weight
        self.kwargs = kwargs
        self.progress_by_epoch = progress_by_epoch
        self.callbacks = list(callbacks)

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

    def __call__(self, node: DSLSearchNode) -> float:
        """
        Initialize a pl.Trainer object and train on a partial program.
        Returns validation cost after training.

        :param node: The partial program DSLSearchNode to compute the score for.

        :returns: The validation loss as a `float`.
        """
        try:
            log(f"Training {render_s_expression(node.program)}")
            _, val_loss = self.validate_model(program=node.program)
        except UninitializableProgramError as e:
            log(e.message)
            return self.error_loss

        return (
            1 - self.structural_cost_weight
        ) * val_loss + self.structural_cost_weight * self.structural_cost(
            program=node.program
        )

    def validate_model(
        self, program: SExpression
    ) -> Tuple[TorchProgramModule, pl.Trainer]:
        """
        Initializes a TorchProgramModule and trains it using a pl.Trainer. Returns the trained module,
        and the trainer object.

        :param program: The program to validate.

        :returns: A tuple containing the trained TorchProgramModule and the pl.Trainer object.
        """
        trainer, pbar = self._get_trainer_and_pbar()
        module, val_loss = self._fit_trainer(trainer, program, pbar)
        return module, val_loss

    @staticmethod
    def _duplicate(callbacks):
        """
        Reinitialize all callbacks to avoid sharing state between different validation runs.
        """
        out = []
        for cb in callbacks:
            out.append(
                cb.__class__(
                    **{
                        k: getattr(cb, k)
                        for k in cb.__init__.__code__.co_varnames
                        if hasattr(cb, k)
                    }
                )
            )
        return out

    def _get_trainer_and_pbar(self):
        callbacks = list(self.callbacks)
        callbacks = self._duplicate(self.callbacks)
        if self.progress_by_epoch:
            pbar = tqdm.tqdm(total=self.trainer_cfg.n_epochs, desc="Training")
            callbacks.append(_ProgressBar(self.trainer_cfg.n_epochs, pbar))
        else:
            pbar = None

        self.kwargs["max_epochs"] = self.kwargs.get(
            "max_epochs", self.trainer_cfg.n_epochs
        )
        self.kwargs["devices"] = self.kwargs.get("devices", "auto")
        self.kwargs["accelerator"] = self.kwargs.get("accelerator", "cpu")
        self.kwargs["enable_checkpointing"] = self.kwargs.get(
            "enable_checkpointing", False
        )
        self.kwargs["logger"] = self.kwargs.get("logger", False)
        self.kwargs["callbacks"] = callbacks
        self.kwargs["deterministic"] = self.kwargs.get("deterministic", True)
        trainer = self.kwargs
        return trainer, pbar

    def _fit_trainer(self, trainer, program, pbar):
        try:
            model = TorchProgramModule(dsl=self.neural_dsl, program=program)
        except PartialProgramNotFoundError as e:
            raise UninitializableProgramError(
                f"Partial Program not found for {render_s_expression(program)}"
            ) from e

        if len(list(model.parameters())) == 0:
            raise UninitializableProgramError(
                f"No parameters in program {render_s_expression(program)}"
            )

        model, val_loss = _train_model(
            model,
            self.datamodule,
            accelerator=trainer["accelerator"],
            # lr=trainer["lr"],
            lr=self.trainer_cfg.lr,
            # weight_decay=trainer["weight_decay"],
            weight_decay=self.trainer_cfg.weight_decay,
            n_epochs=trainer["max_epochs"],
            seed=self.trainer_cfg.seed,
            scheduler_type=self.trainer_cfg.scheduler,
            optimizer_type=self.trainer_cfg.optimizer,
            loss_callback=self.trainer_cfg.loss_callback,
        )
        if self.progress_by_epoch:
            pbar.close()

        return model, val_loss


class UninitializableProgramError(Exception):
    """
    UninitializableProgramError is raised when a program cannot be
    initialized due to either an inability to fill a hole in a partial program
    or when a program has no parameters.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def _train_model(
    model,
    datamodule,
    *,
    accelerator,
    lr,
    weight_decay=0,
    n_epochs,
    seed=0,
    scheduler_type="cosine",
    optimizer_type=torch.optim.Adam,
    loss_callback=classification_mse_loss,
):
    optimizer, schedulers = schedule_optimizer(
        optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay),
        scheduler_type,
        len(datamodule.train),
        n_epochs,
    )
    torch.manual_seed(seed)
    model = model.train()
    model = model.to(accelerator)
    for _ in range(n_epochs):
        for batch in datamodule.train_dataloader():
            batch = {k: v.to(accelerator) for k, v in batch.items()}
            x, y = batch["inputs"], batch["outputs"]
            optimizer.zero_grad()
            loss = loss_callback(model(x, environment=()), y)
            loss.backward()
            optimizer.step()
            for scheduler in schedulers:
                scheduler.step()

    model = model.eval()
    val_loss_sum = 0
    val_loss_count = 0
    for batch in datamodule.val_dataloader():
        batch = {k: v.to(accelerator) for k, v in batch.items()}
        x, y = batch["inputs"], batch["outputs"]
        with torch.no_grad():
            val_loss_sum += loss_callback(model(x, environment=()), y).item()
        val_loss_count += 1
    model = model.train().cpu()
    return model, val_loss_sum / val_loss_count
