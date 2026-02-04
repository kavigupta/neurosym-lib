import copy

import torch

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import (
    IdentityProgramEmbedding,
    MinimalStepsNearStructuralCost,
    NearCost,
    NearValidationHeuristic,
    ProgramEmbedding,
)
from neurosym.examples.near.methods.base_trainer import schedule_optimizer
from neurosym.examples.near.methods.near_example_trainer import NEARTrainerConfig
from neurosym.examples.near.metrics import compute_metrics
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
    ) -> float:
        """
        Initializes a TorchProgramModule and trains it. Returns the trained module, and the
        validation loss.

        :param program: The program to validate.

        :returns: A tuple containing the trained TorchProgramModule and the validation loss.
        """
        log(f"Training {render_s_expression(model.uninitialize())}")

        model = self.program_to_module(dsl, model, embedding)

        try:
            val_loss = _train_model(
                model,
                self.datamodule,
                n_epochs=self.n_epochs,
                trainer_cfg=self.trainer_cfg,
            )
        except torch.OutOfMemoryError:
            log(
                "  Out of memory during training even with reduced batch size, returning inf cost"
            )
            return float("inf")

        return val_loss

    def program_to_module(
        self, dsl: DSL, program: InitializedSExpression, embedding: ProgramEmbedding
    ) -> torch.nn.Module:
        """
        Convert a program to a TorchProgramModule, which can then be trained.
        This can be overriden in subclasses to provide custom behavior, e.g.,
        integrating the program module into a larger model.

        :param program: The program to convert.
        :returns: The full Torch nn.Module to train.
        """
        program_module = TorchProgramModule(dsl, program)

        model = embedding.embed_initialized_program(program_module)
        return model


def default_near_cost(
    *,
    trainer_cfg: NEARTrainerConfig,
    datamodule: DatasetWrapper,
    progress_by_epoch=False,
    embedding: ProgramEmbedding = IdentityProgramEmbedding(),
    structural_cost_penalty: float = 0.01,
    symbol_costs=None,
    cost=MinimalStepsNearStructuralCost,
    **kwargs,
):
    """
    Default NearCost. Uses additive combination: f = validation_cost + penalty * structural_cost

    :param neural_dsl: The neural DSL to use.
    :param trainer_cfg: The configuration for the trainer.
    :param datamodule: The data module to use.
    :param progress_by_epoch: Whether to display progress by epoch.
    :param embedding: The embedding to use.
    :param structural_cost_penalty: Penalty multiplier for structural cost (default: 0.01).
    :param kwargs: Additional arguments to pass to the trainer.
    """
    return NearCost(
        structural_cost=cost(symbol_costs=symbol_costs or {}),
        validation_heuristic=ValidationCost(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
            progress_by_epoch=progress_by_epoch,
            **kwargs,
        ),
        structural_cost_penalty=structural_cost_penalty,
        embedding=embedding,
    )


class _ValidationState:
    """Tracks the training search state for validation and early stopping."""

    def __init__(self):
        self.best_val = float("inf")
        self.best_weights = None
        self.epochs_no_improve = 0
        self.best_acc = -float("inf")

    def update_best_acc(self, val_acc: float):
        """Update the best accuracy seen so far."""
        self.best_acc = max(self.best_acc, val_acc)


def _validate_model(model, datamodule, trainer_cfg, validation_state: _ValidationState):
    """
    Run validation on the model and compute metrics.

    :param model: The model to validate.
    :param datamodule: The data module containing validation data.
    :param trainer_cfg: The trainer configuration.
    :param validation_state: The validation state tracker.
    :returns: should_break (bool) indicating whether training should stop early.
    """
    model.eval()
    val_loss_sum, val_cnt = 0.0, 0
    labels, predictions = [], []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            batch = [v.to(trainer_cfg.accelerator) for v in batch]
            x, y = batch
            pred_y = model(x, environment=())
            val_loss_sum += trainer_cfg.loss_callback(pred_y, y).item()
            val_cnt += 1
            labels.append(y.cpu())
            predictions.append(pred_y.cpu())
    val_loss = val_loss_sum / val_cnt
    labels = torch.cat(labels, dim=0)
    predictions = torch.cat(predictions, dim=0).detach()
    metrics = compute_metrics(
        predictions=predictions.cpu().numpy(),
        ground_truth=labels.cpu().numpy(),
        metric_name="all",
    )
    metric_key = trainer_cfg.validation_metric
    if metric_key not in metrics:
        raise KeyError(
            f"Validation metric '{metric_key}' is unavailable. "
            f"Available metrics: {list(metrics.keys())}"
        )
    val_acc = float(metrics[metric_key])
    validation_state.update_best_acc(val_acc)

    should_break = False
    if trainer_cfg.early_stopping:
        if val_loss < validation_state.best_val - trainer_cfg.early_stopping_min_delta:
            validation_state.best_val = val_loss
            validation_state.best_weights = copy.deepcopy(model.state_dict())
            validation_state.epochs_no_improve = 0
        else:
            validation_state.epochs_no_improve += 1
            if (
                validation_state.epochs_no_improve
                >= trainer_cfg.early_stopping_patience
            ):
                should_break = True

    return should_break


# pylint: disable=too-many-branches,too-many-statements
def _train_model(model, datamodule, *, n_epochs, trainer_cfg: NEARTrainerConfig):
    validation_state = _ValidationState()
    if n_epochs is None:
        n_epochs = trainer_cfg.n_epochs
    if any(
        p.requires_grad for p in model.parameters()
    ):  # only train if there are parameters to train
        torch.manual_seed(trainer_cfg.seed)

        optimizer, schedulers = schedule_optimizer(
            trainer_cfg.optimizer(
                model.parameters(),
                lr=trainer_cfg.lr,
                weight_decay=trainer_cfg.weight_decay,
            ),
            trainer_cfg.scheduler,
            len(datamodule.train),
            n_epochs,
        )
        model = model.to(trainer_cfg.accelerator)
        for epoch in range(n_epochs):
            model.train()
            for batch in datamodule.train_dataloader():
                optimizer.zero_grad()
                batch = [v.to(trainer_cfg.accelerator) for v in batch]
                x, y = batch
                loss = trainer_cfg.loss_callback(model(x, environment=()), y)
                loss.backward()
                optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()

            if epoch == (n_epochs - 1) or epoch % trainer_cfg.validation_interval == 0:
                if _validate_model(model, datamodule, trainer_cfg, validation_state):
                    break
    else:
        _validate_model(model, datamodule, trainer_cfg, validation_state)

    if validation_state.best_weights is not None:
        model.load_state_dict(validation_state.best_weights)
    model.cpu()
    # return best_val
    return -1 * validation_state.best_acc
