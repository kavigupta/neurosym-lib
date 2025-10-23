import copy

import numpy as np
import torch
from sklearn.metrics import hamming_loss

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
from neurosym.examples.near.methods.near_example_trainer import \
    NEARTrainerConfig
from neurosym.examples.near.models.torch_program_module import \
    TorchProgramModule
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.utils.imports import import_pytorch_lightning
from neurosym.utils.logging import log

pl = import_pytorch_lightning()


def flatten_sequence_logits_and_labels(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten model logits and ground-truth labels so that they share a common
    sample dimension.

    This handles sequence models whose logits have extra temporal dimensions
    (e.g., ``(batch, time, num_classes)``) while the labels may either already be
    expanded to match that temporal dimension or provided as a single label per
    batch element.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim < 2:
        raise ValueError(
            f"Expected logits with at least 2 dimensions, got shape {logits.shape}"
        )

    feature_shape = logits.shape[:-1]
    batch_size = feature_shape[0]
    temporal_factor = int(np.prod(feature_shape[1:])) if len(feature_shape) > 1 else 1

    flattened_logits = logits.reshape(-1, logits.shape[-1])
    total_samples = flattened_logits.shape[0]
    labels_flat = labels.reshape(-1)
    if labels_flat.shape[0] == total_samples:
        return flattened_logits, labels_flat

    if labels_flat.shape[0] == batch_size and temporal_factor > 1:
        labels_flat = np.repeat(labels_flat, temporal_factor)
        return flattened_logits, labels_flat

    labels_reshaped = labels.reshape(batch_size, -1)
    if labels_reshaped.shape[1] == 1 and temporal_factor > 1:
        labels_flat = np.repeat(labels_reshaped[:, 0], temporal_factor)
    elif labels_reshaped.shape[1] == temporal_factor:
        labels_flat = labels_reshaped.reshape(-1)
    else:
        raise ValueError(
            "Unable to align logits and labels shapes: "
            f"logits={logits.shape}, labels={labels.shape}"
        )

    if labels_flat.shape[0] != total_samples:
        raise ValueError(
            "Unable to align logits and labels after broadcasting: "
            f"logits={logits.shape}, labels={labels.shape}"
        )
    return flattened_logits, labels_flat


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
    ) -> float:
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
    structural_cost_weight: float = 0.5,
    symbol_costs=None,
    cost=MinimalStepsNearStructuralCost,
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
        structural_cost=cost(symbol_costs=symbol_costs or {}),
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
    if any(
        p.requires_grad for p in model.parameters()
    ):  # only train if there are parameters to train
        torch.manual_seed(trainer_cfg.seed)
        best_val = float("inf")
        best_acc = 0.0
        epochs_no_improve = 0
        best_weights = None

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
                threshold = np.quantile(
                    predictions.cpu().numpy(), 1 - labels.cpu().float().numpy().mean()
                )
                predictions = torch.sigmoid(predictions) < threshold
                val_acc = 1 - hamming_loss(
                    labels.cpu().numpy().flatten().astype(int),
                    predictions.cpu().numpy().flatten(),
                )

                if trainer_cfg.early_stopping:
                    if val_loss < best_val - trainer_cfg.early_stopping_min_delta:
                        best_val = val_loss
                        best_weights = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        best_acc = val_acc
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= trainer_cfg.early_stopping_patience:
                            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.cpu()
    # return best_val
    return -1 * best_acc
