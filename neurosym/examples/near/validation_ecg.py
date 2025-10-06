"""ECG-specific validation cost with custom metrics."""

from typing import Tuple

import torch

from neurosym.datasets.load_data import DatasetWrapper
from neurosym.dsl.dsl import DSL
from neurosym.examples.near.cost import ProgramEmbedding
from neurosym.examples.near.methods.ecg_example_trainer import (
    ECGTrainerConfig,
    compute_ecg_metrics,
)
from neurosym.examples.near.validation import ValidationCost, _train_model
from neurosym.programs.s_expression import InitializedSExpression
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.utils.logging import log


class ECGValidationCost(ValidationCost):
    """
    ECG-specific validation cost that computes additional metrics like AUROC and F1.

    :param trainer_cfg: ECGTrainerConfig with num_labels specified
    :param datamodule: The data module to use
    :param progress_by_epoch: Whether to display progress by epoch
    :param n_epochs: Number of epochs to train (overrides trainer_cfg if specified)
    """

    def __init__(
        self,
        *,
        trainer_cfg: ECGTrainerConfig,
        datamodule: DatasetWrapper,
        progress_by_epoch=False,
        n_epochs=None,
    ):
        super().__init__(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
            progress_by_epoch=progress_by_epoch,
            n_epochs=n_epochs,
        )
        assert isinstance(trainer_cfg, ECGTrainerConfig), (
            "ECGValidationCost requires ECGTrainerConfig"
        )
        assert trainer_cfg.num_labels > 0, "num_labels must be set in ECGTrainerConfig"
        self.ecg_trainer_cfg = trainer_cfg

    def compute_cost(
        self, dsl: DSL, model: InitializedSExpression, embedding: ProgramEmbedding
    ) -> Tuple[InitializedSExpression, float]:
        """
        Trains the model and returns validation loss along with ECG-specific metrics.

        :param dsl: The DSL
        :param model: The initialized program
        :param embedding: Program embedding
        :return: Validation loss (lower is better)
        """
        log(f"Training ECG model: {render_s_expression(model.uninitialize())}")

        torch_model = self.program_to_module(dsl, model, embedding)

        # Train and get validation loss
        val_loss = _train_model_with_metrics(
            torch_model,
            self.datamodule,
            n_epochs=self.n_epochs,
            trainer_cfg=self.ecg_trainer_cfg,
        )

        return val_loss


def _train_model_with_metrics(model, datamodule, *, n_epochs, trainer_cfg: ECGTrainerConfig):
    """
    Extended training function that also computes and logs ECG-specific metrics.

    :param model: The torch model to train
    :param datamodule: Data module
    :param n_epochs: Number of training epochs
    :param trainer_cfg: ECG trainer configuration
    :return: Validation loss
    """
    # Use the standard training loop
    val_loss = _train_model(model, datamodule, n_epochs=n_epochs, trainer_cfg=trainer_cfg)

    # Compute additional ECG metrics on validation set
    model = model.eval()
    all_predictions = []
    all_targets = []

    for batch in datamodule.val_dataloader():
        batch = {k: v.to(trainer_cfg.accelerator) for k, v in batch.items()}
        x, y = batch["inputs"], batch["outputs"]
        with torch.no_grad():
            pred = model(x, environment=())
            all_predictions.append(pred.cpu())
            all_targets.append(y.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute ECG metrics
    metrics = compute_ecg_metrics(
        all_predictions, all_targets, trainer_cfg.num_labels
    )

    # Log metrics (optional - can be extended to use wandb or other loggers)
    log(f"  Validation metrics: {metrics}")

    model = model.train().cpu()
    return val_loss
