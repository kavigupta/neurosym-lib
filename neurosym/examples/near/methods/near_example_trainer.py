# pylint: skip-file
from dataclasses import dataclass

import torch
from torch import nn

from .base_trainer import BaseTrainer, BaseTrainerConfig


@dataclass
class NEARTrainerConfig(BaseTrainerConfig):
    max_seq_len: int = 100
    loss_fn: str = "CrossEntropyLoss"
    num_labels: int = -1  # Set Programmatically
    is_regression: bool = False


class NEARTrainer(BaseTrainer):
    """
    An abstract class that defines the basic functions to
    implement and train a neural module.
    """

    def __init__(self, model: nn.Module, config: NEARTrainerConfig):
        super().__init__(model=model, config=config)
        assert config.num_labels > 0, "Number of labels must be set programmatically"
        self.is_regression = config.is_regression

        self.loss_fn = self.get_loss_fn(self.config.loss_fn)

    def _step(self, inputs, outputs, validation=False, **kwargs):
        # pylint: disable=arguments-differ
        del kwargs
        predictions = self.model(inputs)
        losses = dict(loss=self.loss(predictions, outputs))

        if validation and self.logger is not None and self.current_epoch % 2 == 0:
            self._log_program_accuracy(predictions, outputs, inputs)

        return losses

    def _compute_auroc(self, predictions, outputs):
        # scikit learn.
        # F1 score
        raise NotImplementedError("TODO")

    def _log_program_accuracy(self, predictions, outputs, inputs):
        del inputs
        if self.config.num_labels > 1:
            predictions = torch.argmax(predictions, dim=-1)
        else:
            predictions = torch.round(torch.sigmoid(predictions))
        targets = outputs
        correctness = NEARTrainer.label_correctness(
            predictions, targets, self.config.num_labels
        )
        self.logger.log_metrics(correctness, step=self.global_step)
