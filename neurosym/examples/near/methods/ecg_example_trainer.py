# pylint: skip-file
from dataclasses import dataclass

import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .base_trainer import BaseTrainer, BaseTrainerConfig, TrainingError, get_loss_fn


@dataclass
class ECGTrainerConfig(BaseTrainerConfig):
    max_seq_len: int = 100
    loss_fn: str = "CrossEntropyLoss"
    num_labels: int = -1  # Set Programmatically
    is_regression: bool = False
    is_multilabel: bool = False


class ECGTrainer(BaseTrainer):
    """
    An abstract class that defines the basic functions to
    implement and train a neural module.
    """

    def __init__(self, model: nn.Module, config: ECGTrainerConfig):
        super().__init__(model=model, config=config)
        assert config.num_labels > 0, "Number of labels must be set programmatically"
        self.is_regression = config.is_regression
        self.loss_fn = get_loss_fn(self.config.loss_fn)

    def _step(self, inputs, outputs, validation=False, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        try:
            predictions = self.model(inputs.float())
            predictions = predictions.clamp(min=1e-10, max=1e10)
            metrics_dict = self.program_accuracy(
                predictions.detach().cpu(), outputs.cpu(), inputs.cpu()
            )
            losses = dict(loss=self.loss(predictions, outputs))
            if validation:
                self.log("val_acc", metrics_dict["hamming_accuracy"])
                self.log("val_auroc", metrics_dict["auroc"])
            else:
                self.log("train_acc", metrics_dict["hamming_accuracy"])
                self.log("train_auroc", metrics_dict["auroc"])
        except Exception as e:
            # print("training error")
            # import IPython; IPython.embed()
            raise TrainingError(e) from e  # pylint: disable=raise-missing-from
        return losses

    def program_accuracy(self, predictions, outputs, inputs):
        del inputs
        if self.config.num_labels > 1:
            predictions = predictions.softmax(dim=-1)
        else:
            predictions = torch.round(torch.sigmoid(predictions))
        metrics_dict = ECGTrainer.label_correctness(
            predictions.argmax(dim=-1), outputs.argmax(dim=-1), self.config.num_labels
        )
        metrics_dict["auroc"] = roc_auc_score(y_true=outputs, y_score=predictions)
        return metrics_dict
