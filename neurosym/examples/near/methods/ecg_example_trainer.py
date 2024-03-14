from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from torch import nn

from .base_trainer import BaseTrainer, BaseTrainerConfig, TrainingError


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
        match self.config.loss_fn:
            case "CrossEntropyLoss":
                self.loss_fn = nn.CrossEntropyLoss()
            case "BCEWithLogitsLoss":
                self.loss_fn = nn.BCEWithLogitsLoss()
            case "MSELoss":
                self.loss_fn = nn.MSELoss()
            case "NLLLoss":
                self.loss_fn = nn.NLLLoss()
            case _:
                raise NotImplementedError(
                    f"Loss function {self.config.loss_fn} not implemented"
                )  # noqa: E501

    @staticmethod
    def compute_average_f1_score(
        predictions: torch.Tensor, targets: torch.Tensor, num_labels: int
    ):  # noqa: E501
        if num_labels > 1:
            weighted_avg_f1 = 1 - f1_score(targets, predictions, average="weighted")
            unweighted_avg_f1 = 1 - f1_score(targets, predictions, average="macro")
            all_f1 = 1 - f1_score(targets, predictions, average=None)
            return dict(
                weighted_avg_f1=weighted_avg_f1,
                unweighted_avg_f1=unweighted_avg_f1,
                all_f1s=all_f1,
            )
        avg_f1 = 1 - f1_score(targets, predictions, average="binary")
        all_f1 = 1 - f1_score(targets, predictions, average=None)
        return dict(avg_f1=avg_f1, all_f1s=all_f1)

    @staticmethod
    def label_correctness(
        predictions: torch.Tensor, targets: torch.Tensor, num_labels: int
    ):  # noqa: E501
        hamming_accuracy = 1 - hamming_loss(
            targets.squeeze().cpu(), predictions.squeeze().cpu()
        )
        f1_scores = ECGTrainer.compute_average_f1_score(
            predictions, targets, num_labels
        )  # noqa: E501

        return dict(hamming_accuracy=hamming_accuracy, **f1_scores)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        # pylint: disable=arguments-differ
        if len(predictions.shape) == 3:
            # Handling seq2seq classification loss.
            match self.config.loss_fn:
                case "CrossEntropyLoss":
                    targets = targets.squeeze(-1)
                    assert (
                        len(targets.shape) == 2
                    ), "Targets must be 2D for classification"
                    predictions = predictions.view(-1, predictions.shape[-1])
                    targets = targets.view(-1)
                case "MSELoss":
                    if not self.is_regression:
                        targets = targets.squeeze(-1)
                        targets = torch.nn.functional.one_hot( # pylint: disable=no-member
                            targets, num_classes=self.config.num_labels
                        ).float()
                    predictions = predictions.view(-1, predictions.shape[-1])
                    targets = targets.view(-1, targets.shape[-1])
                case "NLLLoss":
                    predictions = (
                        predictions.view(-1, predictions.shape[-1])
                        .clamp(min=1e-10)
                        .log()
                        .log_softmax(dim=-1)
                    )
                    targets = targets.view(-1)
                case "MultiLabelSoftMarginLoss":
                    # Available in pytorch.
                    raise NotImplementedError("TODO")
                case _:
                    raise NotImplementedError(
                        f"Loss function {self.config.loss_fn} not implemented for seq2seq models"
                    )
        elif len(predictions.shape) == 2:
            match self.config.loss_fn:
                case "CrossEntropyLoss":
                    targets = targets.argmax(dim=-1)
                    # predictions = predictions.softmax(dim=-1)

        loss = self.loss_fn(predictions, targets)
        return loss

    def _step(self, inputs, outputs, validation=False, **kwargs): # pylint: disable=arguments-differ
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
            raise TrainingError(e) from e # pylint: disable=raise-missing-from
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
