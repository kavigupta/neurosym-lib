import sys
from dataclasses import dataclass
from typing import Tuple

import lightning as L
import torch
from sklearn.metrics import f1_score, hamming_loss
from torch import nn


class TrainingError(Exception):
    pass


def get_loss_fn(loss_fn):
    match loss_fn:
        case "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        case "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        case "MSELoss":
            return nn.MSELoss()
        case "NLLLoss":
            return nn.NLLLoss()
        case _:
            raise NotImplementedError(f"Loss function {loss_fn} not implemented")  # noqa: E501


@dataclass
class BaseTrainerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_epochs: int = 10
    train_steps: int = -1  # Set programmaticaly
    seed: int = 44
    evaluate: bool = True
    resume: str = ""
    scheduler: str = "cosine"
    sav_dir: str = "data/shapeworldonly_checkpoints"
    _filter_param_list: Tuple[str] = ()
    scheduler: str = "cosine"
    optimizer: str = "adam"


class BaseTrainer(L.LightningModule):
    """
    An abstract class that defines the supporting code to
    train a neural module. We use pytorch-lightning as the
    base framework.
    """

    @classmethod
    def from_arch(cls, model: nn.Module, config: BaseTrainerConfig):
        """
        Instantiates a neural module from a torch.nn.Module.
        """
        return cls(model, config)

    def __init__(self, model: nn.Module, config: BaseTrainerConfig):
        super().__init__()
        self.config = config
        self.model = model
        self.save_hyperparameters(ignore=["model"])

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
        f1_scores = BaseTrainer.compute_average_f1_score(
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
                    predictions = predictions.reshape(-1, predictions.shape[-1])
                    targets = targets.view(-1)
                case "MSELoss":
                    if not self.is_regression:
                        targets = targets.squeeze(-1)
                        # pylint: disable=not-callable
                        targets = torch.nn.functional.one_hot(
                            targets, num_classes=self.config.num_labels
                        ).float()
                    predictions = predictions.reshape(-1, predictions.shape[-1])
                    targets = targets.view(-1, targets.shape[-1])
                case "NLLLoss":
                    predictions = (
                        predictions.reshape(-1, predictions.shape[-1])
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

    def _step(self) -> dict:
        raise NotImplementedError("Step function not implemented.")

    def training_step(self, train_batch, batch_idx):
        # pylint: disable=arguments-differ
        del batch_idx
        losses = self._step(validation=False, **train_batch)
        train_loss = sum(losses.values()) / len(losses)
        for key, value in losses.items():
            self.log(f"train_{key}", value.item(), prog_bar=True, sync_dist=True)

        self.log("train_loss", train_loss, prog_bar=True, sync_dist=True)
        return {"loss": train_loss}

    def evaluation_step(self, evaluation_batch, batch_idx, split):
        # pylint: disable=arguments-differ
        del batch_idx
        losses = self._step(validation=True, **evaluation_batch)
        evaluation_loss = sum(losses.values()) / len(losses)
        for key, value in losses.items():
            self.log(f"{split}_{key}", value.item(), prog_bar=True, sync_dist=True)

        self.log(f"{split}_loss", evaluation_loss, prog_bar=True, sync_dist=True)
        return {f"{split}_loss": evaluation_loss}

    def validation_step(self, val_batch, batch_idx):
        # pylint: disable=arguments-differ
        return self.evaluation_step(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        # pylint: disable=arguments-differ
        return self.evaluation_step(test_batch, batch_idx, "test")

    @staticmethod
    def filter_parameters(named_parameters: dict, filter_param_list: list):
        params = []
        for name, param in named_parameters:
            valid_name = not any(x in name for x in filter_param_list)
            if param.requires_grad and valid_name:
                params.append(param)
            else:
                print(
                    f"WARNING: Parameter {name} removed from optimizer", file=sys.stderr
                )
        return params

    @staticmethod
    def warm_and_decay_lr_scheduler(warmup_steps_pct, decay_steps_pct, total_steps):
        def f(step):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < (
                total_steps + 1
            ), f"Step {step} is greater thantotal steps {total_steps}"
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= 0.5 ** (step / decay_steps)
            return factor

        return f

    def step_lr_scheduler(self, total_steps: int):
        def f(step: int):
            if step < total_steps * 0.3:
                factor = 1
            elif step < total_steps * 0.3:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        return f

    def configure_optimizers(self):
        """
        A rather verbose function that instantiates the optimizer and scheduler.
        """
        # pylint: disable=protected-access
        params = self.filter_parameters(
            self.named_parameters(), self.config._filter_param_list
        )

        match self.config.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(
                    params, lr=self.config.lr, weight_decay=self.config.weight_decay
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    params, lr=self.config.lr, weight_decay=self.config.weight_decay
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    params, lr=self.config.lr, weight_decay=self.config.weight_decay
                )
            case _:
                raise NotImplementedError(
                    f"Optimizer {self.config.optimizer} not implemented"
                )  # noqa: E501

        assert self.config.train_steps != -1, "Train steps not set"
        total_steps = int(self.config.n_epochs * (self.config.train_steps))

        match self.config.scheduler:
            case "none":
                return optimizer
            case "cosine":
                warmup_steps_pct = 0.02
                decay_steps_pct = 0.2
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=self.warm_and_decay_lr_scheduler(
                        warmup_steps_pct, decay_steps_pct, total_steps
                    ),
                )
                return (
                    [optimizer],
                    [
                        {
                            "scheduler": scheduler,
                            "interval": "step",
                        }
                    ],
                )
            case "step":
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer, lr_lambda=self.step_lr_scheduler(total_steps)
                )
                return (
                    [optimizer],
                    [
                        {
                            "scheduler": scheduler,
                            "interval": "step",
                        }
                    ],
                )
            case _:
                raise ValueError(f"Invalid scheduler {self.config.scheduler}")
