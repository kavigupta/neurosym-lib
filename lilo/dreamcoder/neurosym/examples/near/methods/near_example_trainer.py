from dataclasses import dataclass
from typing import Callable

import torch
from sklearn.metrics import f1_score, hamming_loss
from torch import nn

from neurosym.utils.documentation import internal_only

from .base_trainer import BaseTrainer, BaseTrainerConfig


def classification_mse_loss(predictions, targets):
    """
    Loss function that computes the mean squared error between the predictions
    and the targets.

    :param predictions: The predictions from the model. Shape: ``(..., num_classes)``
    :param targets: The target labels, as integers. Shape: ``(..., 1)``
    """
    # pylint: disable=not-callable
    targets = torch.nn.functional.one_hot(
        targets.squeeze(-1), num_classes=predictions.shape[-1]
    ).float()
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1, targets.shape[-1])
    return nn.functional.mse_loss(predictions, targets)


@dataclass
class NEARTrainerConfig(BaseTrainerConfig):
    """
    Configuration class for training a NEAR model.

    :param lr: Learning rate for the optimizer (default: 1e-4)
    :param weight_decay: Weight decay for the optimizer (default: 0.0)
    :param n_epochs: Number of epochs to train for (default: 10)
    :param seed: Random seed for reproducibility (default: 44)
    :param evaluate: Whether to evaluate the model after training (default: True)
    :param resume: Path to a checkpoint to resume training from (default: "")
    :param scheduler: Learning rate scheduler to use (default: "cosine")
    :param sav_dir: Directory to save checkpoints to (default: "data/shapeworldonly_checkpoints")
    :param optimizer: Optimizer to use (default: torch.optim.Adam)
    :param max_seq_len: Maximum sequence length for the model (default: 100)
    :param loss_callback: Loss function to use (default: :py:func:`neurosym.examples.near.classification_mse_loss`)
    """

    max_seq_len: int = 100
    loss_callback: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        classification_mse_loss
    )
    num_labels: int = -1  # Set Programmatically


class NEARTrainer(BaseTrainer):
    """
    Trainer class for training a NEAR model. Is a pl.LightningModule.

    :param model: The module to train
    :param config: Configuration for the trainer
    """

    def __init__(self, model: nn.Module, config: NEARTrainerConfig):
        super().__init__(model=model, config=config)
        assert config.num_labels > 0, "Number of labels must be set programmatically"
        self.loss_fn = config.loss_callback

    @classmethod
    def _compute_average_f1_score(
        cls, predictions: torch.Tensor, targets: torch.Tensor, num_labels: int
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

    @classmethod
    def label_correctness(
        cls, predictions: torch.Tensor, targets: torch.Tensor, num_labels: int
    ):  # noqa: E501
        hamming_accuracy = 1 - hamming_loss(
            targets.squeeze().cpu(), predictions.squeeze().cpu()
        )
        f1_scores = cls._compute_average_f1_score(
            predictions, targets, num_labels
        )  # noqa: E501
        return dict(hamming_accuracy=hamming_accuracy, **f1_scores)

    @internal_only
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        # pylint: disable=arguments-differ
        loss = self.loss_fn(predictions, targets)
        return loss

    def _step(self, inputs, outputs, validation=False, **kwargs):
        # pylint: disable=arguments-differ
        del kwargs
        predictions = self.model(inputs, environment=())
        losses = dict(loss=self.loss(predictions, outputs))

        if validation and self.logger is not None and self.current_epoch % 2 == 0:
            self._log_program_accuracy(predictions, outputs, inputs)

        return losses

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


def _main():
    from neurosym.utils.imports import import_pytorch_lightning

    pl = import_pytorch_lightning()
    from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper

    def dataset_factory(train_seed):
        return DatasetWrapper(
            DatasetFromNpy(
                "../data/classification_example/train_ex_data.npy",
                "../data/classification_example/train_ex_labels.npy",
                train_seed,
            ),
            DatasetFromNpy(
                "../data/classification_example/test_ex_data.npy",
                "../data/classification_example/test_ex_labels.npy",
                None,
            ),
            batch_size=200,
        )

    datamodule = dataset_factory(42)
    # model = nn.Sequential(
    #     nn.Linear(2, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 2),
    # )
    model = nn.Linear(2, 2)
    # model.weight.data = torch.tensor([[0., 1.], [0., 0.]])
    # model.bias.data = torch.tensor([0., 0.])

    def nll_loss(predictions, targets):
        predictions = (
            predictions.view(-1, predictions.shape[-1])
            .clamp(min=1e-10)
            .log()
            .log_softmax(dim=-1)
        )
        targets = targets.view(-1)
        return nn.functional.nll_loss(predictions, targets)

    trainer_cfg = NEARTrainerConfig(
        lr=1e-4,
        max_seq_len=100,
        n_epochs=10000,
        num_labels=2,
        train_steps=len(datamodule.train),
        loss_callback=nll_loss,
    )
    pl_model = NEARTrainer(model, config=trainer_cfg)
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.n_epochs,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[],
    )
    trainer.fit(pl_model, datamodule.train_dataloader(), datamodule.val_dataloader())
    trainer.validate(pl_model, datamodule.val_dataloader())

    print(trainer.callback_metrics)


if __name__ == "__main__":
    _main()
