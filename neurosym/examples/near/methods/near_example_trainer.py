from dataclasses import dataclass

import torch
from sklearn.metrics import f1_score, hamming_loss
from torch import nn

from .base_trainer import BaseTrainer, BaseTrainerConfig


@dataclass
class NEARTrainerConfig(BaseTrainerConfig):
    max_seq_len: int = 100
    loss_fn: str = "CrossEntropyLoss"
    num_labels: int = -1  # Set Programmatically


class NEARTrainer(BaseTrainer):
    """
    An abstract class that defines the basic functions to
    implement and train a neural module.
    """

    def __init__(self, model: nn.Module, config: NEARTrainerConfig):
        super().__init__(model=model, config=config)
        assert config.num_labels > 0, "Number of labels must be set programmatically"
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
        f1_scores = NEARTrainer.compute_average_f1_score(
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
                    # pylint: disable=not-callable
                    targets = torch.nn.functional.one_hot(
                        targets.squeeze(-1), num_classes=self.config.num_labels
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
                case _:
                    raise NotImplementedError(
                        f"Loss function {self.config.loss_fn} not implemented for seq2seq models"
                    )
        loss = self.loss_fn(predictions, targets)
        return loss

    def _step(self, inputs, outputs, validation=False, **kwargs):
        # pylint: disable=arguments-differ
        del kwargs
        predictions = self.model(inputs)
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


def main():
    import pytorch_lightning as pl

    from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper

    dataset_factory = lambda train_seed: DatasetWrapper(
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
    trainer_cfg = NEARTrainerConfig(
        lr=1e-4,
        max_seq_len=100,
        n_epochs=10000,
        num_labels=2,
        train_steps=len(datamodule.train),
        loss_fn="NLLLoss",
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
    main()
