from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


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
class NEARTrainerConfig:
    """
    Configuration class for training a NEAR model.

    :param lr: Learning rate for the optimizer (default: 1e-4)
    :param weight_decay: Weight decay for the optimizer (default: 0.0)
    :param n_epochs: Number of epochs to train for (default: 10)
    :param seed: Random seed for reproducibility (default: 44)
    :param scheduler: Learning rate scheduler to use (default: "cosine")
    :param accelerator: Accelerator to use for training (default: "cpu")
    :param optimizer: Optimizer to use (default: torch.optim.Adam)
    :param loss_callback: Loss function to use (default: :py:func:`neurosym.examples.near.classification_mse_loss`)
    """

    lr: float = 1e-4
    weight_decay: float = 0.0
    n_epochs: int = 10
    seed: int = 44
    scheduler: str = "cosine"
    accelerator: str = "cpu"
    # This is necessary for some unfathomable reason. I assume that torch.optim.Adam
    # is overwritten by pytorch-lightning, so we need to pass it as a lambda to avoid
    # passing the outdated version.
    #
    # pylint: disable=unnecessary-lambda
    optimizer: str = lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs)

    loss_callback: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        classification_mse_loss
    )
