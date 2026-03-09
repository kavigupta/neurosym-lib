# pylint: skip-file
from dataclasses import dataclass

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

from .near_example_trainer import NEARTrainerConfig


@dataclass
class ECGTrainerConfig(NEARTrainerConfig):
    """
    Configuration class for training ECG models with NEAR.

    Extends NEARTrainerConfig with ECG-specific parameters.

    :param num_labels: Number of output labels (required)
    :param is_regression: Whether this is a regression task (default: False)
    :param is_multilabel: Whether this is a multi-label classification task (default: False)
    """

    num_labels: int = -1  # Must be set programmatically
    is_regression: bool = False
    is_multilabel: bool = False


def ecg_cross_entropy_loss(predictions, targets):
    """
    Cross-entropy loss for ECG classification tasks.

    :param predictions: Model predictions. Shape: ``(..., num_classes)``
    :param targets: Target labels as one-hot vectors. Shape: ``(..., num_classes)``
    """
    # predictions are logits, targets are one-hot
    predictions = predictions.view(-1, predictions.shape[-1])
    if targets.ndim == predictions.ndim:
        targets = targets.view(-1, targets.shape[-1])
        target_indices = targets.argmax(dim=-1)
    else:
        target_indices = targets.view(-1).long()
    return nn.functional.cross_entropy(predictions, target_indices)


def compute_ecg_metrics(predictions, targets, num_labels):
    """
    Compute ECG-specific metrics: AUROC, balanced accuracy, and F1 scores.

    :param predictions: Model predictions (logits or probabilities)
    :param targets: Target labels (one-hot encoded)
    :param num_labels: Number of classes
    :return: Dictionary of metrics
    """
    # Convert to numpy for sklearn metrics
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Get probabilities from logits
    if predictions_np.min() < 0 or predictions_np.max() > 1:
        predictions_probs = torch.softmax(
            torch.from_numpy(predictions_np), dim=-1
        ).numpy()
    else:
        predictions_probs = predictions_np

    # Support both one-hot and integer targets.
    if targets_np.ndim == predictions_probs.ndim:
        target_classes = targets_np.argmax(axis=-1)
        target_one_hot = targets_np
    else:
        target_classes = targets_np.reshape(-1).astype("int64")
        target_one_hot = (
            torch.nn.functional.one_hot(
                torch.from_numpy(target_classes), num_classes=num_labels
            )
            .numpy()
            .astype("float32")
        )

    # Get predicted classes
    pred_classes = predictions_probs.argmax(axis=-1)

    metrics = {}

    # AUROC
    try:
        metrics["auroc"] = roc_auc_score(
            y_true=target_one_hot,
            y_score=predictions_probs,
            multi_class="ovr",
        )
    except ValueError:
        metrics["auroc"] = 0.0

    # Balanced accuracy
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true=target_classes, y_pred=pred_classes
    )

    # Hamming accuracy (exact match)
    metrics["hamming_accuracy"] = (pred_classes == target_classes).mean()

    # F1 scores
    if num_labels > 2:
        metrics["weighted_avg_f1"] = f1_score(
            target_classes, pred_classes, average="weighted", zero_division=0
        )
        metrics["unweighted_avg_f1"] = f1_score(
            target_classes, pred_classes, average="macro", zero_division=0
        )
    else:
        metrics["f1"] = f1_score(
            target_classes, pred_classes, average="binary", zero_division=0
        )

    return metrics
