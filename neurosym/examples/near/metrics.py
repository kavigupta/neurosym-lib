"""
Compute evaluation metrics for NEAR predictions.

Warning: This file is an AI-generated amalgamation
of various metric computation utilities.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    roc_curve,
)

from neurosym.utils.documentation import internal_only


@internal_only
def _is_binary_indicator(arr: np.ndarray) -> bool:
    vals = np.unique(arr)
    return np.all(np.isin(vals, [0, 1]))


@internal_only
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@internal_only
def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


@internal_only
def _looks_like_probabilities(arr: np.ndarray) -> bool:
    return np.min(arr) >= 0.0 and np.max(arr) <= 1.0


@internal_only
def threshold_predictions(
    y_true: torch.Tensor, y_scores: torch.Tensor, threshold_type: str = "quantile"
) -> torch.Tensor:
    """
    Apply dynamic thresholding to predictions.
    Args:
        y_true: Ground truth labels
        y_scores: Model prediction scores (logits or probabilities)
        threshold_type: Type of thresholding ('quantile', 'roc', or 'static')
    Returns:
        Binary predictions
    """
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    else:
        y_true = y_true.float()
    if not torch.is_tensor(y_scores):
        y_scores = torch.tensor(y_scores, dtype=torch.float32)
    else:
        y_scores = y_scores.float()

    y_true_flat = y_true.view(-1)
    y_scores_flat = y_scores.view(-1)

    y_true_np = y_true_flat.detach().cpu().numpy()
    y_scores_np = y_scores_flat.detach().cpu().numpy()

    if threshold_type == "quantile":
        prevalence = y_true_np.mean()
        quantile = 1.0 - prevalence
        quantile = np.clip(quantile, 0.0, 1.0)
        threshold = np.quantile(y_scores_np, quantile)
    elif threshold_type == "roc":
        fpr, tpr, thresholds = roc_curve(y_true_np, y_scores_np)
        threshold = thresholds[np.argmax(tpr - fpr)]
    elif threshold_type == "static":
        threshold = 0.5
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    # Use sigmoid to accommodate logits; if scores already in [0,1], sigmoid is monotonic.
    probs = torch.sigmoid(y_scores_flat)
    preds = (probs < threshold).int()
    return preds.view_as(y_scores)


def _is_one_hot(arr: np.ndarray) -> bool:
    if arr.ndim == 0 or not _is_binary_indicator(arr):
        return False
    summed = arr.sum(axis=-1)
    return np.all(np.isclose(summed, 1.0) | np.isclose(summed, 0.0))


@internal_only
def _prepare_multiclass_targets_and_preds(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    probs = y_pred if _looks_like_probabilities(y_pred) else _softmax(y_pred, axis=-1)
    y_pred_cls = probs.argmax(axis=-1).astype(np.int32)

    base_shape = y_pred.shape[:-1]
    if y_true.shape == base_shape:
        y_true_cls = np.asarray(np.rint(y_true), dtype=np.int32)
    elif y_true.shape == base_shape + (1,):
        y_true_cls = np.asarray(np.rint(np.squeeze(y_true, axis=-1)), dtype=np.int32)
    elif y_true.shape == y_pred.shape:
        if _is_one_hot(y_true):
            y_true_cls = y_true.argmax(axis=-1).astype(np.int32)
        else:
            y_true_cls = y_true.argmax(axis=-1).astype(np.int32)
    else:
        raise ValueError(
            f"Shape mismatch for multiclass: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    return y_true_cls.reshape(-1), y_pred_cls.reshape(-1)


@internal_only
def _prepare_multilabel_targets_and_preds(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_bin = y_true.astype(np.int32)

    if _is_binary_indicator(y_pred):
        y_pred_bin = y_pred.astype(np.int32)
    else:
        scores = _sigmoid(y_pred) if not _looks_like_probabilities(y_pred) else y_pred
        y_pred_bin = (scores >= 0.5).astype(np.int32)

    n_labels = y_true_bin.shape[-1]
    y_true_flat = y_true_bin.reshape(-1, n_labels)
    y_pred_flat = y_pred_bin.reshape(-1, n_labels)
    return y_true_flat, y_pred_flat


@internal_only
def _prepare_binary_targets_and_preds(
    y_true: np.ndarray, y_pred: np.ndarray, threshold_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_flat = y_true.reshape(-1)
    y_true_bin = np.asarray(np.rint(y_true_flat), dtype=np.int32)

    if _is_binary_indicator(y_pred):
        y_pred_bin = y_pred.reshape(-1).astype(np.int32)
    else:
        preds = threshold_predictions(
            torch.tensor(y_true_bin, dtype=torch.float32),
            torch.tensor(y_pred.reshape(-1), dtype=torch.float32),
            threshold_type=threshold_type,
        )
        y_pred_bin = preds.numpy().astype(np.int32)

    return y_true_bin, y_pred_bin


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metric_name: str = "all",
    threshold_type: str = "quantile",
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for predictions across binary, multiclass, and multilabel tasks.
    Args:
        predictions: Model prediction scores/logits/probabilities.
        ground_truth: Ground truth labels.
        metric_name: Optional selector to compute a subset of metrics.
        threshold_type: Thresholding strategy used for binary problems.
    Returns:
        Dictionary containing weighted/macro/per-class F1, hamming accuracy (or accuracy),
        a classification report (when metric_name == "all"), and the inferred task type.
    """
    y_true_np = np.asarray(ground_truth)
    y_pred_np = np.asarray(predictions)

    # Detect task type.
    same_shape = y_true_np.shape == y_pred_np.shape
    has_label_axis = y_pred_np.ndim >= 2 and y_pred_np.shape[-1] > 1
    is_multilabel = (
        same_shape
        and y_pred_np.ndim >= 2
        and _is_binary_indicator(y_true_np)
        and not _is_one_hot(y_true_np)
    )
    has_class_axis = has_label_axis and not is_multilabel

    task_type = "binary"
    try:
        if is_multilabel:
            task_type = "multilabel"
            y_true_norm, y_pred_norm = _prepare_multilabel_targets_and_preds(
                y_true_np, y_pred_np
            )
        elif has_class_axis:
            task_type = "multiclass"
            y_true_norm, y_pred_norm = _prepare_multiclass_targets_and_preds(
                y_true_np, y_pred_np
            )
        else:
            task_type = "binary"
            y_true_norm, y_pred_norm = _prepare_binary_targets_and_preds(
                y_true_np, y_pred_np, threshold_type
            )
    except ValueError as err:
        print(f"Error preparing predictions for metrics: {err}")
        y_true_norm = y_pred_norm = None

    weighted_avg_f1 = 0.0
    unweighted_avg_f1 = 0.0
    all_f1 = []
    hamming_accuracy = 0.0
    report: Dict[str, Any] = {}

    try:
        if y_true_norm is None or y_pred_norm is None:
            raise ValueError("Prepared targets/predictions were None.")

        if task_type == "multilabel":
            if metric_name in ("all", "weighted_f1"):
                weighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="weighted", zero_division=0
                )
            if metric_name in ("all", "unweighted_f1"):
                unweighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="macro", zero_division=0
                )
            if metric_name == "all":
                all_f1 = f1_score(
                    y_true_norm, y_pred_norm, average=None, zero_division=0
                )
            if metric_name in ("all", "hamming_accuracy"):
                hamming_accuracy = 1.0 - hamming_loss(y_true_norm, y_pred_norm)

            if metric_name == "all":
                report = classification_report(
                    y_true_norm, y_pred_norm, output_dict=True, zero_division=0
                )

        else:  # binary or multiclass
            if metric_name in ("all", "weighted_f1"):
                weighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="weighted", zero_division=0
                )
            if metric_name in ("all", "unweighted_f1"):
                unweighted_avg_f1 = f1_score(
                    y_true_norm, y_pred_norm, average="macro", zero_division=0
                )
            if metric_name == "all":
                all_f1 = f1_score(
                    y_true_norm, y_pred_norm, average=None, zero_division=0
                )

            if metric_name in ("all", "hamming_accuracy"):
                hamming_accuracy = accuracy_score(y_true_norm, y_pred_norm)

            if metric_name == "all":
                report = classification_report(
                    y_true_norm, y_pred_norm, output_dict=True, zero_division=0
                )

    except ValueError as err:
        print(f"Error computing metrics: {err}")

    return {
        "f1_score": float(weighted_avg_f1),
        "unweighted_f1": float(unweighted_avg_f1),
        "all_f1s": np.asarray(all_f1).tolist() if np.ndim(all_f1) else all_f1,
        "hamming_accuracy": float(hamming_accuracy),
        "report": report,
        "task_type": task_type,
    }
