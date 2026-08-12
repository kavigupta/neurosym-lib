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
    roc_auc_score,
    roc_curve,
)

from neurosym.utils.documentation import internal_only
from neurosym.utils.logging import log


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
def _safe_macro_auroc(y_true_bin: np.ndarray, scores: np.ndarray) -> Tuple[float, list]:
    """
    Compute macro-averaged AUROC over label columns, skipping degenerate
    columns (those with only one class present in ``y_true_bin``).

    :param y_true_bin: Binary indicator ground truth, shape (N, C).
    :param scores: Prediction scores/probabilities, same shape as ``y_true_bin``.
    :returns: ``(macro_auroc, per_class_auroc)``. ``macro_auroc`` is ``nan`` if
        no column has both classes present.
    """
    y_true_bin = np.asarray(y_true_bin)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin[:, None]
        scores = scores[:, None]
    per_class = []
    for c in range(y_true_bin.shape[-1]):
        col_true = y_true_bin[:, c]
        if np.unique(col_true).size < 2:
            per_class.append(float("nan"))
            continue
        try:
            per_class.append(float(roc_auc_score(col_true, scores[:, c])))
        except ValueError as err:
            log(f"AUROC undefined for class {c}: {err}")
            per_class.append(float("nan"))
    valid = [v for v in per_class if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_class


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


def compute_metrics(  # pylint: disable=too-many-branches,too-many-statements
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metric_name: str = "all",
    threshold_type: str = "quantile",
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for predictions across binary, multiclass, and multilabel tasks.

    :param predictions: Model prediction scores/logits/probabilities.
    :type predictions: np.ndarray
    :param ground_truth: Ground truth labels.
    :type ground_truth: np.ndarray
    :param metric_name: Optional selector to compute a subset of metrics.
    :type metric_name: str
    :param threshold_type: Thresholding strategy used for binary problems.
    :type threshold_type: str
    :returns: Dictionary containing weighted/macro/per-class F1, hamming accuracy (or accuracy),
        a classification report (when metric_name == "all"), and the inferred task type.
    :rtype: Dict[str, Any]
    """
    y_true_np = np.asarray(ground_truth)
    y_pred_np = np.asarray(predictions)

    regression_metrics: Dict[str, Any] = {}
    if y_true_np.dtype.kind == "f":
        diff = y_true_np - y_pred_np
        # L1 distance: mean of L1 norms (sum of absolute differences)
        # L2 distance: mean of L2 norms (sqrt of sum of squares)
        l1_dist = np.mean(np.sum(np.abs(diff), axis=-1))
        l2_dist = np.mean(np.sum(diff**2, axis=-1))
        regression_metrics = {
            "neg_l1_dist": -float(l1_dist),
            "neg_l2_dist": -float(l2_dist),
        }
        # A float matrix that is a pure 0/1 indicator with a label axis is a
        # multi-label classification target (BCE stores multi-hot as float),
        # NOT a regression target. Fall through to classification metrics so
        # f1/hamming/AUROC are reachable; genuine continuous regression
        # targets are not binary indicators and return here unchanged.
        is_multilabel_float = (
            y_true_np.ndim >= 2
            and y_true_np.shape[-1] > 1
            and _is_binary_indicator(y_true_np)
            and not _is_one_hot(y_true_np)
        )
        if not is_multilabel_float:
            return regression_metrics

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
        log(f"Error preparing predictions for metrics: {err}")
        y_true_norm = y_pred_norm = None

    weighted_avg_f1 = 0.0
    unweighted_avg_f1 = 0.0
    all_f1 = []
    hamming_accuracy = 0.0
    macro_auroc = float("nan")
    per_class_auroc: list = []
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

            if metric_name in ("all", "macro_auroc"):
                n_labels = y_true_norm.shape[-1]
                ml_scores = y_pred_np.reshape(-1, n_labels).astype(np.float64)
                if not _looks_like_probabilities(ml_scores):
                    ml_scores = _sigmoid(ml_scores)
                macro_auroc, per_class_auroc = _safe_macro_auroc(y_true_norm, ml_scores)

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
        log(f"Error computing metrics: {err}")

    result: Dict[str, Any] = {
        "f1_score": float(weighted_avg_f1),
        "unweighted_f1": float(unweighted_avg_f1),
        "all_f1s": np.asarray(all_f1).tolist() if np.ndim(all_f1) else all_f1,
        "hamming_accuracy": float(hamming_accuracy),
        "macro_auroc": float(macro_auroc),
        "per_class_auroc": per_class_auroc,
        "report": report,
        "task_type": task_type,
    }
    # Keep regression distances available for multi-hot float labels so that
    # existing callers requesting neg_l2_dist/neg_l1_dist still work.
    result.update(regression_metrics)
    return result
