"""Metric helpers that use torch-ecg's classification metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _import_metrics_from_confusion_matrix():
    try:
        from torch_ecg.utils.utils_metrics import metrics_from_confusion_matrix
    except ImportError as exc:
        raise ImportError(
            "torch-ecg is required for ECG metrics. Install it with "
            "`pip install torch-ecg`."
        ) from exc
    return metrics_from_confusion_matrix


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_float(value: Any) -> float:
    return float(np.asarray(value).item())


def compute_torch_ecg_classification_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    label_mode: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute ECG metrics using torch-ecg's metric implementation.

    Parameters
    ----------
    predictions:
        Model outputs. For ``label_mode='single'``, logits/probabilities of
        shape ``(n_samples, n_classes)``. For ``label_mode='multi'``, logits or
        probabilities of shape ``(n_samples, n_classes)``.
    ground_truth:
        Labels. Integer class indices for single-label or multi-hot binary
        vectors for multi-label.
    label_mode:
        ``'single'`` or ``'multi'``.
    threshold:
        Threshold for multi-label binarization.
    """
    metric_fn = _import_metrics_from_confusion_matrix()

    y_pred = np.asarray(predictions)
    y_true = np.asarray(ground_truth)

    if label_mode == "single":
        if y_pred.ndim != 2:
            raise ValueError(
                f"Expected single-label predictions shape (N, C), got {y_pred.shape}"
            )
        n_classes_pred = int(y_pred.shape[-1])
        y_true_idx = y_true.reshape(-1).astype(np.int64)
        y_pred_idx = y_pred.argmax(axis=-1).astype(np.int64)
        # Ensure num_classes covers both true labels and prediction columns.
        n_classes = max(n_classes_pred, int(y_true_idx.max()) + 1)
        raw = metric_fn(
            labels=y_true_idx,
            outputs=y_pred_idx,
            num_classes=n_classes,
            fillna=0.0,
        )
        support = float(y_true_idx.shape[0])
        task_type = "multiclass"
    elif label_mode == "multi":
        if y_pred.ndim != 2:
            raise ValueError(
                f"Expected multi-label predictions shape (N, C), got {y_pred.shape}"
            )
        y_true_bin = y_true.astype(np.int32)
        if y_true_bin.ndim != 2:
            raise ValueError(
                f"Expected multi-label targets shape (N, C), got {y_true_bin.shape}"
            )
        if y_pred.min() < 0.0 or y_pred.max() > 1.0:
            y_pred_scores = _sigmoid(y_pred)
        else:
            y_pred_scores = y_pred
        raw = metric_fn(
            labels=y_true_bin,
            outputs=y_pred_scores,
            num_classes=int(y_true_bin.shape[-1]),
            thr=threshold,
            fillna=0.0,
        )
        support = float(y_true_bin.sum())
        task_type = "multilabel"
    else:
        raise ValueError("label_mode must be 'single' or 'multi'")

    metrics = {
        "macro_prec": _as_float(raw["macro_prec"]),
        "macro_sens": _as_float(raw["macro_sens"]),
        "macro_f1": _as_float(raw["macro_f1"]),
        "macro_acc": _as_float(raw["macro_acc"]),
        "macro_auroc": _as_float(raw["macro_auroc"]),
        "macro_auprc": _as_float(raw["macro_auprc"]),
        "prec": np.asarray(raw["prec"]).tolist(),
        "sens": np.asarray(raw["sens"]).tolist(),
        "f1": np.asarray(raw["f1"]).tolist(),
        "acc": np.asarray(raw["acc"]).tolist(),
        "auroc": np.asarray(raw["auroc"]).tolist(),
        "auprc": np.asarray(raw["auprc"]).tolist(),
        "task_type": task_type,
    }

    # Backward-compatible aliases used by existing scripts/notebooks.
    metrics.update(
        {
            "precision": metrics["macro_prec"],
            "recall": metrics["macro_sens"],
            "f1_score": metrics["macro_f1"],
            "unweighted_f1": metrics["macro_f1"],
            "hamming_accuracy": metrics["macro_acc"],
            "all_f1s": metrics["f1"],
            "report": {
                "macro avg": {
                    "precision": metrics["macro_prec"],
                    "recall": metrics["macro_sens"],
                    "f1-score": metrics["macro_f1"],
                    "support": support,
                }
            },
        }
    )
    return metrics
