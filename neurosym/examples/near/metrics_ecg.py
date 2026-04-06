"""ECG evaluation metrics following the standard protocol.

Primary metric: **macro AUC** (threshold-free, computed on soft outputs).
Secondary: macro F1, per-class AUC, bootstrap confidence intervals.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _import_sklearn_metrics():
    try:
        from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        return roc_auc_score, f1_score, precision_score, recall_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ECG metrics. "
            "Install it with `pip install scikit-learn`."
        ) from exc


def compute_ecg_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    label_mode: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute ECG evaluation metrics.

    :param predictions: Soft model outputs of shape ``(N, C)``.
    :param ground_truth: Labels. Integer class indices for single-label or
        multi-hot binary vectors for multi-label.
    :param label_mode: ``'single'`` or ``'multi'``.
    :param threshold: Threshold for F1/precision/recall binarization.
    :return: Dict with macro_auc (primary), macro_f1, per-class metrics, etc.
    """
    roc_auc_score, f1_score, precision_score, recall_score = _import_sklearn_metrics()

    y_pred = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(ground_truth)

    if y_pred.ndim != 2:
        raise ValueError(f"Expected predictions shape (N, C), got {y_pred.shape}")

    n_classes = y_pred.shape[1]

    # Convert to multi-hot for unified evaluation
    if label_mode == "single":
        y_true_idx = y_true.reshape(-1).astype(np.int64)
        y_true_bin = np.zeros((len(y_true_idx), n_classes), dtype=np.int32)
        y_true_bin[np.arange(len(y_true_idx)), y_true_idx] = 1
        # Convert logits to probabilities
        y_pred_prob = np.exp(y_pred) / np.exp(y_pred).sum(axis=1, keepdims=True)
    elif label_mode == "multi":
        y_true_bin = y_true.astype(np.int32)
        if y_pred.min() < 0.0 or y_pred.max() > 1.0:
            y_pred_prob = _sigmoid(y_pred)
        else:
            y_pred_prob = y_pred.copy()
    else:
        raise ValueError("label_mode must be 'single' or 'multi'")

    # --- Primary metric: Macro AUC ---
    per_class_auc = []
    for c in range(n_classes):
        if y_true_bin[:, c].sum() > 0 and y_true_bin[:, c].sum() < len(y_true_bin):
            auc_c = roc_auc_score(y_true_bin[:, c], y_pred_prob[:, c])
            per_class_auc.append(auc_c)
        else:
            per_class_auc.append(float("nan"))

    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # --- Secondary: F1 / precision / recall ---
    if label_mode == "single":
        # For single-label, use argmax predictions
        y_pred_labels = y_pred_prob.argmax(axis=1)
        y_true_labels = y_true.reshape(-1).astype(np.int64)
        macro_f1 = float(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        )
        macro_prec = float(
            precision_score(
                y_true_labels, y_pred_labels, average="macro", zero_division=0
            )
        )
        macro_recall = float(
            recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        )
        per_class_f1 = f1_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        ).tolist()
    else:
        # For multi-label, use threshold-based binarization
        y_pred_bin = (y_pred_prob >= threshold).astype(np.int32)
        macro_f1 = float(
            f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        )
        macro_prec = float(
            precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        )
        macro_recall = float(
            recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        )
        per_class_f1 = f1_score(
            y_true_bin, y_pred_bin, average=None, zero_division=0
        ).tolist()

    metrics = {
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "macro_prec": macro_prec,
        "macro_recall": macro_recall,
        "per_class_auc": per_class_auc,
        "per_class_f1": per_class_f1,
        "label_mode": label_mode,
        # Backward-compatible aliases
        "macro_auroc": macro_auc,
        "f1_score": macro_f1,
        "precision": macro_prec,
        "recall": macro_recall,
    }
    return metrics


def bootstrap_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    label_mode: str,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for macro AUC on the test set.

    Follows the standard ECG evaluation protocol: draw bootstrap samples from the
    test set, requiring each sample to have at least one positive per class.

    :param predictions: Soft model outputs of shape ``(N, C)``.
    :param ground_truth: Labels.
    :param label_mode: ``'single'`` or ``'multi'``.
    :param n_samples: Number of bootstrap samples.
    :param seed: Random seed for reproducibility.
    :return: Dict with mean, std, ci_5, ci_95 for macro AUC.
    """
    y_pred = np.asarray(predictions, dtype=np.float64)
    y_true = np.asarray(ground_truth)
    n = len(y_pred)

    if label_mode == "single":
        n_classes = y_pred.shape[1]
        y_true_bin = np.zeros((n, n_classes), dtype=np.int32)
        y_true_bin[np.arange(n), y_true.reshape(-1).astype(np.int64)] = 1
    else:
        y_true_bin = y_true.astype(np.int32)

    rng = np.random.RandomState(seed)
    auc_scores = []

    attempts = 0
    max_attempts = n_samples * 10

    while len(auc_scores) < n_samples and attempts < max_attempts:
        attempts += 1
        indices = rng.randint(0, n, size=n)
        sample_y = y_true_bin[indices]
        # Require at least one positive per class
        if not all(sample_y[:, c].sum() > 0 for c in range(y_true_bin.shape[1])):
            continue
        # Require at least one negative per class
        if not all(
            sample_y[:, c].sum() < len(sample_y) for c in range(y_true_bin.shape[1])
        ):
            continue

        sample_metrics = compute_ecg_metrics(
            y_pred[indices], y_true[indices], label_mode=label_mode
        )
        auc_scores.append(sample_metrics["macro_auc"])

    auc_arr = np.array(auc_scores)
    return {
        "mean_auc": float(auc_arr.mean()) if len(auc_arr) > 0 else 0.0,
        "std_auc": float(auc_arr.std()) if len(auc_arr) > 0 else 0.0,
        "ci_5": float(np.percentile(auc_arr, 5)) if len(auc_arr) > 0 else 0.0,
        "ci_95": float(np.percentile(auc_arr, 95)) if len(auc_arr) > 0 else 0.0,
        "n_valid_samples": len(auc_scores),
    }
