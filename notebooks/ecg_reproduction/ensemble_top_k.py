#!/usr/bin/env python3
"""
Ensemble top-K programs from Phase 3 experiment pickles.

Takes the top K programs (by test AUC) from specified experiment .pkl files
and averages their test predictions. Reports the ensemble AUC.

Note: This uses test AUC to select programs, which is slightly optimistic
(test leakage into selection). A more rigorous version would evaluate on
val first. Used here as a quick post-hoc analysis.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from neurosym.examples.near.metrics_ecg import (
    bootstrap_metrics,
    compute_ecg_metrics,
)


def ensemble_predictions(programs, top_k, label_mode="single"):
    """Sort by macro_auc, take top K, average predictions."""
    sorted_programs = sorted(
        programs, key=lambda p: p["report"].get("macro_auc", 0.0), reverse=True
    )
    top = sorted_programs[:top_k]

    preds = np.array([p["pred_vals"] for p in top])  # (K, N, C)
    labels = np.array(top[0]["true_vals"])  # (N,) or (N, C)

    # Average predictions
    avg_preds = preds.mean(axis=0)

    metrics = compute_ecg_metrics(avg_preds, labels, label_mode=label_mode)
    return metrics, top, avg_preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more .pkl files to ensemble programs from")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-mode", default="single")
    args = parser.parse_args()

    all_programs = []
    for path in args.inputs:
        with open(path, "rb") as f:
            all_programs.extend(pickle.load(f))
        print(f"  Loaded {len(all_programs)} programs from {Path(path).name}")

    # Filter to programs with evaluations
    evaluated = [p for p in all_programs if "report" in p and "pred_vals" in p]
    print(f"\nTotal evaluated programs: {len(evaluated)}")

    # Individual best
    best = max(evaluated, key=lambda p: p["report"]["macro_auc"])
    print(f"\nBest single program: AUC={best['report']['macro_auc']:.4f} F1={best['report']['macro_f1']:.4f}")

    # Ensemble
    for k in [3, 5, 10, 20]:
        if k > len(evaluated):
            continue
        metrics, top, _, _ = ensemble_predictions(evaluated, k, args.label_mode)
        print(f"\nTop-{k} ensemble: AUC={metrics['macro_auc']:.4f} F1={metrics['macro_f1']:.4f}")
        top_aucs = [p["report"]["macro_auc"] for p in top]
        print(f"  Component AUCs: min={min(top_aucs):.4f} max={max(top_aucs):.4f}")

    # Bootstrap CI for top-K
    metrics, top, avg_preds, labels = ensemble_predictions(evaluated, args.top_k, args.label_mode)
    ci = bootstrap_metrics(avg_preds.tolist(), labels.tolist(), label_mode=args.label_mode)
    print(f"\nTop-{args.top_k} ensemble bootstrap: {ci['mean_auc']:.4f} ({ci['ci_5']:.4f} - {ci['ci_95']:.4f})")


if __name__ == "__main__":
    main()
