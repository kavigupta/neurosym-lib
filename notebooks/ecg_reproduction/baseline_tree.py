#!/usr/bin/env python3
"""
Tree-based baseline for ECG classification.

Supports single-label (argmax) and multi-label (multi-hot) targets using the
same standardized dataset splits as the NEAR reproduction.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import neurosym as ns
from neurosym.examples.near.metrics_torch_ecg import (
    compute_torch_ecg_classification_metrics,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


def build_estimator(model: str, seed: int, n_estimators: int, max_depth: int | None):
    if model == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG tree baseline")
    parser.add_argument(
        "--model",
        choices=["decision_tree", "random_forest"],
        default="decision_tree",
        help="Which tree model to use",
    )
    parser.add_argument(
        "--label-mode",
        choices=["single", "multi"],
        default="single",
        help="Label mode to train against (single=argmax, multi=multi-hot)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ecg_classification/ecg",
        help="Path to the standardized ECG data directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON path for metrics output",
    )
    args = parser.parse_args()

    is_regression = args.label_mode == "multi"
    datamodule = ns.datasets.ecg_data_example(
        train_seed=args.seed,
        label_mode=args.label_mode,
        is_regression=is_regression,
        data_dir=args.data_dir,
        batch_size=1024,
    )

    x_train = datamodule.train.inputs
    y_train = datamodule.train.outputs
    x_test = datamodule.test.inputs
    y_test = datamodule.test.outputs

    if args.label_mode == "single":
        y_train = y_train.reshape(-1).astype(np.int64)
        y_test = y_test.reshape(-1).astype(np.int64)
    else:
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

    estimator = build_estimator(args.model, args.seed, args.n_estimators, args.max_depth)
    if args.label_mode == "multi":
        estimator = MultiOutputClassifier(estimator)

    print("=" * 80)
    print("ECG Tree Baseline")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Label mode: {args.label_mode}")
    print(f"Train size: {len(x_train)}")
    print(f"Test size: {len(x_test)}")
    print("=" * 80)

    start_time = time.time()
    estimator.fit(x_train, y_train)
    train_time = time.time() - start_time

    if args.label_mode == "single":
        probs = estimator.predict_proba(x_test)
        preds = np.asarray(probs)
    else:
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(x_test)
            if isinstance(probs, list):
                preds = np.column_stack(
                    [
                        p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.reshape(-1)
                        for p in probs
                    ]
                ).astype(np.float32)
            else:
                preds = np.asarray(probs, dtype=np.float32)
        else:
            preds = estimator.predict(x_test).astype(np.float32)

    metrics = compute_torch_ecg_classification_metrics(
        preds, y_test, label_mode=args.label_mode
    )

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Macro F1: {metrics.get('macro_f1', 0.0):.6f}")
    print(f"Macro Accuracy: {metrics.get('macro_acc', 0.0):.6f}")
    print(f"Macro Precision: {metrics.get('macro_prec', 0.0):.6f}")
    print(f"Macro Recall: {metrics.get('macro_sens', 0.0):.6f}")
    print("=" * 80)

    output_path = args.output
    if not output_path:
        output_path = f"outputs/ecg_results/baseline_tree_{args.model}_{args.label_mode}.json"

    result = {
        "model": args.model,
        "label_mode": args.label_mode,
        "train_time": train_time,
        "report": metrics,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
