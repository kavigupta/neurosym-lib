#!/usr/bin/env python3
"""
Baseline MLP for ECG classification.

3-layer MLP (256-256-256) matching the ECG paper's MLP baseline.
Evaluates with macro AUC (primary) and macro F1 (secondary).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import neurosym as ns
from neurosym.examples.near.metrics_ecg import (
    bootstrap_metrics,
    compute_ecg_metrics,
)


def build_model(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def train_epoch(model, loader, criterion, optimizer, device, label_mode: str) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        if label_mode == "single":
            yb = yb.view(-1).long()
        else:
            yb = yb.float()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.shape[0]
        total += xb.shape[0]
    return running_loss / max(total, 1)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG baseline MLP")
    parser.add_argument(
        "--label-mode",
        choices=["single", "multi"],
        default="single",
        help="Label mode (single=filtered, multi=multi-hot)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ecg",
        help="Path to ECG data directory",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON path for metrics output",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_regression = args.label_mode == "multi"
    datamodule = ns.datasets.ecg_data_example(
        train_seed=args.seed,
        label_mode=args.label_mode,
        is_regression=is_regression,
        data_dir=args.data_dir,
        normalize=True,
        batch_size=args.batch_size,
    )

    x_train = datamodule.train.inputs
    y_train = datamodule.train.outputs
    x_val = datamodule.val.inputs
    y_val = datamodule.val.outputs
    x_test = datamodule.test.inputs
    y_test = datamodule.test.outputs

    if args.label_mode == "single":
        y_train = y_train.reshape(-1).astype(np.int64)
        y_val = y_val.reshape(-1).astype(np.int64)
        y_test = y_test.reshape(-1).astype(np.int64)
        output_dim = int(y_train.max()) + 1
        criterion = nn.CrossEntropyLoss()
    else:
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)
        output_dim = y_train.shape[-1]
        criterion = nn.BCEWithLogitsLoss()

    input_dim = x_train.shape[-1]
    model = build_model(input_dim, output_dim, args.hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print("=" * 80)
    print("ECG Baseline MLP")
    print("=" * 80)
    print(f"Label mode: {args.label_mode}")
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)

    best_val_auc = -1.0
    best_model_state = None

    start_time = time.time()
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, train_loader, criterion, optimizer, args.device, args.label_mode
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_preds, val_labels = predict(model, val_loader, args.device)
            val_metrics = compute_ecg_metrics(
                val_preds, val_labels, label_mode=args.label_mode
            )
            val_auc = val_metrics["macro_auc"]
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs}: "
                f"loss={loss:.6f}, val_auc={val_auc:.4f}"
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - start_time

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(args.device)

    preds, labels = predict(model, test_loader, args.device)
    metrics = compute_ecg_metrics(preds, labels, label_mode=args.label_mode)

    ci = bootstrap_metrics(preds, labels, label_mode=args.label_mode)

    print("=" * 80)
    print("RESULTS (Test Set - Fold 10)")
    print("=" * 80)
    print(f"Macro AUC: {metrics.get('macro_auc', 0.0):.6f}")
    print(f"Macro F1: {metrics.get('macro_f1', 0.0):.6f}")
    print(f"Macro Precision: {metrics.get('macro_prec', 0.0):.6f}")
    print(f"Macro Recall: {metrics.get('macro_recall', 0.0):.6f}")
    print(
        f"Bootstrap AUC: {ci['mean_auc']:.3f} "
        f"({ci['ci_5']:.3f} - {ci['ci_95']:.3f})"
    )
    print("=" * 80)

    output_path = args.output
    if not output_path:
        output_path = f"outputs/ecg_results/baseline_nn_{args.label_mode}.json"

    result = {
        "model": "mlp",
        "label_mode": args.label_mode,
        "train_time": train_time,
        "best_val_auc": best_val_auc,
        "report": metrics,
        "bootstrap": ci,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
