#!/usr/bin/env python3
"""
Train torch-ecg baselines on the standardized ECG dataset used by NEAR.

This script provides two baselines:
- `ecg_crnn_multi_scopic`: torch_ecg.models.ECG_CRNN configured with multi_scopic CNN
- `multi_scopic_head`: torch_ecg.models.MultiScopicCNN backbone + linear head
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import neurosym as ns
from neurosym.examples.near.metrics_torch_ecg import (
    compute_torch_ecg_classification_metrics,
)


def _import_torch_ecg():
    try:
        from torch_ecg import model_configs, models
    except ImportError as exc:
        raise ImportError(
            "torch-ecg is required for this baseline script. "
            "Install it with `pip install torch-ecg`."
        ) from exc
    return model_configs, models


def _to_channel_sequence_features(x_flat: np.ndarray) -> np.ndarray:
    """
    Convert flattened ECG features (N, 144) to (N, 12, 12) for torch-ecg CNN models.
    """
    if x_flat.ndim != 2 or x_flat.shape[1] != 144:
        raise ValueError(
            f"Expected x shape (N, 144), got {x_flat.shape}. "
            "This baseline targets the standardized ECG feature format."
        )
    return x_flat.reshape(-1, 12, 6, 2).reshape(-1, 12, 12).astype(np.float32)


class MultiScopicClassifier(nn.Module):
    def __init__(self, n_leads: int, out_dim: int):
        super().__init__()
        model_configs, models = _import_torch_ecg()
        self.backbone = models.MultiScopicCNN(
            in_channels=n_leads,
            **model_configs.multi_scopic,
        )
        with torch.no_grad():
            probe = torch.zeros(1, n_leads, 12)
            feature = self.backbone(probe)
            if feature.ndim != 3:
                raise ValueError(f"Unexpected MultiScopicCNN output shape: {feature.shape}")
            feat_dim = int(feature.shape[1])
        self.classifier = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(x)
        pooled = feature.mean(dim=-1)
        return self.classifier(pooled)


def _build_ecg_crnn_multi_scopic(classes: list[str], n_leads: int) -> nn.Module:
    model_configs, models = _import_torch_ecg()
    cfg = copy.deepcopy(model_configs.ECG_CRNN_CONFIG)
    cfg["cnn"]["name"] = "multi_scopic"
    cfg["rnn"]["name"] = "none"
    cfg["attn"]["name"] = "none"
    return models.ECG_CRNN(
        classes=classes,
        n_leads=n_leads,
        config=cfg,
    )


@dataclass
class _FitResult:
    train_time_sec: float
    best_val_metric: float
    best_epoch: int


def _make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    label_mode: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if label_mode == "single":
        y_tensor = torch.tensor(y.reshape(-1), dtype=torch.long)
    else:
        y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    return compute_torch_ecg_classification_metrics(
        pred_arr, true_arr, label_mode=label_mode
    )


def _fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_mode: str,
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> _FitResult:
    model.to(device)
    if label_mode == "single":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    best_val = -float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if label_mode == "single":
                loss = criterion(logits, yb)
            else:
                loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val_loader, label_mode, device)
        val_score = float(val_metrics.get("macro_f1", 0.0))
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"    epoch={epoch + 1:03d}/{n_epochs} "
            f"loss={np.mean(epoch_losses):.6f} "
            f"val_macro_f1={val_score:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return _FitResult(
        train_time_sec=time.time() - start,
        best_val_metric=best_val,
        best_epoch=best_epoch,
    )


def _run_single_model(
    model_name: str,
    out_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    label_mode: str,
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    if model_name == "ecg_crnn_multi_scopic":
        classes = [str(i) for i in range(out_dim)]
        model = _build_ecg_crnn_multi_scopic(classes=classes, n_leads=12)
    elif model_name == "multi_scopic_head":
        model = MultiScopicClassifier(n_leads=12, out_dim=out_dim)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    print(f"\n[model] {model_name}")
    fit_result = _fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_mode=label_mode,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
    )
    test_metrics = _evaluate(model, test_loader, label_mode, device)
    print(
        f"    best_val_macro_f1={fit_result.best_val_metric:.6f} "
        f"(epoch={fit_result.best_epoch + 1}), "
        f"test_macro_f1={float(test_metrics.get('macro_f1', 0.0)):.6f}, "
        f"test_macro_acc={float(test_metrics.get('macro_acc', 0.0)):.6f}"
    )

    return {
        "model": model_name,
        "train_time_sec": fit_result.train_time_sec,
        "best_val_macro_f1": fit_result.best_val_metric,
        "best_epoch": fit_result.best_epoch + 1,
        "test_metrics": test_metrics,
    }


def run_baselines(  # pylint: disable=too-many-arguments
    output_path: str,
    data_dir: str,
    dataset_name: str | None,
    db_dir: str | None,
    working_dir: str | None,
    dataset_kwargs: dict[str, Any] | None,
    max_records: int | None,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
    label_mode: str,
    train_seed: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: str,
    models_to_run: list[str],
) -> dict[str, Any]:
    if label_mode not in {"single", "multi"}:
        raise ValueError("label_mode must be `single` or `multi`.")

    print("=" * 80)
    print("torch-ecg Baseline Benchmark")
    print("=" * 80)
    print(f"data_dir={data_dir}")
    print(f"dataset_name={dataset_name}")
    print(f"db_dir={db_dir}")
    print(f"label_mode={label_mode}")
    print(f"batch_size={batch_size}")
    print(f"epochs={n_epochs}")
    print(f"lr={lr}")
    print(f"device={device}")
    print(f"models={models_to_run}")

    if dataset_name:
        datamodule = ns.datasets.torch_ecg_data_example(
            train_seed=train_seed,
            dataset_name=dataset_name,
            db_dir=db_dir,
            working_dir=working_dir,
            label_mode=label_mode,
            is_regression=(label_mode == "multi"),
            max_records=max_records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            batch_size=batch_size,
            dataset_kwargs=dataset_kwargs,
        )
    else:
        datamodule = ns.datasets.ecg_data_example(
            train_seed=train_seed,
            label_mode=label_mode,
            is_regression=(label_mode == "multi"),
            data_dir=data_dir,
            batch_size=batch_size,
        )

    x_train = _to_channel_sequence_features(datamodule.train.inputs)
    x_val = _to_channel_sequence_features(datamodule.val.inputs)
    x_test = _to_channel_sequence_features(datamodule.test.inputs)
    y_train = datamodule.train.outputs
    y_val = datamodule.val.outputs
    y_test = datamodule.test.outputs

    if label_mode == "single":
        out_dim = int(y_train.max()) + 1
    else:
        out_dim = y_train.shape[-1]

    train_loader = _make_dataloader(x_train, y_train, label_mode, batch_size, shuffle=True)
    val_loader = _make_dataloader(x_val, y_val, label_mode, batch_size, shuffle=False)
    test_loader = _make_dataloader(x_test, y_test, label_mode, batch_size, shuffle=False)

    torch_device = torch.device(device)
    started = time.time()
    reports = []
    for model_name in models_to_run:
        reports.append(
            _run_single_model(
                model_name=model_name,
                out_dim=out_dim,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                label_mode=label_mode,
                n_epochs=n_epochs,
                lr=lr,
                device=torch_device,
            )
        )

    result = {
        "label_mode": label_mode,
        "dataset_name": dataset_name,
        "db_dir": db_dir,
        "data_dir": data_dir,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "input_shape": [int(x_train.shape[1]), int(x_train.shape[2])],
        "out_dim": int(out_dim),
        "batch_size": int(batch_size),
        "epochs": int(n_epochs),
        "lr": float(lr),
        "device": device,
        "total_runtime_sec": float(time.time() - started),
        "reports": reports,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved baseline report to: {output_file}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run torch-ecg baseline models on standardized ECG splits."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ecg_results/baseline_torch_ecg_single.json",
        help="Output JSON file for baseline metrics.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ecg_classification/ecg",
        help="Directory containing standardized ECG .npz splits.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional torch-ecg dataset class name, e.g. CPSC2018. "
        "If set, data is loaded via ns.datasets.torch_ecg_data_example.",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Optional torch-ecg dataset directory when --dataset-name is set.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Optional torch-ecg working/cache directory when --dataset-name is set.",
    )
    parser.add_argument(
        "--dataset-kwargs-json",
        type=str,
        default="{}",
        help="JSON object of additional kwargs for torch-ecg dataset constructor.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Maximum records for torch-ecg dataset loading.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Validation fraction when split is generated.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Test fraction when split is generated.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for generated data splits.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Single-label or multi-label setup.",
    )
    parser.add_argument("--train-seed", type=int, default=0, help="Seed for train ordering.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu, cuda:0.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ecg_crnn_multi_scopic,multi_scopic_head",
        help="Comma-separated model keys: ecg_crnn_multi_scopic,multi_scopic_head",
    )
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        dataset_kwargs = json.loads(args.dataset_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for --dataset-kwargs-json: {args.dataset_kwargs_json}"
        ) from exc
    if not isinstance(dataset_kwargs, dict):
        raise ValueError("--dataset-kwargs-json must decode to a JSON object.")

    run_baselines(
        output_path=args.output,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        db_dir=args.db_dir,
        working_dir=args.working_dir,
        dataset_kwargs=dataset_kwargs,
        max_records=args.max_records if args.max_records > 0 else None,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        label_mode=args.label_mode,
        train_seed=args.train_seed,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        models_to_run=models_to_run,
    )


if __name__ == "__main__":
    main()
