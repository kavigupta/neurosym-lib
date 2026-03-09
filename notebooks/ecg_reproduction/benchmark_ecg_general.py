#!/usr/bin/env python3
"""
General ECG benchmark script using torch-ecg datasets.

This mirrors the existing ECG benchmark flow while using
`ns.datasets.torch_ecg_data_example` to build train/val/test splits from a
torch-ecg database class.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

import neurosym as ns
from neurosym.examples import near
from neurosym.examples.near.metrics_torch_ecg import (
    compute_torch_ecg_classification_metrics,
)
from neurosym.types.type import ArrowType, AtomicType


class ChannelSelector(torch.nn.Module):
    def __init__(self, num_channels: int = 12):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, environment=()):
        del environment
        batch_shape = x.shape[:-1]
        logits = self.logits.expand(*batch_shape, -1)
        return torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)


class ChannelHoleFiller(near.NeuralHoleFiller):
    def __init__(self, num_channels: int = 12):
        self.num_channels = num_channels

    def initialize_module(self, type_with_environment):
        typ = type_with_environment.typ

        def is_channel(t):
            return isinstance(t, AtomicType) and t.name == "channel"

        if isinstance(typ, ArrowType):
            out = typ.output_type
            if isinstance(out, ArrowType):
                out = out.output_type
            if is_channel(out):
                return ChannelSelector(self.num_channels)
        if is_channel(typ):
            return ChannelSelector(self.num_channels)
        return None


def ce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.view(-1).long()
    predictions = predictions.view(-1, predictions.shape[-1])
    return torch.nn.functional.cross_entropy(predictions, targets)


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1, targets.shape[-1]).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)


def eval_program(module, feature_data, labels, label_mode: str) -> tuple:
    predictions = (
        module(torch.tensor(feature_data), environment=()).detach().numpy()
    ).reshape(feature_data.shape[0], -1)
    metrics = compute_torch_ecg_classification_metrics(
        predictions, labels, label_mode=label_mode
    )
    return metrics, predictions


def run_experiment(  # pylint: disable=too-many-arguments,too-many-locals
    dataset_name: str,
    output_path: str = "outputs/ecg_results/reproduction_general.pkl",
    db_dir: str | None = None,
    working_dir: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
    max_records: int | None = 5000,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_seed: int = 42,
    normalize_per_split: bool = True,
    num_programs: int = 20,
    hidden_dim: int = 16,
    neural_hidden_size: int = 16,
    batch_size: int = 1024,
    n_epochs: int = 30,
    final_n_epochs: int = 40,
    lr: float = 1e-4,
    structural_cost_penalty: float = 0.01,
    max_depth: int = 10,
    train_seed: int = 0,
    device: str = "cuda:0",
    label_mode: str = "single",
) -> List[Dict[str, Any]]:
    print("=" * 80)
    print("General ECG NEAR Experiment - torch-ecg backend")
    print("=" * 80)
    print("Configuration:")
    print(f"  Dataset name: {dataset_name}")
    print(f"  Output path: {output_path}")
    print(f"  DB dir: {db_dir}")
    print(f"  Working dir: {working_dir}")
    print(f"  Max records: {max_records}")
    print(f"  Val fraction: {val_fraction}")
    print(f"  Test fraction: {test_fraction}")
    print(f"  Split seed: {split_seed}")
    print(f"  Normalize per split: {normalize_per_split}")
    print(f"  Number of programs: {num_programs}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Neural hidden size: {neural_hidden_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs (search): {n_epochs}")
    print(f"  Epochs (final): {final_n_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Structural cost weight: {structural_cost_penalty}")
    print(f"  Max depth: {max_depth}")
    print(f"  Device: {device}")
    print(f"  Label mode: {label_mode}")
    print("=" * 80)

    print("\n[1/5] Loading ECG dataset from torch-ecg...")
    is_regression = label_mode == "multi"
    datamodule = ns.datasets.torch_ecg_data_example(
        train_seed=train_seed,
        dataset_name=dataset_name,
        db_dir=db_dir,
        working_dir=working_dir,
        label_mode=label_mode,
        is_regression=is_regression,
        max_records=max_records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        normalize_per_split=normalize_per_split,
        batch_size=batch_size,
        dataset_kwargs=dataset_kwargs,
    )

    input_dim = datamodule.train.inputs.shape[-1]
    if label_mode == "single":
        output_dim = int(datamodule.train.outputs.max()) + 1
        loss_fn = ce_loss
        validation_metric = "hamming_accuracy"
    else:
        output_dim = datamodule.train.outputs.shape[-1]
        loss_fn = bce_loss
        validation_metric = "neg_l2_dist"

    original_dsl = near.simple_ecg_dsl(
        input_dim=input_dim,
        num_classes=output_dim,
        hidden_dim=hidden_dim,
    )
    print(f"  Train samples: {len(datamodule.train.inputs)}")
    print(f"  Validation samples: {len(datamodule.val.inputs)}")
    print(f"  Test samples: {len(datamodule.test.inputs)}")
    print(f"  Input features: {input_dim}")
    print(f"  Output dim: {output_dim}")
    if hasattr(datamodule, "class_names"):
        print(f"  Classes ({len(datamodule.class_names)}): {datamodule.class_names}")

    print("\n[2/5] Setting up trainer and neural DSL...")
    trainer_cfg = near.NEARTrainerConfig(
        n_epochs=n_epochs,
        lr=lr,
        loss_callback=loss_fn,
        accelerator=device,
        validation_metric=validation_metric,
    )

    neural_dsl = near.NeuralDSL.from_dsl(
        dsl=original_dsl,
        neural_hole_filler=near.UnionNeuralHoleFiller(
            ChannelHoleFiller(),
            near.GenericMLPRNNNeuralHoleFiller(hidden_size=neural_hidden_size),
        ),
    )

    cost = near.default_near_cost(
        trainer_cfg=trainer_cfg,
        datamodule=datamodule,
        structural_cost_penalty=structural_cost_penalty,
    )

    print("\n[3/5] Creating NEAR search graph...")
    graph = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        is_goal=lambda _: True,
        cost=cost,
    )

    print(f"\n[4/5] Searching for programs (max {num_programs})...")
    iterator = ns.search.BoundedAStar(max_depth=max_depth)(graph)

    programs_list = []
    start_time = time.time()
    while True:
        try:
            program = next(iterator)
        except StopIteration:
            print(f"  Search exhausted after {len(programs_list)} programs")
            break

        timer = time.time() - start_time
        programs_list.append({"program": program, "time": timer})
        print(f"  Found program {len(programs_list)}: {program}")
        if len(programs_list) >= num_programs:
            print(f"  Reached max programs limit ({num_programs})")
            break

    search_time = time.time() - start_time
    print(f"\n  Total search time: {search_time:.2f} seconds")
    print(f"  Programs found: {len(programs_list)}")

    print("\n[5/5] Evaluating programs on test set...")
    for i, d in enumerate(programs_list):
        print(f"\n  Evaluating program {i + 1}/{len(programs_list)}...")
        program = d["program"]
        initialized_program = neural_dsl.initialize(program)

        _ = cost.validation_heuristic.with_n_epochs(final_n_epochs).compute_cost(
            neural_dsl, initialized_program, cost.embedding
        )

        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs

        module = ns.examples.near.TorchProgramModule(neural_dsl, initialized_program)
        metrics, predictions = eval_program(module, feature_data, labels, label_mode)

        d["report"] = metrics
        d["true_vals"] = labels.tolist()
        d["pred_vals"] = predictions.tolist()
        print(f"    Macro F1: {metrics.get('macro_f1', 0.0):.6f}")
        print(f"    Macro Accuracy: {metrics.get('macro_acc', 0.0):.6f}")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"\n  Saved final results to: {output_path_obj}")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    if programs_list:
        best_program = max(programs_list, key=lambda x: x["report"].get("macro_f1", 0.0))
        print(f"Best program: {best_program['program']}")
        print(f"  Macro F1: {best_program['report'].get('macro_f1', 0.0):.6f}")
        print(f"  Macro accuracy: {best_program['report'].get('macro_acc', 0.0):.6f}")
        print(f"  Discovery time: {best_program['time']:.2f}s")
    print("=" * 80)
    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="Run NEAR benchmark on a torch-ecg dataset."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="torch-ecg database class name (e.g. CPSC2018, CINC2021).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ecg_results/reproduction_general.pkl",
        help="Output path for results",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Local dataset directory passed to torch-ecg.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Working/cache directory passed to torch-ecg.",
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
        help="Maximum number of records to include (after split extraction).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Validation fraction when no official val split exists.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Test fraction when no official test split exists.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for split generation.",
    )
    parser.add_argument(
        "--no-normalize-per-split",
        action="store_true",
        help="Disable per-split min-max normalization.",
    )
    parser.add_argument(
        "--num-programs", type=int, default=20, help="Number of programs to discover"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=16, help="Hidden dimension for DSL"
    )
    parser.add_argument(
        "--neural-hidden-size",
        type=int,
        default=16,
        help="Hidden size for neural hole filler",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for search training"
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=40,
        help="Number of epochs for final training",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--structural-cost-penalty",
        type=float,
        default=0.1,
        help="Penalty multiplier for structural cost",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Label mode to use (single or multi)",
    )
    args = parser.parse_args()

    try:
        dataset_kwargs = json.loads(args.dataset_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for --dataset-kwargs-json: {args.dataset_kwargs_json}"
        ) from exc
    if not isinstance(dataset_kwargs, dict):
        raise ValueError("--dataset-kwargs-json must decode to a JSON object.")

    results = run_experiment(
        dataset_name=args.dataset_name,
        output_path=args.output,
        db_dir=args.db_dir,
        working_dir=args.working_dir,
        dataset_kwargs=dataset_kwargs,
        max_records=args.max_records if args.max_records > 0 else None,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        normalize_per_split=not args.no_normalize_per_split,
        num_programs=args.num_programs,
        hidden_dim=args.hidden_dim,
        neural_hidden_size=args.neural_hidden_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        final_n_epochs=args.final_epochs,
        lr=args.lr,
        structural_cost_penalty=args.structural_cost_penalty,
        device=args.device,
        label_mode=args.label_mode,
    )

    summary_path = args.output.replace(".pkl", "_summary.json")
    summary = {
        "dataset_name": args.dataset_name,
        "num_programs": len(results),
        "label_mode": args.label_mode,
        "programs": [
            {
                "program": str(r["program"]),
                "time": r["time"],
                "macro_f1": r["report"].get("macro_f1", 0.0),
                "macro_accuracy": r["report"].get("macro_acc", 0.0),
            }
            for r in results
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
