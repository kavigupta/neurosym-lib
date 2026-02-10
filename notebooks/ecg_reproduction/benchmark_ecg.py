#!/usr/bin/env python3
"""
Reproduction script for NEAR experiments on the ECG dataset.

This script mirrors the flyvfly/bball reproduction structure and uses a
minimal ECG DSL for program search.
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any

import torch

import neurosym as ns
from neurosym.examples import near
from neurosym.examples.near.metrics import compute_metrics
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
    """
    Cross-entropy loss for single-label ECG targets.
    """
    targets = targets.view(-1).long()
    predictions = predictions.view(-1, predictions.shape[-1])
    return torch.nn.functional.cross_entropy(predictions, targets)


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for multi-label ECG targets.
    """
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1, targets.shape[-1]).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)


def eval_program(module, feature_data, labels, label_mode: str) -> tuple:
    """
    Evaluate a program module on test data.
    """
    predictions = (
        module(torch.tensor(feature_data), environment=()).detach().numpy()
    ).reshape(feature_data.shape[0], -1)
    if label_mode == "multi":
        metrics = compute_metrics(predictions, labels.astype("int32"))
    else:
        metrics = compute_metrics(predictions, labels)
    return metrics, predictions


def run_experiment(
    output_path: str = "outputs/ecg_results/reproduction.pkl",
    num_programs: int = 40,
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
    """
    Run the NEAR experiment on the ECG dataset.
    """
    print("=" * 80)
    print("ECG NEAR Experiment - Neurosym-lib Implementation")
    print("=" * 80)
    print("Configuration:")
    print(f"  Output path: {output_path}")
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

    # Prepare data and DSL
    print("\n[1/5] Loading ECG dataset...")
    is_regression = label_mode == "multi"
    datamodule = ns.datasets.ecg_data_example(
        train_seed=train_seed,
        label_mode=label_mode,
        is_regression=is_regression,
        batch_size=batch_size,
    )
    input_dim = datamodule.train.inputs.shape[-1]
    if label_mode == "single":
        output_dim = int(datamodule.train.outputs.max()) + 1
        loss_fn = ce_loss
        validation_metric = "hamming_accuracy"
    else:
        output_dim = datamodule.train.outputs.shape[-1]
        loss_fn = bce_loss
        # compute_metrics treats float labels as regression; use neg_l2_dist for search heuristic
        validation_metric = "neg_l2_dist"

    original_dsl = near.simple_ecg_dsl(
        input_dim=input_dim,
        num_classes=output_dim,
        hidden_dim=hidden_dim,
    )
    print(f"  Train samples: {len(datamodule.train.inputs)}")
    print(f"  Test/Val samples: {len(datamodule.test.inputs)}")
    print(f"  Input features: {input_dim}")
    print(f"  Output dim: {output_dim}")

    # Trainer configuration
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

    # Create the NEAR graph
    print("\n[3/5] Creating NEAR search graph...")
    g = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        is_goal=lambda _: True,
        cost=cost,
    )

    # Search for programs with bounded A*
    print(f"\n[4/5] Searching for programs (max {num_programs})...")
    iterator = ns.search.BoundedAStar(max_depth=max_depth)(g)

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

    # Evaluate each discovered program
    print("\n[5/5] Evaluating programs on test set...")
    for i, d in enumerate(programs_list):
        print(f"\n  Evaluating program {i + 1}/{len(programs_list)}...")
        program = d["program"]
        initialized_program = neural_dsl.initialize(program)

        # Train with final epoch count
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

        print(f"    Hamming accuracy: {metrics.get('hamming_accuracy', 0.0):.6f}")

    # Save final results
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"\n  Saved final results to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    if programs_list:
        best_program = max(
            programs_list, key=lambda x: x["report"].get("hamming_accuracy", 0.0)
        )
        print(f"Best program: {best_program['program']}")
        print(
            f"  Hamming accuracy: {best_program['report'].get('hamming_accuracy', 0.0):.6f}"
        )
        print(f"  Discovery time: {best_program['time']:.2f}s")
    print("=" * 80)

    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce NEAR results on ECG dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ecg_results/reproduction.pkl",
        help="Output path for results",
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

    results = run_experiment(
        output_path=args.output,
        num_programs=args.num_programs,
        hidden_dim=args.hidden_dim,
        neural_hidden_size=args.neural_hidden_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        structural_cost_penalty=args.structural_cost_penalty,
        final_n_epochs=args.final_epochs,
        lr=args.lr,
        device=args.device,
        label_mode=args.label_mode,
    )

    summary_path = args.output.replace(".pkl", "_summary.json")
    summary = {
        "num_programs": len(results),
        "label_mode": args.label_mode,
        "programs": [
            {
                "program": str(r["program"]),
                "time": r["time"],
                "hamming_accuracy": r["report"].get("hamming_accuracy", 0.0),
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
