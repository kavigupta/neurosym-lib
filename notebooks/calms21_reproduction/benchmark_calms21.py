#!/usr/bin/env python3
"""
Reproduction script for NEAR experiments on the CALMS21 investigation task.

This script reproduces the results from the neurosym-lib implementation of NEAR
for the CALMS21 investigation vs. not investigation classification task.
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


def calms21_bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss with class weighting for CALMS21 investigation.

    Args:
        predictions: Model predictions with shape (B, T, O) or (B, O)
        targets: Ground truth labels with shape (B, T, 1) or (B, 1)

    Returns:
        Binary cross-entropy loss
    """
    targets = targets.squeeze(-1)  # (B, T, 1) -> (B, T) or (B, 1) -> (B,)
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets_one_hot.float(),
        weight=torch.tensor([2.0, 1.0], device=predictions.device),
    )


def eval_program(module, feature_data, labels) -> tuple:
    """
    Evaluate a program module on test data.

    Args:
        module: Trained program module
        feature_data: Input features
        labels: Ground truth labels

    Returns:
        Tuple of (metrics dict, raw predictions)
    """
    predictions = module(torch.tensor(feature_data), environment=()).detach().numpy()
    metrics = compute_metrics(predictions, labels)
    return metrics, predictions


def run_experiment(
    output_path: str = "outputs/calms21_results/reproduction.pkl",
    num_programs: int = 40,
    hidden_dim: int = 16,
    neural_hidden_size: int = 16,
    batch_size: int = 50,
    n_epochs: int = 12,
    final_n_epochs: int = 40,
    lr: float = 1e-4,
    structural_cost_penalty: float = 0.05,
    max_depth: int = 6,
    train_seed: int = 0,
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """
    Run the NEAR experiment on the CALMS21 investigation dataset.

    Args:
        output_path: Path to save results
        num_programs: Number of programs to discover
        hidden_dim: Hidden dimension for DSL
        neural_hidden_size: Hidden size for neural hole filler
        batch_size: Training batch size
        n_epochs: Number of epochs for search training
        final_n_epochs: Number of epochs for final training
        lr: Learning rate
        structural_cost_penalty: Weight for structural cost in search
        max_depth: Maximum program depth
        train_seed: Random seed for data
        device: Device to use for training

    Returns:
        List of discovered programs with metrics
    """
    print("=" * 80)
    print("CALMS21 Investigation NEAR Experiment - Neurosym-lib Implementation")
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
    print("=" * 80)

    # Prepare data and DSL
    print("\n[1/5] Loading CALMS21 investigation dataset...")
    datamodule = ns.datasets.calms21_investigation_example(
        train_seed=train_seed, batch_size=batch_size
    )
    _, output_dim = datamodule.train.get_io_dims()
    seq_len = datamodule.train.inputs.shape[1]
    original_dsl = near.simple_calms21_dsl(num_classes=output_dim, hidden_dim=hidden_dim)
    print(f"  Train samples: {len(datamodule.train.inputs)}")
    print(f"  Test/Val samples: {len(datamodule.test.inputs)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input features: {datamodule.train.inputs.shape[-1]}")
    print(f"  Output classes: {output_dim}")

    # Trainer configuration
    print("\n[2/5] Setting up trainer and neural DSL...")
    trainer_cfg = near.NEARTrainerConfig(
        n_epochs=n_epochs,
        lr=lr,
        loss_callback=calms21_bce_loss,
        accelerator=device,
    )

    neural_dsl = near.NeuralDSL.from_dsl(
        dsl=original_dsl,
        neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(
            hidden_size=neural_hidden_size,
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

    # Collect programs
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

    # Save raw program list
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = str(output_path_obj).replace(".pkl", "_raw.pkl")
    with open(raw_output_path, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"  Saved raw programs to: {raw_output_path}")

    # Evaluate each discovered program
    print("\n[5/5] Evaluating programs on test set...")
    for i, d in enumerate(programs_list):
        print(f"\n  Evaluating program {i+1}/{len(programs_list)}...")
        program = d["program"]
        initialized_program = neural_dsl.initialize(program)

        # Train with final epoch count
        _ = cost.validation_heuristic.with_n_epochs(final_n_epochs).compute_cost(
            neural_dsl, initialized_program, cost.embedding
        )

        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs

        module = ns.examples.near.TorchProgramModule(neural_dsl, initialized_program)
        metrics, predictions = eval_program(module, feature_data, labels)

        d["report"] = metrics
        d["true_vals"] = labels.tolist()
        d["pred_vals"] = predictions.tolist()

        report = metrics.get("report", {})
        if "0" in report and "1" in report:
            print(
                f"    F1 (not investigation): {report['0']['f1-score']:.6f}"
            )
            print(
                f"    F1 (investigation): {report['1']['f1-score']:.6f}"
            )
        print(f"    Accuracy: {metrics['hamming_accuracy']:.6f}")

    # Save final results
    with open(output_path, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"\n  Saved final results to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    if programs_list:
        best_program = max(programs_list, key=lambda x: x["report"]["f1_score"])
        print(f"Best program: {best_program['program']}")
        print(f"  Weighted F1: {best_program['report']['f1_score']:.6f}")
        print(f"  Accuracy: {best_program['report']['hamming_accuracy']:.6f}")
        print(f"  Discovery time: {best_program['time']:.2f}s")
    print("=" * 80)

    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce NEAR results on the CALMS21 investigation dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calms21_results/reproduction.pkl",
        help="Output path for results",
    )
    parser.add_argument(
        "--num-programs",
        type=int,
        default=40,
        help="Number of programs to discover",
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
        "--epochs", type=int, default=12, help="Number of epochs for search training"
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
        default=0.05,
        help=(
            "Penalty multiplier for structural cost (default: 0.05, "
            "roughly balancing structural and validation cost scales)"
        ),
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="Maximum program depth"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )

    args = parser.parse_args()

    # Run experiment
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
        max_depth=args.max_depth,
    )

    # Save summary
    summary_path = args.output.replace(".pkl", "_summary.json")
    summary = {
        "num_programs": len(results),
        "programs": [
            {
                "program": str(r["program"]),
                "time": r["time"],
                "f1_score": r["report"]["f1_score"],
                "hamming_accuracy": r["report"]["hamming_accuracy"],
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
