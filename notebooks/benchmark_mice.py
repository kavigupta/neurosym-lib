#!/usr/bin/env python3
"""
Reproduction script for MICE-DSL NEAR experiments on CRIM-13 dataset.

This script reproduces the results from the neurosym-lib implementation of NEAR
for the CRIM-13 mice behavior classification task.

Expected baseline results:
- F1-score: 0.471933
- Hamming accuracy: 0.893698
- Time: ~112 seconds
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_curve

import neurosym as ns
from neurosym.examples import near


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross entropy loss with class weighting.

    Args:
        predictions: Model predictions with shape (B, T, O)
        targets: Ground truth labels with shape (B, T, 1)

    Returns:
        Binary cross-entropy loss
    """
    targets = targets.squeeze(-1)  # (B, T, 1) -> (B, T)
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view_as(predictions)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions.float(),
        targets.float(),
        weight=torch.tensor([2.0], device=predictions.device),
    )


def threshold_predictions(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    threshold_type: str = "quantile"
) -> torch.Tensor:
    """
    Apply dynamic thresholding to predictions.

    Args:
        y_true: Ground truth labels
        y_scores: Model prediction scores (logits)
        threshold_type: Type of thresholding ('quantile', 'roc', or 'static')

    Returns:
        Binary predictions
    """
    if threshold_type == "quantile":
        threshold = np.quantile(y_scores, 1 - y_true.float().mean())
    elif threshold_type == "roc":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        threshold = thresholds[np.argmax(tpr - fpr)]
    elif threshold_type == "static":
        threshold = 0.5
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    predictions = (torch.sigmoid(y_scores) < threshold).int()
    return predictions


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """
    Compute evaluation metrics for predictions.

    Args:
        predictions: Raw prediction scores (logits)
        ground_truth: Ground truth labels

    Returns:
        Dictionary containing various metrics
    """
    predictions = (
        threshold_predictions(
            torch.tensor(ground_truth, dtype=torch.float32),
            torch.tensor(predictions, dtype=torch.float32),
            threshold_type="quantile",
        )
        .numpy()
        .flatten()
    )
    ground_truth = ground_truth.flatten()

    try:
        weighted_avg_f1 = f1_score(predictions, ground_truth, average="weighted")
        unweighted_avg_f1 = f1_score(predictions, ground_truth, average="macro")
        all_f1 = f1_score(predictions, ground_truth, average=None)
        hamming_accuracy = 1 - hamming_loss(ground_truth, predictions)
    except ValueError as e:
        print(f"Error computing metrics: {e}")
        weighted_avg_f1 = 0.0
        unweighted_avg_f1 = 0.0
        all_f1 = []
        hamming_accuracy = 0.0

    report = classification_report(ground_truth, predictions, output_dict=True)
    return {
        "f1_score": weighted_avg_f1,
        "unweighted_f1": unweighted_avg_f1,
        "all_f1s": all_f1,
        "hamming_accuracy": hamming_accuracy,
        "report": report,
    }


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
    predictions = (
        module(torch.tensor(feature_data), environment=()).detach().numpy().flatten()
    )
    metrics = compute_metrics(predictions, labels)
    return metrics, predictions


def run_experiment(
    output_path: str = "outputs/mice_results/reproduction.pkl",
    num_programs: int = 10,
    hidden_dim: int = 1,
    neural_hidden_size: int = 32,
    batch_size: int = 50,
    n_epochs: int = 15,
    final_n_epochs: int = 15,
    lr: float = 1e-4,
    structural_cost_weight: float = 0.005,
    max_depth: int = 10,
    frontier_capacity: int = 8,
    train_seed: int = 0,
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """
    Run the NEAR experiment on CRIM-13 dataset.

    Args:
        output_path: Path to save results
        num_programs: Number of programs to discover
        hidden_dim: Hidden dimension for DSL
        neural_hidden_size: Hidden size for neural hole filler
        batch_size: Training batch size
        n_epochs: Number of epochs for search training
        final_n_epochs: Number of epochs for final training
        lr: Learning rate
        structural_cost_weight: Weight for structural cost in search
        max_depth: Maximum program depth
        frontier_capacity: Frontier capacity for bounded A*
        train_seed: Random seed for data
        device: Device to use for training

    Returns:
        List of discovered programs with metrics
    """
    print("="*80)
    print("CRIM-13 NEAR Experiment - Neurosym-lib Implementation")
    print("="*80)
    print(f"Configuration:")
    print(f"  Output path: {output_path}")
    print(f"  Number of programs: {num_programs}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Neural hidden size: {neural_hidden_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs (search): {n_epochs}")
    print(f"  Epochs (final): {final_n_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Structural cost weight: {structural_cost_weight}")
    print(f"  Max depth: {max_depth}")
    print(f"  Frontier capacity: {frontier_capacity}")
    print(f"  Device: {device}")
    print("="*80)

    # Prepare data and DSL
    print("\n[1/5] Loading CRIM-13 dataset...")
    datamodule = ns.datasets.crim13_data_example(train_seed=train_seed, batch_size=batch_size)
    output_dim = 1
    original_dsl = near.simple_crim13_dsl(num_classes=output_dim, hidden_dim=hidden_dim)
    print(f"  Train samples: {len(datamodule.train.inputs)}")
    print(f"  Test/Val samples: {len(datamodule.test.inputs)}")
    print(f"  Input features: {datamodule.train.inputs.shape[-1]}")

    # Trainer configuration
    print("\n[2/5] Setting up trainer and neural DSL...")
    trainer_cfg = near.NEARTrainerConfig(
        n_epochs=n_epochs,
        lr=lr,
        loss_callback=bce_loss,
        accelerator=device,
    )

    neural_dsl = near.NeuralDSL.from_dsl(
        dsl=original_dsl,
        neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=neural_hidden_size),
    )

    cost = near.default_near_cost(
        trainer_cfg=trainer_cfg,
        datamodule=datamodule,
        structural_cost_weight=structural_cost_weight,
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
    iterator = ns.search.bounded_astar(g, max_depth=max_depth, frontier_capacity=frontier_capacity)

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
    print(f"\n[5/5] Evaluating programs on test set...")
    for i, d in enumerate(programs_list):
        print(f"\n  Evaluating program {i+1}/{len(programs_list)}...")
        program = d["program"]
        initialized_program = neural_dsl.initialize(program)

        # Train with final epoch count
        _ = cost.validation_heuristic.with_n_epochs(final_n_epochs).compute_cost(
            neural_dsl, initialized_program, cost.embedding
        )

        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs.flatten()

        module = ns.examples.near.TorchProgramModule(neural_dsl, initialized_program)
        metrics, predictions = eval_program(module, feature_data, labels)

        d["report"] = metrics
        d["true_vals"] = labels.tolist()
        d["pred_vals"] = predictions.tolist()

        print(f"    F1-score: {metrics['f1_score']:.6f}")
        print(f"    Hamming accuracy: {metrics['hamming_accuracy']:.6f}")

    # Save final results
    with open(output_path, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"\n  Saved final results to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    if programs_list:
        best_program = max(programs_list, key=lambda x: x["report"]["f1_score"])
        print(f"Best program: {best_program['program']}")
        print(f"  F1-score: {best_program['report']['f1_score']:.6f}")
        print(f"  Hamming accuracy: {best_program['report']['hamming_accuracy']:.6f}")
        print(f"  Discovery time: {best_program['time']:.2f}s")
    print("="*80)

    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce MICE-DSL NEAR results on CRIM-13 dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mice_results/reproduction.pkl",
        help="Output path for results"
    )
    parser.add_argument(
        "--num-programs",
        type=int,
        default=10,
        help="Number of programs to discover"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1,
        help="Hidden dimension for DSL"
    )
    parser.add_argument(
        "--neural-hidden-size",
        type=int,
        default=32,
        help="Hidden size for neural hole filler"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs for search training"
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=15,
        help="Number of epochs for final training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training"
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
        final_n_epochs=args.final_epochs,
        lr=args.lr,
        device=args.device,
    )

    # Save summary
    summary_path = args.output.replace(".pkl", "_summary.json")
    summary = {
        "num_programs": len(results),
        "programs": [
            {
                "program": r["program"],
                "time": r["time"],
                "f1_score": r["report"]["f1_score"],
                "hamming_accuracy": r["report"]["hamming_accuracy"],
            }
            for r in results
        ]
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
