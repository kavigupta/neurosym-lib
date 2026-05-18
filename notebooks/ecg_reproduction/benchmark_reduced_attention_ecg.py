#!/usr/bin/env python3
"""
Reproduction script for NEAR experiments on ECG with the reduced attention DSL.

Groups 12 ECG leads into 5 anatomical territories + 1 global channel = 6 variables.
Uses ECGDeli pre-extracted features and evaluates with macro AUC.
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

import neurosym as ns
from neurosym.examples import near
from neurosym.examples.near.dsls.attention_ecg_dsl import (
    ChannelUnpackEmbedding,
    FeatureGroupUnpackEmbedding,
    phase1_typed_ecg_dsl,
    reduced_attention_ecg_dsl,
)
from neurosym.examples.near.cost import NearCost, NumberHolesNearStructuralCost
from neurosym.examples.near.validation import ValidationCost
from neurosym.examples.near.metrics_ecg import (
    bootstrap_metrics,
    compute_ecg_metrics,
)


def ce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for single-label targets."""
    targets = targets.view(-1).long()
    predictions = predictions.view(-1, predictions.shape[-1])
    return torch.nn.functional.cross_entropy(predictions, targets)


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for multi-label targets."""
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1, targets.shape[-1]).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)


def eval_program(module, feature_data, labels, label_mode: str) -> tuple:
    """Evaluate a program module on test data."""
    x = torch.tensor(feature_data)
    predictions = (
        module(x, environment=()).detach().numpy()
    ).reshape(feature_data.shape[0], -1)
    metrics = compute_ecg_metrics(
        predictions, labels, label_mode=label_mode
    )
    return metrics, predictions


def _save_summary(programs_list, summary_path, label_mode):
    """Write a JSON summary of results so far."""
    summary = {
        "num_programs": len(programs_list),
        "label_mode": label_mode,
        "programs": [
            {
                "program": str(r["program"]),
                "time": r["time"],
                "macro_auc": r.get("report", {}).get("macro_auc", 0.0),
                "macro_f1": r.get("report", {}).get("macro_f1", 0.0),
            }
            for r in programs_list
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def run_experiment(
    output_path: str = "outputs/ecg_results/phase3_reduced/baseline.pkl",
    data_dir: str = "data/ecg",
    num_programs: int = 200,
    hidden_dim: int = 32,
    neural_hidden_size: int = 32,
    batch_size: int = 256,
    n_epochs: int = 50,
    final_n_epochs: int = 200,
    lr: float = 1e-3,
    structural_cost_penalty: float = 0.12,
    max_depth: int = 6,
    train_seed: int = 0,
    device: str = "cuda:0",
    label_mode: str = "single",
    validation_metric: str = "",
    restrict_to_hidden: bool = True,
    enable_bilinear: bool = False,
    enable_gate: bool = False,
    enable_embed_bool: bool = False,
    enable_feature_embeds: bool = False,
    enable_channel_attention: bool = False,
    disable_arith: bool = False,
    enable_phase1_embeds: bool = False,
    mlp_embed: bool = False,
    feature_major: bool = False,
    enable_phase1_attn_embeds: bool = False,
    phase1_typed: bool = False,
    use_number_holes: bool = False,
) -> List[Dict[str, Any]]:
    """Run the NEAR experiment on ECG with Reduced Attention DSL."""
    print("=" * 80)
    print("ECG NEAR Experiment - Reduced Attention DSL (6 grouped channels)")
    print("=" * 80)
    print("Configuration:")
    print(f"  Output path: {output_path}")
    print(f"  Data dir: {data_dir}")
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
    print(f"  Restrict to hidden: {restrict_to_hidden}")
    print(f"  Enable bilinear: {enable_bilinear}")
    print(f"  Enable gate: {enable_gate}")
    print(f"  Enable embed_bool: {enable_embed_bool}")
    print(f"  Use NumberHoles cost: {use_number_holes}")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading ECG dataset...")
    is_regression = label_mode == "multi"
    datamodule = ns.datasets.ecg_data_example(
        train_seed=train_seed,
        label_mode=label_mode,
        is_regression=is_regression,
        data_dir=data_dir,
        batch_size=batch_size,
        grouped=(not feature_major) and (not phase1_typed),
        feature_major=feature_major,
        feature_groups=phase1_typed,
    )

    if label_mode == "single":
        output_dim = int(datamodule.train.outputs.max()) + 1
        loss_fn = ce_loss
        default_metric = "hamming_accuracy"
    else:
        output_dim = datamodule.train.outputs.shape[-1]
        loss_fn = bce_loss
        default_metric = "neg_l2_dist"
    validation_metric = validation_metric or default_metric

    # Build DSL
    if phase1_typed:
        # Heterogeneous-typed Phase 1 DSL: 5 lambda variables, one per feature group.
        original_dsl = phase1_typed_ecg_dsl(
            num_classes=output_dim,
            hidden_dim=hidden_dim,
            max_overall_depth=max_depth,
            restrict_to_hidden=restrict_to_hidden,
            enable_attn_embeds=enable_phase1_attn_embeds,
            disable_arith=disable_arith,
        )
        num_channels = 5
        features_per_channel = None  # heterogeneous; not a single value
        embedding = FeatureGroupUnpackEmbedding()
    else:
        num_channels = datamodule.num_leads + datamodule.num_global_features
        features_per_channel = datamodule.features_per_lead
        original_dsl = reduced_attention_ecg_dsl(
            num_channels=num_channels,
            features_per_channel=features_per_channel,
            num_classes=output_dim,
            hidden_dim=hidden_dim,
            max_overall_depth=max_depth,
            restrict_to_hidden=restrict_to_hidden,
            enable_bilinear=enable_bilinear,
            enable_gate=enable_gate,
            enable_embed_bool=enable_embed_bool,
            enable_feature_embeds=enable_feature_embeds,
            enable_channel_attention=enable_channel_attention,
            disable_arith=disable_arith,
            enable_phase1_embeds=enable_phase1_embeds,
            mlp_embed=mlp_embed,
            enable_phase1_attn_embeds=enable_phase1_attn_embeds,
        )
        embedding = ChannelUnpackEmbedding()
    print(f"  Train samples: {len(datamodule.train.inputs)}")
    print(f"  Val samples: {len(datamodule.val.inputs)}")
    print(f"  Test samples: {len(datamodule.test.inputs)}")
    if phase1_typed:
        print(f"  Channels: 5 (heterogeneous): amp(60), int(84), st(12), morph(12), global(9)")
    else:
        print(f"  Channels: {num_channels} x {features_per_channel} features")
        print(f"  Channel names: {getattr(datamodule, 'lead_names', None)}")
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
        neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(
            hidden_size=neural_hidden_size
        ),
    )

    if use_number_holes:
        cost = NearCost(
            structural_cost=NumberHolesNearStructuralCost(),
            validation_heuristic=ValidationCost(
                trainer_cfg=trainer_cfg,
                datamodule=datamodule,
            ),
            structural_cost_penalty=structural_cost_penalty,
            embedding=embedding,
        )
    else:
        cost = near.default_near_cost(
            trainer_cfg=trainer_cfg,
            datamodule=datamodule,
            structural_cost_penalty=structural_cost_penalty,
            embedding=embedding,
        )

    # Create NEAR graph
    print("\n[3/5] Creating NEAR search graph...")
    g = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        max_depth=max_depth,
        is_goal=lambda _: True,
        cost=cost,
    )

    # Search for programs
    print(f"\n[4/5] Searching for programs (max {num_programs})...")
    iterator = ns.search.OSGAstar()(g)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    summary_path = str(output_path_obj).replace(".pkl", "_summary.json")

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
        n = len(programs_list)
        print(f"  Found program {n}: {program}")

        # Evaluate immediately
        initialized_program = neural_dsl.initialize(program)
        _ = cost.validation_heuristic.with_n_epochs(final_n_epochs).compute_cost(
            neural_dsl, initialized_program, cost.embedding
        )
        feature_data = datamodule.test.inputs
        labels = datamodule.test.outputs
        program_module = ns.examples.near.TorchProgramModule(neural_dsl, initialized_program)
        module = embedding.embed_initialized_program(program_module)
        metrics, predictions = eval_program(module, feature_data, labels, label_mode)

        programs_list[-1]["report"] = metrics
        programs_list[-1]["true_vals"] = labels.tolist()
        programs_list[-1]["pred_vals"] = predictions.tolist()
        print(
            f"    Macro AUC: {metrics.get('macro_auc', 0.0):.6f}  "
            f"Macro F1: {metrics.get('macro_f1', 0.0):.6f}"
        )

        # Save intermediate results
        with open(output_path, "wb") as f:
            pickle.dump(programs_list, f)
        _save_summary(programs_list, summary_path, label_mode)

        if n >= num_programs:
            print(f"  Reached max programs limit ({num_programs})")
            break

    search_time = time.time() - start_time
    print(f"\n  Total search time: {search_time:.2f} seconds")
    print(f"  Programs found: {len(programs_list)}")

    # Final save
    with open(output_path, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"\n  Saved final results to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    if programs_list:
        best_program = max(
            programs_list, key=lambda x: x["report"].get("macro_auc", 0.0)
        )
        print(f"Best program: {best_program['program']}")
        print(f"  Macro AUC: {best_program['report'].get('macro_auc', 0.0):.6f}")
        print(f"  Macro F1: {best_program['report'].get('macro_f1', 0.0):.6f}")
        print(f"  Discovery time: {best_program['time']:.2f}s")

        ci = bootstrap_metrics(
            best_program["pred_vals"],
            best_program["true_vals"],
            label_mode=label_mode,
        )
        print(
            f"  Bootstrap AUC: {ci['mean_auc']:.3f} "
            f"({ci['ci_5']:.3f} - {ci['ci_95']:.3f})"
        )
    print("=" * 80)

    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="NEAR ECG experiment with Reduced Attention DSL (6 grouped channels)"
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/ecg_results/phase3_reduced/baseline.pkl",
        help="Output path for results",
    )
    parser.add_argument("--data-dir", type=str, default="data/ecg")
    parser.add_argument("--num-programs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--neural-hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--structural-cost-penalty", type=float, default=0.12)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--label-mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--validation-metric", type=str, default="")
    parser.add_argument("--restrict-to-hidden", action="store_true", default=True)
    parser.add_argument("--no-restrict-to-hidden", dest="restrict_to_hidden", action="store_false")
    parser.add_argument("--enable-bilinear", action="store_true", default=False)
    parser.add_argument("--enable-gate", action="store_true", default=False)
    parser.add_argument("--enable-embed-bool", action="store_true", default=False)
    parser.add_argument(
        "--enable-feature-embeds",
        action="store_true",
        default=False,
        help="Add embed_amp, embed_int, embed_st, embed_morph productions (feature slicing)",
    )
    parser.add_argument(
        "--enable-channel-attention",
        action="store_true",
        default=False,
        help="Add channel_attention production: self-attention across all N channels at once",
    )
    parser.add_argument(
        "--disable-arith",
        action="store_true",
        default=False,
        help="Remove add/mul productions; force the model to combine via ite only",
    )
    parser.add_argument(
        "--feature-major",
        action="store_true",
        default=False,
        help="Use feature-type-major data layout (B, 1, 177) — for Phase 1-style DSL",
    )
    parser.add_argument(
        "--enable-phase1-embeds",
        action="store_true",
        default=False,
        help="Add Phase 1-style embed_amp/int/st/morph/global productions (slices feature-type-major 177-dim input)",
    )
    parser.add_argument(
        "--mlp-embed",
        action="store_true",
        default=False,
        help="Replace single Linear embed with 2-layer MLP + ReLU + Dropout",
    )
    parser.add_argument(
        "--enable-phase1-attn-embeds",
        action="store_true",
        default=False,
        help="Add per-feature-group lead-attention pooling productions (interpretable)",
    )
    parser.add_argument(
        "--phase1-typed",
        action="store_true",
        default=False,
        help="Use heterogeneous-typed Phase 1 DSL: 5 lambda variables, "
             "one per feature group (amp/int/st/morph/global) at native sizes",
    )
    parser.add_argument("--use-number-holes", action="store_true", default=False)
    args = parser.parse_args()

    results = run_experiment(
        output_path=args.output,
        data_dir=args.data_dir,
        num_programs=args.num_programs,
        hidden_dim=args.hidden_dim,
        neural_hidden_size=args.neural_hidden_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        structural_cost_penalty=args.structural_cost_penalty,
        max_depth=args.max_depth,
        final_n_epochs=args.final_epochs,
        lr=args.lr,
        device=args.device,
        label_mode=args.label_mode,
        validation_metric=args.validation_metric,
        restrict_to_hidden=args.restrict_to_hidden,
        enable_bilinear=args.enable_bilinear,
        enable_gate=args.enable_gate,
        enable_embed_bool=args.enable_embed_bool,
        enable_feature_embeds=args.enable_feature_embeds,
        enable_channel_attention=args.enable_channel_attention,
        disable_arith=args.disable_arith,
        feature_major=args.feature_major,
        enable_phase1_embeds=args.enable_phase1_embeds,
        mlp_embed=args.mlp_embed,
        enable_phase1_attn_embeds=args.enable_phase1_attn_embeds,
        phase1_typed=args.phase1_typed,
        use_number_holes=args.use_number_holes,
    )

    summary_path = args.output.replace(".pkl", "_summary.json")
    _save_summary(results, summary_path, args.label_mode)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
