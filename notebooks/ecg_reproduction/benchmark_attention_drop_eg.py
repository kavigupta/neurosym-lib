#!/usr/bin/env python3
"""
Reproduction script for NEAR experiments on the ECG dataset using the
attention + drop-variables ECG DSL.
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


def run_experiment(
    output_path: str = "outputs/ecg_results/attention_drop_eg_reproduction.pkl",
    data_dir: str = "data/ecg_classification/ecg",
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
    print("ECG NEAR Experiment (attention_drop_eg_dsl)")
    print("=" * 80)
    print(f"Data dir: {data_dir}")

    is_regression = label_mode == "multi"
    datamodule = ns.datasets.ecg_data_example(
        train_seed=train_seed,
        label_mode=label_mode,
        is_regression=is_regression,
        data_dir=data_dir,
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
        validation_metric = "neg_l2_dist"

    original_dsl = near.attention_drop_eg_dsl(
        input_dim=input_dim,
        num_classes=output_dim,
        hidden_dim=hidden_dim,
    )

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

    g = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        is_goal=lambda _: True,
        cost=cost,
    )

    iterator = ns.search.BoundedAStar(max_depth=max_depth)(g)
    programs_list = []
    start_time = time.time()

    while True:
        try:
            program = next(iterator)
        except StopIteration:
            break
        programs_list.append({"program": program, "time": time.time() - start_time})
        if len(programs_list) >= num_programs:
            break

    for d in programs_list:
        initialized_program = neural_dsl.initialize(d["program"])
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

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "wb") as f:
        pickle.dump(programs_list, f)
    print(f"Saved results to: {output_path_obj}")

    return programs_list


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce NEAR results with attention_drop_eg_dsl on ECG dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ecg_results/attention_drop_eg_reproduction.pkl",
        help="Output path for results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ecg_classification/ecg",
        help="Directory containing standardized ECG .npz splits.",
    )
    parser.add_argument("--num-programs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--neural-hidden-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--structural-cost-penalty", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="single",
        choices=["single", "multi"],
    )
    args = parser.parse_args()

    results = run_experiment(
        output_path=args.output,
        data_dir=args.data_dir,
        num_programs=args.num_programs,
        hidden_dim=args.hidden_dim,
        neural_hidden_size=args.neural_hidden_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        final_n_epochs=args.final_epochs,
        lr=args.lr,
        structural_cost_penalty=args.structural_cost_penalty,
        max_depth=args.max_depth,
        device=args.device,
        label_mode=args.label_mode,
    )

    summary_path = str(args.output).replace(".pkl", "_summary.json")
    summary = {
        "num_programs": len(results),
        "label_mode": args.label_mode,
        "programs": [
            {
                "program": str(r["program"]),
                "time": r["time"],
                "macro_f1": r.get("report", {}).get("macro_f1", 0.0),
                "macro_accuracy": r.get("report", {}).get("macro_acc", 0.0),
            }
            for r in results
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
