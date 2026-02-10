"""
Timing experiment comparing BoundedAStar vs OSGAstar on the CALMS21 mouse DSL.

Both strategies search for a program in the same DSL with the same cost function.
OSGAstar uses lazy cost evaluation (children inherit parent cost, own cost computed
only when popped), which avoids training neural networks for nodes never explored.

Usage:
    python experiments/osg_astar_mice_timing.py [--device DEVICE] [--n-programs N]
"""

import argparse
import itertools
import json
import time

import torch
from sklearn.metrics import classification_report

import neurosym as ns
from neurosym.examples import near


def _calms21_loss(predictions, targets):
    targets = targets.squeeze(-1)
    predictions = predictions.view(-1, predictions.shape[-1])
    targets = targets.view(-1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=2)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets_one_hot.float(),
        weight=torch.tensor([2.0, 1.0], device=predictions.device),
    )


def setup(device="cpu"):
    datamodule = ns.datasets.calms21_investigation_example(
        train_seed=0, batch_size=1024
    )
    _, output_dim = datamodule.train.get_io_dims()
    dsl = near.simple_calms21_dsl(num_classes=output_dim, hidden_dim=16)
    trainer_cfg = near.NEARTrainerConfig(
        n_epochs=12,
        lr=1e-4,
        loss_callback=_calms21_loss,
        accelerator=device,
    )
    neural_dsl = near.NeuralDSL.from_dsl(
        dsl=dsl,
        neural_hole_filler=near.GenericMLPRNNNeuralHoleFiller(hidden_size=16),
    )
    cost = near.default_near_cost(
        trainer_cfg=trainer_cfg,
        datamodule=datamodule,
    )
    return datamodule, neural_dsl, cost


def evaluate_program(program, neural_dsl, cost, datamodule):
    initialized = neural_dsl.initialize(program)
    cost.validation_heuristic.with_n_epochs(40).compute_cost(
        neural_dsl, initialized, cost.embedding
    )
    module = near.TorchProgramModule(neural_dsl, initialized)
    feature_data = datamodule.test.inputs
    labels = datamodule.test.outputs.flatten()
    predictions = module(torch.tensor(feature_data), environment=()).argmax(-1).numpy()
    report = classification_report(
        labels,
        predictions,
        target_names=["not investigation", "investigation"],
        output_dict=True,
    )
    return {
        "not_investigation_f1": report["not investigation"]["f1-score"],
        "investigation_f1": report["investigation"]["f1-score"],
    }


def run_search(name, strategy, n_programs, device="cpu"):
    print(f"\n--- {name} ---")
    print("Setting up...")
    datamodule, neural_dsl, cost = setup(device)
    g = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        is_goal=lambda _: True,
        cost=cost,
    )
    print(f"Searching for {n_programs} programs...")
    programs = []
    start = time.time()
    for program in itertools.islice(strategy(g), n_programs):
        elapsed = time.time() - start
        program_str = ns.render_s_expression(program)
        programs.append({"program": program_str, "latency": elapsed})
        print(f"  [{elapsed:.1f}s] {program_str}")

    print("Evaluating F1 scores...")
    for entry in programs:
        program = ns.parse_s_expression(entry["program"])
        f1s = evaluate_program(program, neural_dsl, cost, datamodule)
        entry.update(f1s)
        print(
            f"  {entry['program'][:60]}..."
            f"  not_inv={f1s['not_investigation_f1']:.4f}"
            f"  inv={f1s['investigation_f1']:.4f}"
        )

    return programs


def main():
    parser = argparse.ArgumentParser(description="BoundedAStar vs OSGAstar timing")
    parser.add_argument("--device", default="cpu", help="device (default: cpu)")
    parser.add_argument(
        "--n-programs", type=int, default=5, help="programs per strategy (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"CALMS21 Mouse DSL: BoundedAStar vs OSGAstar")
    print(f"device={args.device}  n_programs={args.n_programs}")
    print("=" * 60)

    results = {}
    for name, strategy in [
        ("BoundedAStar(max_depth=5)", ns.search.BoundedAStar(max_depth=5)),
        ("OSGAstar", ns.search.OSGAstar()),
    ]:
        results[name] = run_search(name, strategy, args.n_programs, args.device)

    output = {
        "device": args.device,
        "n_programs": args.n_programs,
        "strategies": results,
    }
    output_path = "experiments/osg_astar_mice_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
