"""
Timing experiment comparing BoundedAStar vs OSGAstar on the CALMS21 mouse DSL.

Both strategies search for a program in the same DSL with the same cost function.
OSGAstar uses lazy cost evaluation (children inherit parent cost, own cost computed
only when popped), which avoids training neural networks for nodes never explored.

Usage:
    python experiments/osg_astar_mice_timing.py [--device DEVICE]
"""

import argparse
import time

import torch

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
    return neural_dsl, cost


def run_search(name, strategy, device="cpu"):
    print(f"\n--- {name} ---")
    print("Setting up...")
    neural_dsl, cost = setup(device)
    g = near.near_graph(
        neural_dsl,
        neural_dsl.valid_root_types[0],
        is_goal=lambda _: True,
        cost=cost,
    )
    print("Searching...")
    start = time.time()
    program = next(strategy(g))
    elapsed = time.time() - start
    program_str = ns.render_s_expression(program)
    print(f"Done in {elapsed:.1f}s")
    return program_str, elapsed


def main():
    parser = argparse.ArgumentParser(description="BoundedAStar vs OSGAstar timing")
    parser.add_argument("--device", default="cpu", help="device (default: cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"CALMS21 Mouse DSL: BoundedAStar vs OSGAstar (device={args.device})")
    print("=" * 60)

    results = {}
    for name, strategy in [
        ("BoundedAStar(max_depth=5)", ns.search.BoundedAStar(max_depth=5)),
        ("OSGAstar", ns.search.OSGAstar()),
    ]:
        program_str, elapsed = run_search(name, strategy, args.device)
        results[name] = (program_str, elapsed)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    for name, (program_str, elapsed) in results.items():
        print(f"\n  {name}")
        print(f"    Time:    {elapsed:.1f}s")
        print(f"    Program: {program_str}")

    times = list(results.values())
    speedup = times[0][1] / times[1][1] if times[1][1] > 0 else float("inf")
    print(f"\n  Speedup: {speedup:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
