import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import neurosym as ns
from neurosym.examples import near

pl = ns.import_pytorch_lightning()


def load_dataset_npz(features_pth, label_pth):
    assert os.path.exists(features_pth), f"{features_pth} does not exist."
    assert os.path.exists(label_pth), f"{label_pth} does not exist."
    X = np.load(features_pth)
    y = np.load(label_pth)
    return X, y


def filter_multilabel(split, data_dir):
    x_fname = os.path.join(data_dir, f"x_{split}.npy")
    y_fname = os.path.join(data_dir, f"y_{split}.npy")
    X = np.load(x_fname)
    y = np.load(y_fname)

    mask = y.sum(-1) == 1

    X = X[mask]
    y = y[mask]

    X = (X - X.min(0)) / (X.max(0) - X.min(0))

    np.save(x_fname.replace(f"{split}", f"{split}_filtered"), X.astype(np.float32))
    np.save(y_fname.replace(f"{split}", f"{split}_filtered"), y.astype(np.float32))


def dataset_factory(data_dir, train_seed):
    return ns.DatasetWrapper(
        ns.DatasetFromNpy(
            os.path.join(data_dir, "x_train_filtered.npy"),
            os.path.join(data_dir, "y_train_filtered.npy"),
            train_seed,
        ),
        ns.DatasetFromNpy(
            os.path.join(data_dir, "x_test_filtered.npy"),
            os.path.join(data_dir, "y_test_filtered.npy"),
            None,
        ),
        batch_size=200,
        num_workers=0,
    )


def plot_trajectory(trajectory, color):
    # TODO: Implement trajectory plotting
    pass


def subset_selector_all_feat(x, channel, typ):
    x = x.reshape(-1, 12, 6, 2)
    typ_idx = torch.full(
        size=(x.shape[0],), fill_value=(0 if typ == "interval" else 1), device=x.device
    )
    channel_mask = channel(x.reshape(-1, 144))
    masked_x = (x * channel_mask[..., None, None]).sum(1)
    return masked_x[torch.arange(x.shape[0]), :, typ_idx]


def ecg_dsl(input_size, output_size, max_overall_depth=6):
    feature_dim = 6
    dslf = ns.DSLFactory(
        I=input_size, O=output_size, F=feature_dim, max_overall_depth=max_overall_depth
    )
    dslf.typedef("fInp", "{f, $I}")
    dslf.typedef("fOut", "{f, $O}")
    dslf.typedef("fFeat", "{f, $F}")

    for i in range(12):
        dslf.concrete(
            f"channel_{i}",
            "() -> () -> channel",
            lambda: lambda x: torch.nn.functional.one_hot(
                torch.full(tuple(x.shape[:-1]), i, device=x.device, dtype=torch.long),
                num_classes=12,
            ),
        )

    dslf.concrete(
        "select_interval",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda ch: lambda x: subset_selector_all_feat(x, ch, "interval"),
    )

    dslf.concrete(
        "select_amplitude",
        "(() -> channel) -> ($fInp) -> $fFeat",
        lambda ch: lambda x: subset_selector_all_feat(x, ch, "amplitude"),
    )

    def guard_callables(fn, **kwargs):
        is_callable = [callable(kwargs[k]) for k in kwargs]
        if any(is_callable):
            return lambda z: fn(
                **{
                    k: (kwargs[k](z) if is_callable[i] else kwargs[k])
                    for i, k in enumerate(kwargs)
                }
            )
        else:
            return fn(**kwargs)

    def filter_constants(x):
        match x:
            case ns.ArrowType(a, b):
                return filter_constants(a) and filter_constants(b)
            case ns.AtomicType(a):
                return a not in ["channel", "feature"]
            case _:
                return True

    dslf.filtered_type_variable("num", filter_constants)
    dslf.concrete(
        "add",
        "(%num, %num) -> %num",
        lambda x, y: guard_callables(fn=lambda x, y: x + y, x=x, y=y),
    )
    dslf.concrete(
        "mul",
        "(%num, %num) -> %num",
        lambda x, y: guard_callables(fn=lambda x, y: x * y, x=x, y=y),
    )
    dslf.parameterized(
        "linear",
        "(($fInp) -> $fFeat) -> $fInp -> {f, 1}",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, 1)),
    )

    dslf.parameterized(
        "output",
        "(($fInp) -> $fFeat) -> $fInp -> $fOut",
        lambda f, lin: lambda x: lin(f(x)),
        dict(lin=lambda: nn.Linear(feature_dim, output_size)),
    )

    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        near.operations.ite_torch,
    )
    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize(), dslf.t


def main(args):
    hidden_size = 16
    logging.basicConfig(level=logging.INFO)
    filter_multilabel("train", args.data_dir)
    filter_multilabel("test", args.data_dir)

    input_size, output_size = 144, 9
    dsl, dsl_type_env = ecg_dsl(input_size=input_size, output_size=output_size)

    datamodule = dataset_factory(args.data_dir, 42)

    t = ns.TypeDefiner(L=input_size, O=output_size)
    t.typedef("fL", "{f, $L}")
    neural_dsl = near.NeuralDSL.from_dsl(
        dsl=dsl,
        modules={
            **near.create_modules(
                "mlp",
                [
                    dsl_type_env("($fInp) -> $fInp"),
                    dsl_type_env("($fInp) -> $fOut"),
                    dsl_type_env("($fInp) -> $fFeat"),
                    dsl_type_env("($fInp) -> {f, 1}"),
                ],
                near.mlp_factory(hidden_size=hidden_size),
            ),
            **near.create_modules(
                "constant_int",
                [dsl_type_env("() -> channel")],
                near.selector_factory(input_size=input_size),
                known_atom_shapes=dict(channel=(12,), feature=(6,)),
            ),
        },
    )

    def cross_entropy_callback(predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets, label_smoothing=1e-2)

    trainer_cfg = near.NEARTrainerConfig(
        lr=1e-3,
        max_seq_len=300,
        n_epochs=100,
        num_labels=output_size,
        train_steps=len(datamodule.train),
        loss_callback=cross_entropy_callback,
        scheduler="none",
        optimizer=torch.optim.Adam,
    )
    torch.set_num_threads(2)
    validation_cost = near.ValidationCost(
        neural_dsl=neural_dsl,
        trainer_cfg=trainer_cfg,
        datamodule=datamodule,
        accelerator="cpu",
        devices="auto",
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=1,
                mode="min",
                # verbose=True,
            )
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        progress_by_epoch=True,
        structural_cost_weight=0.01,
        val_metric="val_weighted_avg_f1"
    )

    g = near.near_graph(
        neural_dsl,
        ns.parse_type(
            s="({f, $L}) -> {f, $O}", env=ns.TypeDefiner(L=input_size, O=output_size)
        ),
        is_goal=neural_dsl.program_has_no_holes,
    )

    if args.max_workers == 0:
        iterator = ns.search.bounded_astar(
            g, validation_cost, max_depth=16
        )
    else:
        iterator = ns.search.bounded_astar_async(
            g,
            validation_cost,
            max_depth=16,
            max_workers=args.max_workers,
        )
    best_program_nodes = []
    while len(best_program_nodes) < args.num_programs:
        try:
            node = next(iterator)
            cost = validation_cost(node)
            best_program_nodes.append((node, cost))
            print("Got another program")
        except StopIteration:
            print("No more programs found.")
            break

    best_program_nodes = sorted(best_program_nodes, key=lambda x: x[1])
    for i, (node, cost) in enumerate(best_program_nodes):
        print(
            "({i}) Cost: {cost:.4f}, {program}".format(
                i=i, program=ns.render_s_expression(node.program), cost=cost
            )
        )

    # Save programs
    with open(f"{args.data_dir}/best_programs.pkl", "wb") as fp:
        pickle.dump(best_program_nodes, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neurosymbolic Software Tutorial - ECG Dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--num_programs", type=int, required=True, help="Number of programs to store."
    )
    parser.add_argument(
        "--max_workers", type=int, required=False, default=0, help="Number of concurrent beams to explore."
    )
    args = parser.parse_args()

    main(args)
