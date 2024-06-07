import torch
import torch.nn as nn

import sys

sys.path.append("C:\\Users\\asola\\OneDrive\\Documents\\GitHub\\neurosym-lib")

from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.examples.near.operations.lists import fold_torch, map_torch

import os
import numpy as np


def bounce_dsl():
    L = 4
    O = 4
    dslf = DSLFactory(L=L, O=O, max_overall_depth=5)
    ## DSL for the bounce example.
    dslf.typedef("fL", "{f, $L}")

    def foo():
        layer = nn.Linear(4, 4)
        layer.weight = nn.Parameter(
            torch.tensor(
                [[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.bias = nn.Parameter(torch.tensor([0, 0, 0, -0.98]), requires_grad=False)

        return layer

    dslf.parameterized(
        "linear_bool",
        "() -> $fL -> f",
        lambda lin: lin,
        dict(lin=lambda: nn.Linear(L, 1)),
    )
    dslf.parameterized(
        "linear", "() -> $fL -> $fL", lambda lin: lin, dict(lin=lambda: nn.Linear(L, L))
    )

    #   dslf.parameterized("cheat", "() -> $fL -> $fL", lambda lin, dummy: lambda x: lin(x) + 0 * dummy(x) , dict(lin=foo, dummy=lambda: nn.Linear(L, L)) )

    dslf.concrete(
        "ite",
        "(#a -> f, #a -> #a, #a -> #a) -> #a -> #a",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
    )
    dslf.concrete(
        "map", "(#a -> #b) -> [#a] -> [#b]", lambda f: lambda x: map_torch(f, x)
    )

    dslf.prune_to("[$fL] -> [$fL]")
    return dslf.finalize()


dsl = bounce_dsl()
print("Defined DSL")

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper

dataset_factory = lambda train_seed: DatasetWrapper(
    DatasetFromNpy(
        "./demodata/bounce_example/train_ex_data.npy",
        "./demodata/bounce_example/train_ex_labels.npy",
        train_seed,
    ),
    DatasetFromNpy(
        "./demodata/bounce_example/test_ex_data.npy",
        "./demodata/bounce_example/test_ex_labels.npy",
        None,
    ),
    batch_size=200,
)
datamodule = dataset_factory(42)
input_dim, output_dim = 4, 4
print(input_dim, output_dim)


import neurosym as ns
from neurosym.examples import near

t = ns.TypeDefiner(L=4, O=4)
t.typedef("fL", "{f, $L}")
neural_dsl = near.NeuralDSL.from_dsl(
    dsl=dsl,
    modules={
        **near.create_modules(
            "mlp",
            [t("($fL) -> $fL")],
            near.mlp_factory(hidden_size=10),
        ),
        **near.create_modules(
            "rnn_seq2seq",
            [t("([$fL]) -> [$fL]")],
            near.rnn_factory_seq2seq(hidden_size=10),
        ),
    },
)

print("Defined NEAR")

import pytorch_lightning as pl
import logging

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

trainer_cfg = near.NEARTrainerConfig(
    lr=7e-3,
    max_seq_len=300,
    n_epochs=100,
    num_labels=output_dim,
    train_steps=len(datamodule.train),
    loss_fn="MSELossRegression",
    scheduler="cosine",
    optimizer="adam",
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=15, verbose=False, mode="min"
)


def validation_cost(node):
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.n_epochs,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        # enable_model_summary=False,
        # enable_progress_bar=False,
        logger=False,
        callbacks=[early_stop_callback],
    )
    try:
        initialized_p = neural_dsl.initialize(node.program)
    except near.PartialProgramNotFoundError:
        return 10000.0

    model = neural_dsl.compute(initialized_p)
    if not isinstance(model, torch.nn.Module):
        del model
        del initialized_p
        model = near.TorchProgramModule(dsl=neural_dsl, program=node.program)
    pl_model = near.NEARTrainer(model, config=trainer_cfg)
    trainer.fit(pl_model, datamodule.train_dataloader(), datamodule.val_dataloader())
    return trainer.callback_metrics["val_loss"].item()


def checker(node):
    """
    In NEAR, any program that has no holes is valid.
    The hole checking is done before this function will
    be called so we can assume that the program has no holes.
    """
    return set(ns.symbols_for_program(node.program)) - set(dsl.symbols()) == set()


g = near.near_graph(
    neural_dsl,
    ns.parse_type(
        s="([{f, $L}]) -> [{f, $O}]", env=ns.TypeDefiner(L=input_dim, O=output_dim)
    ),
    is_goal=checker,
)

iterator = ns.search.bounded_astar(g, validation_cost, max_depth=16)
best_program_nodes = []
# Let's collect the top four programs
while len(best_program_nodes) <= 3:
    try:
        node = next(iterator)
        cost = validation_cost(node)
        best_program_nodes.append((node, cost))
        print("Got another program")
    except StopIteration:
        print("No more programs found.")
        break

print("THE END")

best_program_nodes = sorted(best_program_nodes, key=lambda x: x[1])
for i, (node, cost) in enumerate(best_program_nodes):
    print(
        "({i}) Cost: {cost:.4f}, {program}".format(
            i=i, program=ns.render_s_expression(node.program), cost=cost
        )
    )


def testProgram(best_program_node):
    module = near.TorchProgramModule(
        dsl=neural_dsl, program=best_program_node[0].program
    )
    pl_model = near.NEARTrainer(module, config=trainer_cfg)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-6, patience=5, verbose=False, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=2500,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        # enable_model_summary=False,
        # enable_progress_bar=False,
        logger=False,
        callbacks=[],
    )

    trainer.fit(pl_model, datamodule.train_dataloader(), datamodule.val_dataloader())
    T = 100
    path = np.zeros((T, 4))
    X = torch.tensor(
        np.array([0.21413583, 4.4062634, 3.4344807, 0.12440437]), dtype=torch.float32
    )
    for t in range(T):
        path[t, :] = X.detach().numpy()
        Y = module(X.unsqueeze(0)).squeeze(0)
        X = Y
    return path


trajectory = testProgram(best_program_nodes[0])
trajectoryb = testProgram(best_program_nodes[1])

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize


title = "Trajectories and their quadrants"

print(trajectory[:])
plt.scatter(trajectory[:, 0], trajectory[:, 1], marker="o")

plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color="gray")

plt.scatter(trajectoryb[:, 0], trajectoryb[:, 1], marker="o")

plt.plot(trajectoryb[:, 0], trajectoryb[:, 1], alpha=0.2, color="gray")

truth = datamodule.train.inputs[0, :, :]

print(truth[0, :])

plt.scatter(truth[:, 0], truth[:, 1], marker="o")

plt.plot(truth[:, 0], truth[:, 1], alpha=0.2, color="orange")


plt.title(title)
plt.xlim(-5, 10)
plt.ylim(-5, 7)
plt.grid(True)
plt.show()
