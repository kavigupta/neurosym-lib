import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import neurosym as ns
from neurosym.examples import near
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.examples.near.operations.lists import map_torch


def bounce_dsl():
    L = 4
    O = 4
    dslf = ns.DSLFactory(L=L, O=O, max_overall_depth=5)
    "YOUR CODE HERE"
    dslf.prune_to("[$fL] -> [$fL]")
    return dslf.finalize()


dsl = bounce_dsl()
print("Defined DSL")

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper

root = os.path.dirname(os.path.abspath(__file__))

dataset_factory = lambda train_seed: DatasetWrapper(
    DatasetFromNpy(
        f"{root}/data/bounce_example/train_ex_data.npy",
        f"{root}/data/bounce_example/train_ex_labels.npy",
        train_seed,
    ),
    DatasetFromNpy(
        f"{root}/data/bounce_example/test_ex_data.npy",
        f"{root}/data/bounce_example/test_ex_labels.npy",
        None,
    ),
    batch_size=200,
)
datamodule = dataset_factory(42)
input_dim, output_dim = 4, 4
print("Data has been loaded.")

print("Now, you have to add the code to actually search for a program using NEAR")

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

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)


trainer_cfg = near.NEARTrainerConfig(
    lr=5e-3,
    max_seq_len=300,
    n_epochs=100,
    num_labels=output_dim,
    train_steps=len(datamodule.train),
    loss_callback=torch.nn.functional.mse_loss,
    scheduler="cosine",
    optimizer=torch.optim.Adam,
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
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5)
        ],
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

# The code below assumes you found some top 3 programs and stored them in the best_program_nodes variable.
best_program_nodes = sorted(best_program_nodes, key=lambda x: x[1])
for i, (node, cost) in enumerate(best_program_nodes):
    print(
        "({i}) Cost: {cost:.4f}, {program}".format(
            i=i, program=ns.render_s_expression(node.program), cost=cost
        )
    )


# The function below is set up to further fine tune the program, test it, and return a set of values produced by it.
def testProgram(best_program_node):
    module = near.TorchProgramModule(
        dsl=neural_dsl, program=best_program_node[0].program
    )
    pl_model = near.NEARTrainer(module, config=trainer_cfg)
    trainer = pl.Trainer(
        max_epochs=4000,
        devices="auto",
        accelerator="cpu",
        enable_checkpointing=False,
        # enable_model_summary=False,
        # enable_progress_bar=False,
        logger=False,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5)
        ],
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


# We generate trajectories for the top 2 programs.
trajectory = testProgram(best_program_nodes[0])
trajectoryb = testProgram(best_program_nodes[1])


# And then the code below plots it to show how it compares to a trajectory in the training set.
import matplotlib.pyplot as plt

title = "Bouncing ball (ground truth in black)"

plt.figure(figsize=(8, 8))

print(trajectory[:])
plt.scatter(trajectory[:, 0], trajectory[:, 1], marker="o", color="C0")

plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color="C0")

plt.scatter(trajectoryb[:, 0], trajectoryb[:, 1], marker="o", color="C1")

plt.plot(trajectoryb[:, 0], trajectoryb[:, 1], alpha=0.2, color="C1")

truth = datamodule.train.inputs[0, :, :]

print(truth[0, :])

plt.scatter(truth[:, 0], truth[:, 1], marker="o", color="black")

plt.plot(truth[:, 0], truth[:, 1], alpha=0.2, color="black")


plt.title(title)
plt.xlim(-5, 10)
plt.ylim(-5, 7)
plt.grid(True)
plt.show()
