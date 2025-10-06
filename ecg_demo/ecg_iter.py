import os
import warnings

import numpy as np
import torch
import torch.nn as nn

import neurosym as ns
from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples import near
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.types.type import ArrowType, AtomicType

warnings.filterwarnings("ignore")
import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.loops.evaluation_loop").setLevel(logging.WARNING)


def load_dataset_npz(features_pth, label_pth):
    assert os.path.exists(features_pth), f"{features_pth} does not exist."
    assert os.path.exists(label_pth), f"{label_pth} does not exist."
    X = np.load(features_pth)
    y = np.load(label_pth)
    return X, y


def filter_multilabel(split):
    x_fname = f"data/ecg_multitask_example/x_{split}.npy"
    y_fname = f"data/ecg_multitask_example/y_{split}.npy"
    X = np.load(x_fname)
    y = np.load(y_fname)

    mask = y.sum(-1) == 1

    # filter
    X = X[mask]
    y = y[mask]

    # normalize each column of X to [0, 1]
    X = (X - X.min(0)) / (X.max(0) - X.min(0))

    # save as filtered
    np.save(x_fname.replace(f"{split}", f"{split}_filtered"), X)
    np.save(y_fname.replace(f"{split}", f"{split}_filtered"), y)


def create_dataset_factory(train_seed, is_regression, num_workers):
    """Creates a dataset factory for generating training and testing datasets.

    This factory function wraps the training and testing datasets with the
    `DatasetWrapper` class, handles batching and other dataset-related operations.

    Args:
        train_seed (int): The seed for random operations in the training dataset.
        is_regression (bool): Whether the dataset follows a regression or classification task.

    Returns:
        DatasetWrapper: An instance of `DatasetWrapper` containing both the
        training and testing datasets.
    """
    return DatasetWrapper(
        DatasetFromNpy(
            "data/ecg_multitask_example/x_train_filtered.npy",
            "data/ecg_multitask_example/y_train_filtered.npy",
            seed=train_seed,
            is_regression=is_regression,
        ),
        DatasetFromNpy(
            "data/ecg_multitask_example/x_test_filtered.npy",
            "data/ecg_multitask_example/y_test_filtered.npy",
            seed=0,
            is_regression=is_regression,
        ),
        batch_size=1000,
        num_workers=num_workers,
    )


datamodule = create_dataset_factory(train_seed=42, is_regression=False, num_workers=0)
# Retrieve input and output dimensions from the training dataset
# Use is_regression=True to get actual output dimension from shape (targets are one-hot)
input_dim, output_dim = datamodule.train.get_io_dims(is_regression=True)


def subset_selector_all_feat(x, channel, typ):
    x = x.reshape(-1, 12, 6, 2)
    typ_idx = torch.full(
        size=(x.shape[0],), fill_value=(0 if typ == "interval" else 1), device=x.device
    )
    channel_mask = channel(x.reshape(-1, 144))  # [B, 12]
    masked_x = (x * channel_mask[..., None, None]).sum(1)
    return masked_x[torch.arange(x.shape[0]), :, typ_idx]


def ecg_dsl(input_dim, output_dim, max_overall_depth=6):
    """Creates a domain-specific language (DSL) for neural symbolic computation.

    This function sets up a DSL with basic operations like addition, multiplication,
    and folds, as well as neural network components like linear layers.

    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output features.

    Returns:
        DSLFactory: An instance of `DSLFactory` with the defined operations and types.
    """
    feature_dim = 6
    dslf = DSLFactory(
        I=input_dim, O=output_dim, F=feature_dim, max_overall_depth=max_overall_depth
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
            case ArrowType(a, b):
                return filter_constants(a) and filter_constants(b)
            case AtomicType(a):
                return a not in ["channel", "feature"]
            case _:
                return True

    dslf.filtered_type_variable("num", lambda x: filter_constants(x))
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
        dict(lin=lambda: nn.Linear(feature_dim, output_dim)),
    )

    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
    )

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize(), dslf.t


print("input_dim", input_dim, "output_dim", output_dim)
dsl, dsl_type_env = ecg_dsl(input_dim, output_dim, max_overall_depth=6)
print("DSL")
print(dsl.render())

# Create neural DSL with the new API
neural_dsl = near.NeuralDSL.from_dsl(
    dsl=dsl,
    neural_hole_filler=near.UnionNeuralHoleFiller(
        # MLP for various transformations
        near.create_modules(
            [
                dsl_type_env("($fInp) -> $fInp"),
                dsl_type_env("($fInp) -> $fOut"),
                dsl_type_env("($fInp) -> $fFeat"),
                dsl_type_env("($fInp) -> {f, 1}"),
                dsl_type_env("() -> channel"),  # Use MLP for channel selection too
            ],
            near.mlp_factory(hidden_size=10),
        ),
    ),
)


# Configuration for ECG Trainer - using new simplified API
trainer_cfg = near.ECGTrainerConfig(
    lr=1e-3,
    n_epochs=100,
    num_labels=output_dim,
    loss_callback=near.ecg_cross_entropy_loss,
    scheduler="cosine",
    is_regression=False,
    is_multilabel=False,
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
)


# Create ECG-specific validation cost
validation_cost = near.ECGValidationCost(
    trainer_cfg=trainer_cfg,
    datamodule=datamodule,
    progress_by_epoch=False,
)

# Create NEAR cost with validation
cost_function = near.NearCost(
    structural_cost=near.MinimalStepsNearStructuralCost(symbol_costs={}),
    validation_heuristic=validation_cost,
    structural_cost_weight=0.5,
    embedding=near.IdentityProgramEmbedding(),
)


def checker(node):
    """
    In NEAR, any program that has no holes is valid.
    The hole checking is done before this function will
    be called so we can assume that the program has no holes.
    """
    return set(ns.symbols_for_program(node.program)) - set(dsl.symbols()) == set()


max_depth = 15
g = near.near_graph(
    neural_dsl,
    ns.parse_type(
        s="({f, $L}) -> {f, $O}", env=ns.TypeDefiner(L=input_dim, O=output_dim)
    ),
    max_depth=max_depth,
    is_goal=checker,
    cost=cost_function,
)

# Use bounded_astar_async with the new API
iterator = ns.search.bounded_astar_async(
    g, max_depth=max_depth, max_workers=32
)

count = 0
best_program_nodes = []
for node in iterator:
    # Compute final cost with full metrics
    final_cost = cost_function.compute_cost(node)
    best_program_nodes.append((node, final_cost))
    print(f"Found program with cost {final_cost}: {render_s_expression(node.program)}")
    count += 1
    if count > 10:
        break


# save the best program nodes
import pickle

with open("best_program_nodes.pkl", "wb") as f:
    pickle.dump(best_program_nodes, f)

import IPython; IPython.embed()
