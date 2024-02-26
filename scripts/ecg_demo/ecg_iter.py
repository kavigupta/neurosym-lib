import os
import numpy as np
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import neurosym as ns
from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.dsl.dsl_factory import DSLFactory
from neurosym.examples import near
from neurosym.examples.near.operations.basic import ite_torch
from neurosym.programs.s_expression_render import render_s_expression
from neurosym.types.type import ArrowType, AtomicType

import warnings
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

    # normalize each column of X to [-1, 1]
    X = (X - X.min(0)) / (X.max(0) - X.min(0))

    # save as filtered
    np.save(x_fname.replace(f"{split}", f"{split}_filtered"), X)
    np.save(y_fname.replace(f"{split}", f"{split}_filtered"), y)


def create_dataset_factory(train_seed, is_regression, n_workers):
    """Creates a dataset factory for generating training and testing datasets.

    This factory function wraps the training and testing datasets with the
    `DatasetWrapper` class, handles batching and other dataset-related operations.

    Args:
        train_seed (int): The seed for random operations in the training dataset.
        is_regression (ool): Whether the dataset follows a regression or classification task.

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
        n_workers=n_workers,
    )


datamodule = create_dataset_factory(train_seed=42, is_regression=False, n_workers=0)
# Retrieve input and output dimensions from the training dataset
input_dim, output_dim = datamodule.train.get_io_dims()


def subset_selector(x, channel, feat, typ):
    """
    Assuming X looks like this:
    [--------------------------------------------------]
    <-------ch1—--------->   ..   <-------ch12--------->
    <-- f1 -->..<-- f6 --> |
    <int><amp>..<int><amp>
    """
    typ_dim = 1
    feat_dim = typ_dim * 2
    channel_dim = feat_dim * 6
    typ_idx = typ_dim * (0 if typ == "interval" else 1)
    feat_idx = feat_dim * feat(x).squeeze(-1)
    ch_idx = channel_dim * channel(x).squeeze(-1)
    idx = ch_idx + feat_idx + typ_idx
    return x.gather(dim=-1, index=idx[..., None])

def subset_selector_all_feat(x, channel, typ):
    x = x.reshape(-1, 12, 6, 2)
    typ_idx = torch.full(size=(x.shape[0],), fill_value=(0 if typ == "interval" else 1), device=x.device)
    ch_idx = channel(x).flatten()[:x.shape[0]] # [B]
    return x[torch.arange(x.shape[0]), ch_idx, :, typ_idx]

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
            lambda: lambda x: torch.full(
                tuple(x.shape[:-1] + (1,)), i, device=x.device
            ),
        )

    # for i in range(6):
    #     dslf.concrete(
    #         f"feature_{i}",
    #         "() -> () -> feature",
    #         lambda: lambda x: torch.full(
    #             tuple(x.shape[:-1] + (1,)), i, device=x.device
    #         ),
    #     )

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
    dslf.concrete(
        "sub",
        "(%num, %num) -> %num",
        lambda x, y: guard_callables(fn=lambda x, y: x - y, x=x, y=y),
    )

    # dslf.parameterized("linear_bool", "() -> $fFeat -> $fFeat", lambda lin: lin, dict(lin=lambda: nn.Linear(input_dim, 1)))
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

    # dslf.concrete("iteA", "(#a -> $fFeat, #a -> #a, #a -> #a) -> #a -> #a", lambda cond, fx, fy: ite_torch(cond, fx, fy))
    dslf.concrete(
        "ite",
        "(#a -> {f, 1}, #a -> #b, #a -> #b) -> #a -> #b",
        lambda cond, fx, fy: ite_torch(cond, fx, fy),
        # lambda cond, fx, fy: guard_callables(fn=partial(ite_torch, condition=cond), if_true=fx, if_else=fy),
    )
    # dslf.concrete("map", "(#a -> #b) -> [#a] -> [#b]", lambda f: lambda x: map_torch(f, x))
    # dslf.concrete("compose", "(#a -> #b, #b -> #c) -> #a -> #c", lambda f, g: lambda x: g(f(x)))
    # dslf.concrete("fold", "((#a, #a) -> #a) -> [#a] -> #a", lambda f: lambda x: fold_torch(f, x))

    dslf.prune_to("($fInp) -> $fOut")
    return dslf.finalize(), dslf.t


print("input_dim", input_dim, "output_dim", output_dim)
dsl, dsl_type_env = ecg_dsl(input_dim, output_dim, max_overall_depth=6)
print("DSL")
print(dsl.render())
# t = ns.TypeDefiner(I=input_dim, O=output_dim)
# t.typedef("fI", "{f, $I}")
# t.typedef("fO", "{f, $O}")
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
                #  t("([$fI]) -> [$fI]"), t("([$fI]) -> [$fO]")
            ],
            near.mlp_factory(hidden_size=10),
        ),
        **near.create_modules(
            "constant_int",
            [dsl_type_env("() -> channel")],
            near.constant_factory(sample_categorical=True),
            known_atom_shapes=dict(channel=(12,), feature=(6,)),
        ),
    },
)


# Configuration for NEARTrainer
trainer_cfg = near.ECGTrainerConfig(
    lr=1e-3,
    max_seq_len=100,
    n_epochs=25,
    num_labels=output_dim,
    train_steps=len(datamodule.train),
    loss_fn="CrossEntropyLoss",
    scheduler="cosine",
    optimizer="adam",
    is_regression=False,
    is_multilabel=False,
)


def validation_cost(node):
    """Calculate the validation cost for a given node using NEARTrainer.

    Args:
        node: The node for which to calculate the validation cost.

    Returns:
        The validation cost as a float. Returns a high cost if the partial
        program is not found.

    Raises:
        near.PartialProgramNotFoundError: If the partial program is not found.
    """
    # Initialize the trainer with the given configuration

    early_stop_callback = EarlyStopping(
        monitor="train_acc", min_delta=1e-2, patience=5, verbose=False, mode="max"
    )
    trainer = Trainer(
        max_epochs=100,
        devices="auto",
        accelerator="cpu",
        # accelerator="gpu",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
    )
    try:
        # Initialize the program
        initialized_p = neural_dsl.initialize(node.program)
    except near.PartialProgramNotFoundError:
        print("Partial program not found\n", render_s_expression(node.program))
        import IPython; IPython.embed()
        # Return a high cost if the partial program is not found
        return 10000.0

    # Compute the model from the initialized program
    if initialized_p.symbol.startswith(neural_dsl.neural_fn_tag):
        model = neural_dsl.compute(initialized_p)
    else:
        del initialized_p
        model = near.TorchProgramModule(dsl=neural_dsl, program=node.program)

    # Initialize NEARTrainer with the model and configuration
    pl_model = near.ECGTrainer(model, config=trainer_cfg)

    try:
        trainer.fit(pl_model, datamodule.train_dataloader())
        trainer.validate(pl_model, datamodule.val_dataloader(), verbose=False)
    except near.TrainingError as e:
        print("Training error\n", e, "\n", render_s_expression(node.program))
        return 10000.0

    # Return the validation loss
    return trainer.callback_metrics["val_acc"].item()

def cost_plus_heuristic(node):
    cost = validation_cost(node) # accuracy [0, 1]
    # n_holes = len(list(ns.all_holes(node.program)))
    return (-1 * cost)


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
        s="({f, $L}) -> {f, $O}", env=ns.TypeDefiner(L=input_dim, O=output_dim)
    ),
    is_goal=checker,
)
iterator = ns.search.bounded_astar(g, cost_plus_heuristic, max_depth=7)
best_program_nodes = []
for node in iterator:
    cost = validation_cost(node)
    best_program_nodes.append((node, cost))

# save the best program nodes
import pickle
with open("best_program_nodes.pkl", "wb") as f:
    pickle.dump(best_program_nodes, f)

# import asyncio

# async def find_best_programs():
#     iterator = ns.search.async_bounded_astar(g, cost_plus_heuristic, max_depth=7)
#     best_program_nodes = []
#     async for node in iterator:
#         cost = validation_cost(node)
#         best_program_nodes.append((node, cost))

#     return best_program_nodes

# loop = asyncio.get_event_loop()
# best_program_nodes = loop.run_until_complete(find_best_programs())
import IPython; IPython.embed()
