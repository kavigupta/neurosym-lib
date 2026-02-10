"""
Use a neural network to train on the ECG dataset task
"""
import os
import warnings

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
    x_fname = f"data/ecg_classification/ecg_process/x_{split}.npy"
    y_fname = f"data/ecg_classification/ecg_process/y_{split}.npy"
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
    return ns.datasets.ecg_data_example(
        train_seed=train_seed,
        label_mode="multi",
        is_regression=is_regression,
        batch_size=1000,
        num_workers=n_workers,
    )


datamodule = create_dataset_factory(train_seed=42, is_regression=True, n_workers=0)
# Retrieve input and output dimensions from the training dataset
input_dim, output_dim = datamodule.train.get_io_dims(is_regression=True)


early_stop_callback = EarlyStopping(
    monitor="train_acc", min_delta=1e-2, patience=5, verbose=False, mode="max"
)
trainer_cfg = near.ECGTrainerConfig(
    lr=1e-4,
    max_seq_len=100,
    n_epochs=100,
    num_labels=output_dim,
    train_steps=len(datamodule.train),
    loss_fn="CrossEntropyLoss",
    scheduler="none",
    optimizer="sgd",
    is_regression=False,
    is_multilabel=False,
)
trainer = Trainer(
    max_epochs=trainer_cfg.n_epochs,
    devices="auto",
    # accelerator="cpu",
    accelerator="gpu",
    enable_checkpointing=False,
    enable_model_summary=False,
    logger=False,
    # callbacks=[early_stop_callback],
    enable_progress_bar=True,
)

model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, output_dim)
)

pl_model = near.ECGTrainer(model, config=trainer_cfg)
trainer.fit(pl_model, datamodule.train_dataloader())
metrics = trainer.validate(pl_model, datamodule.val_dataloader(), verbose=False)
import IPython; IPython.embed()
