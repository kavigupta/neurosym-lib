"""
Use a neural network to train on the ECG dataset task
"""

import os
import warnings

import numpy as np
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.examples import near
import neurosym as ns
from neurosym.examples import near


warnings.filterwarnings("ignore")
import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.loops.evaluation_loop").setLevel(logging.WARNING)


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


datamodule = create_dataset_factory(train_seed=42, is_regression=False, n_workers=0)
# Retrieve input and output dimensions from the training dataset
input_dim, output_dim = datamodule.train.get_io_dims()


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
    nn.Linear(64, output_dim),
)

pl_model = near.ECGTrainer(model, config=trainer_cfg)
trainer.fit(pl_model, datamodule.train_dataloader())
metrics = trainer.validate(pl_model, datamodule.val_dataloader(), verbose=False)
import IPython

IPython.embed()
