"""
Use a neural network to train on the ECG dataset task
"""

import logging
import argparse
import os

import numpy as np

import neurosym as ns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV


# from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper

# warnings.filterwarnings("ignore")
# import logging

# logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
# logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
# logging.getLogger("lightning.pytorch.loops.evaluation_loop").setLevel(logging.WARNING)


def load_dataset_npz(features_pth, label_pth):
    assert os.path.exists(features_pth), f"{features_pth} does not exist."
    assert os.path.exists(label_pth), f"{label_pth} does not exist."
    X = np.load(features_pth)
    y = np.load(label_pth)
    return X, y


# def filter_multilabel(split):
#     x_fname = f"data/ecg_multitask_example/x_{split}.npy"
#     y_fname = f"data/ecg_multitask_example/y_{split}.npy"
#     X = np.load(x_fname)
#     y = np.load(y_fname)

#     mask = y.sum(-1) == 1
#     # TODO: change mask to do uniformly at random

#     # filter
#     X = X[mask]
#     y = y[mask]

#     # normalize each column of X
#     # X [B, 144]
#     # X = (X - X.mean(0)) / X.std(0)

#     # save as filtered
#     np.save(x_fname.replace(f"{split}", f"{split}_filtered2"), X)
#     np.save(y_fname.replace(f"{split}", f"{split}_filtered2"), y)


# filter_multilabel("train")
# filter_multilabel("test")


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


def main(args):
    logging.basicConfig(level=logging.INFO)
    input_size, output_size = 144, 9
    datamodule = dataset_factory(args.data_dir, 42)
    X_train = datamodule.train.inputs
    y_train = datamodule.train.outputs
    X_test = datamodule.test.inputs
    y_test = datamodule.test.outputs

    # dtree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier(
        n_estimators=100, max_depth=2, random_state=0
    )
    clf = GridSearchCV(
        random_forest,
        dict(n_estimators=[10, 20, 30, 40, 50], criterion=["gini", "entropy"]),
        cv=5,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    best_clf = clf.best_estimator_
    best_params = clf.best_params_  # [40, 'best']
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(best_params, acc, f1)
    import IPython

    IPython.embed()


# A decision tree classifier shouldn't be so sensitive to data norm:
# No normalization = 35%
# 01 normalization = 14%
# norm normalization = 23%
# norm normalization = 24.96%
# NEAR best program so far: 0.2776


# early_stop_callback = EarlyStopping(
#     monitor="train_acc", min_delta=1e-2, patience=5, verbose=False, mode="max"
# )
# trainer_cfg = near.ECGTrainerConfig(
#     lr=1e-4,
#     max_seq_len=100,
#     n_epochs=100,
#     num_labels=output_dim,
#     train_steps=len(datamodule.train),
#     loss_fn="CrossEntropyLoss",
#     scheduler="none",
#     optimizer="sgd",
#     is_regression=False,
#     is_multilabel=False,
# )
# trainer = Trainer(
#     max_epochs=trainer_cfg.n_epochs,
#     devices="auto",
#     # accelerator="cpu",
#     accelerator="gpu",
#     enable_checkpointing=False,
#     enable_model_summary=False,
#     logger=False,
#     # callbacks=[early_stop_callback],
#     enable_progress_bar=True,
# )

# model = nn.Sequential(
#     nn.Linear(input_dim, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Linear(64, output_dim)
# )

# pl_model = near.ECGTrainer(model, config=trainer_cfg)
# trainer.fit(pl_model, datamodule.train_dataloader())
# metrics = trainer.validate(pl_model, datamodule.val_dataloader(), verbose=False)
# import IPython; IPython.embed()


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
        "--max_workers",
        type=int,
        required=False,
        default=0,
        help="Number of concurrent beams to explore.",
    )
    args = parser.parse_args()

    main(args)
