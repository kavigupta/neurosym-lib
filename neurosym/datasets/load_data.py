import io
import os
from multiprocessing import get_context
from typing import Callable

import numpy as np
import requests
import torch

from neurosym.utils.imports import import_pytorch_lightning

pl = import_pytorch_lightning()


def _get_raw_url(github_folder, filename):
    """
    Get the raw url for a file in a github folder.

    Parameters
    ----------
    github_folder : str
        The github folder containing the file. As a url.
        E.g., github_folder = https://github.com/trishullab/near/tree/master/near_code/data/example
    filename : str
        The name of the file.
            E.g., filename = test_ex_data.npy

    Returns
    -------
    raw_url : str
        The raw url for the file.
        E.g., https://github.com/trishullab/near/raw/master/near_code/data/example/test_ex_data.npy
    """
    raw_url = github_folder.replace("tree", "raw") + "/" + filename
    return raw_url


def _load_npy(path_or_url):
    """
    Load a numpy file from a path or url.

    Parameters
    ----------
    path_or_url : str
        The path or url of the numpy file.

    Returns
    -------
    data : np.ndarray
        The data in the numpy file.
    """
    # pylint: disable=missing-timeout
    if os.path.exists(path_or_url):
        # Load from local path
        data = np.load(path_or_url)
    else:
        data = requests.get(path_or_url).content
        data = np.load(io.BytesIO(data))
    return data


class DatasetFromNpy(torch.utils.data.Dataset):
    """
    A dataset from an array, loaded from a url or local path.

    :param input_url: the url of the numpy file containing the inputs.
    :param output_url: the url of the numpy file containing the outputs.
    :param seed: the seed for the random permutation of the dataset.
    """

    # TODO test/val split

    def __init__(self, input_url, output_url, seed):
        """
        Parameters
        ----------
        url : str
            The url of the numpy file.
        """
        self.inputs = _load_npy(input_url)
        self.outputs = _load_npy(output_url)
        assert len(self.inputs) == len(self.outputs)
        if seed is not None:
            self.ordering = np.random.RandomState(seed=seed).permutation(
                len(self.inputs)
            )
        else:
            self.ordering = np.arange(len(self.inputs))

    def get_io_dims(self, is_regression=False):
        """
        Get the input/output dimensions of the dataset.
        """
        out = self.outputs.shape[-1] if is_regression else len(np.unique(self.outputs))
        return self.inputs.shape[-1], out

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return dict(
            inputs=self.inputs[self.ordering[idx]],
            outputs=self.outputs[self.ordering[idx]],
        )


class DatasetWrapper(pl.LightningDataModule):
    """
    Dataset wrapper for PyTorch Lightning, with a train/test split. Wraps two
        torch.utils.data.Dataset objects.
    """

    def __init__(
        self,
        train: torch.utils.data.Dataset,
        test: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train = train
        self.test = test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            multiprocessing_context=(
                get_context("loky") if (self.num_workers > 0) else None
            ),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            multiprocessing_context=(
                get_context("loky") if (self.num_workers > 0) else None
            ),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            multiprocessing_context=(
                get_context("loky") if (self.num_workers > 0) else None
            ),
        )


def numpy_dataset_from_github(
    github_url: str,
    train_input_path: str,
    train_output_path: str,
    test_input_path: str,
    test_output_path: str,
    **kwargs,
) -> Callable[[int], DatasetWrapper]:
    """
    Load a dataset from a github url.

    :param github_url: the url of the github folder containing the data.
    :param train_input_path: the path to the training input data.
    :param train_output_path: the path to the training output data.
    :param test_input_path: the path to the test input data.
    :param test_output_path: the path to the test output data

    :return: a function that takes a seed and returns a DatasetWrapper.
    """
    return lambda train_seed: DatasetWrapper(
        DatasetFromNpy(
            _get_raw_url(github_url, train_input_path),
            _get_raw_url(github_url, train_output_path),
            train_seed,
        ),
        DatasetFromNpy(
            _get_raw_url(github_url, test_input_path),
            _get_raw_url(github_url, test_output_path),
            None,
        ),
        **kwargs,
    )
