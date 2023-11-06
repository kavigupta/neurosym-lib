import io
import os

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from permacache import permacache


def get_raw_url(github_folder, filename):
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


@permacache("neurosym/data/load_data/load_npy")
def load_npy(path_or_url):
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
    A dataset from an array, loaded from a url.

    TODO test/val split
    """

    def __init__(self, input_url, output_url, seed):
        """
        Parameters
        ----------
        url : str
            The url of the numpy file.
        """
        self.inputs = load_npy(input_url)
        self.outputs = load_npy(output_url)
        assert len(self.inputs) == len(self.outputs)
        if seed is not None:
            self.ordering = np.random.RandomState(seed=seed).permutation(
                len(self.inputs)
            )
        else:
            self.ordering = np.arange(len(self.inputs))

    def get_io_dims(self):
        return self.inputs.shape[-1], len(np.unique(self.outputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return dict(
            inputs=self.inputs[self.ordering[idx]],
            outputs=self.outputs[self.ordering[idx]],
        )


class DatasetWrapper(pl.LightningDataModule):
    def __init__(
        self,
        train: torch.utils.data.Dataset,
        test: torch.utils.data.Dataset,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train = train
        self.test = test
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)


def numpy_dataset_from_github(
    github_url,
    train_input_path,
    train_output_path,
    test_input_path,
    test_output_path,
):
    """
    Load a dataset from a github url.

    Parameters
    ----------
    github_url : str
        The url of the github folder containing the data.
    train_input_path : str
        The path to the training input data.
    train_output_path : str
        The path to the training output data.
    test_input_path : str
        The path to the test input data.
    test_output_path : str
        The path to the test output data.

    Returns
    -------
    dataset : function seed -> DatasetWrapper
        The dataset, as a function of the seed.
    """
    return lambda train_seed: DatasetWrapper(
        DatasetFromNpy(
            get_raw_url(github_url, train_input_path),
            get_raw_url(github_url, train_output_path),
            train_seed,
        ),
        DatasetFromNpy(
            get_raw_url(github_url, test_input_path),
            get_raw_url(github_url, test_output_path),
            None,
        ),
    )
