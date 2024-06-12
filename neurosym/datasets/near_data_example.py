from typing import Callable

from .load_data import DatasetWrapper, numpy_dataset_from_github


class near_data_example:
    """
    Data example from the Near library. Imported from Github.

    :field data: A function that takes a seed and returns a DatasetWrapper object containing the data.
    """

    data: Callable[[int], DatasetWrapper] = numpy_dataset_from_github(
        "https://github.com/trishullab/near/tree/master/near_code/data/example",
        "train_ex_data.npy",
        "train_ex_labels.npy",
        "test_ex_data.npy",
        "test_ex_labels.npy",
    )
