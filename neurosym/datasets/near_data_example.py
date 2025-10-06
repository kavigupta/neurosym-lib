from .load_data import DatasetWrapper, numpy_dataset_from_github


def near_data_example(train_seed, **kwargs) -> DatasetWrapper:
    """
    Data example from the Near library. Imported from Github.
    Takes a seed and returns a DatasetWrapper object containing the data.
    """

    return numpy_dataset_from_github(
        "https://github.com/trishullab/near/tree/master/near_code/data/example",
        "train_ex_data.npy",
        "train_ex_labels.npy",
        "test_ex_data.npy",
        "test_ex_labels.npy",
        **kwargs
    )(train_seed)
