from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper


def const_data_example(train_seed, is_regression=True):
    """Creates a dataset for constant value regression/classification examples.

    This function wraps the training and testing datasets with the
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
            "data/constant_example/train_ex_data.npy",
            "data/constant_example/train_ex_labels.npy",
            seed=train_seed,
            is_regression=is_regression,
        ),
        DatasetFromNpy(
            "data/constant_example/test_ex_data.npy",
            "data/constant_example/test_ex_labels.npy",
            seed=0,
            is_regression=is_regression,
        ),
        batch_size=200,
    )
