# pylint: disable=duplicate-code
from .load_data import DatasetFromNpy, DatasetWrapper


def bball_data_example(train_seed, **kwargs) -> DatasetWrapper:
    """
    Data example for the Basketball dataset. Available in the data/ folder.
    Source for full dataset: @TODO: add source link
    Takes a seed and returns a DatasetWrapper object containing the data.

    :param train_seed: Seed for the train data.
    :param kwargs: Additional arguments to pass to the DatasetWrapper constructor.

    :return: DatasetWrapper object containing the Basketball dataset.
    """

    train_data = "data/basketball_classification/bball/train_bball_data.npz"
    train_labels = "data/basketball_classification/bball/train_bball_labels.npz"
    valid_data = "data/basketball_classification/bball/valid_bball_data.npz"
    valid_labels = "data/basketball_classification/bball/valid_bball_labels.npz"
    test_data = "data/basketball_classification/bball/test_bball_data.npz"
    test_labels = "data/basketball_classification/bball/test_bball_labels.npz"

    train_dataset = DatasetFromNpy(
        train_data,
        train_labels,
        train_seed,
    )
    val_dataset = DatasetFromNpy(
        valid_data,
        valid_labels,
        None,
    )

    # pylint: disable=duplicate-code
    return DatasetWrapper(
        train_dataset,
        DatasetFromNpy(
            test_data,
            test_labels,
            None,
        ),
        val_dataset,
        **kwargs,
    )
    # pylint: enable=duplicate-code
