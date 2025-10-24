# pylint: disable=duplicate-code
from .load_data import DatasetFromNpy, DatasetWrapper, _split_dataset


def flyvfly_data_example(train_seed, **kwargs) -> DatasetWrapper:
    """
    Data example for the Fly v. Fly dataset. Available in the data/ folder.
    Source for full dataset: https://data.caltech.edu/records/zrznw-w7386
    Takes a seed and returns a DatasetWrapper object containing the data.

    :param train_seed: Seed for the train data.
    :param kwargs: Additional arguments to pass to the DatasetWrapper constructor.

    :return: DatasetWrapper object containing the Fly v. Fly dataset.
    """

    train_data = "data/fruitflies_classification/flyvfly/train_flyvfly_data.npz"
    train_labels = "data/fruitflies_classification/flyvfly/train_flyvfly_labels.npz"
    valid_data = "data/fruitflies_classification/flyvfly/val_flyvfly_data.npz"
    valid_labels = "data/fruitflies_classification/flyvfly/val_flyvfly_labels.npz"
    test_data = "data/fruitflies_classification/flyvfly/test_flyvfly_data.npz"
    test_labels = "data/fruitflies_classification/flyvfly/test_flyvfly_labels.npz"

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
