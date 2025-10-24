# pylint: disable=duplicate-code
from .load_data import DatasetFromNpy, DatasetWrapper, _split_dataset


def crim13_data_example(train_seed, behavior='sniff', **kwargs) -> DatasetWrapper:
    """
    Data example for the CRIM13 dataset. Available in the data/ folder.
    Source for full dataset: https://data.caltech.edu/records/4emt5-b0t10
    Takes a seed and returns a DatasetWrapper object containing the data.

    :param train_seed: Seed for the train data.
    :param kwargs: Additional arguments to pass to the DatasetWrapper constructor.

    :return: DatasetWrapper object containing the CRIM13 dataset.
    """

    train_data = "data/mice_classification/crim13/train_crim13_data.npy"
    test_data = "data/mice_classification/crim13/test_crim13_data.npy"
    if behavior == 'sniff':
        train_labels = "data/mice_classification/crim13/train_crim13_labels.npy"
        test_labels = "data/mice_classification/crim13/test_crim13_labels.npy"
    elif behavior == 'other':
        train_labels = "data/mice_classification/crim13/train_crim13_labels_other.npy"
        test_labels = "data/mice_classification/crim13/test_crim13_labels_other.npy"
    else:
        raise ValueError(f"Unknown behavior: {behavior}")

    train_dataset = DatasetFromNpy(
        train_data,
        train_labels,
        train_seed,
    )
    train_dataset, val_dataset = _split_dataset(
        train_dataset, val_fraction=0.15, seed=train_seed
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
