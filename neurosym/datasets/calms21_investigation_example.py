from .load_data import _split_dataset, DatasetFromNpy, DatasetWrapper


def calms21_investigation_example(train_seed, **kwargs) -> DatasetWrapper:
    """
    Data example from the CALMS21 - investigation dataset.
    Source for full dataset: https://data.caltech.edu/records/s0vdx-0k302
    Takes a seed and returns a DatasetWrapper object containing the data.

    :param train_seed: Seed for the train data.
    :param kwargs: Additional arguments to pass to the DatasetWrapper constructor.

    :return: DatasetWrapper object containing the CALMS21 - investigation dataset.
    """

    train_data = "data/mice_classification/calms21_task1/train_data.npy"
    train_labels = (
        "data/mice_classification/calms21_task1/train_investigation_labels.npy"
    )
    test_data = "data/mice_classification/calms21_task1/test_data.npy"
    test_labels = "data/mice_classification/calms21_task1/test_investigation_labels.npy"

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
