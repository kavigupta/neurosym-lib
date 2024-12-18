from .load_data import DatasetFromNpy, DatasetWrapper


def crim13_investigation_example(train_seed, **kwargs) -> DatasetWrapper:
    """
    Data example for the CRIM13 dataset. Available in the data/ folder.
    Source for full dataset: https://data.caltech.edu/records/4emt5-b0t10
    Takes a seed and returns a DatasetWrapper object containing the data.

    :param train_seed: Seed for the train data.
    :param kwargs: Additional arguments to pass to the DatasetWrapper constructor.

    :return: DatasetWrapper object containing the CRIM13 dataset.
    """

    train_data = "data/mice_classification/calms21_task1/train_data.npy"
    train_labels = (
        "data/mice_classification/calms21_task1/train_investigation_labels.npy"
    )
    val_data = "data/mice_classification/calms21_task1/val_data.npy"
    val_labels = "data/mice_classification/calms21_task1/val_investigation_labels.npy"

    return DatasetWrapper(
        DatasetFromNpy(
            train_data,
            train_labels,
            train_seed,
        ),
        DatasetFromNpy(
            val_data,
            val_labels,
            None,
        ),
        **kwargs,
    )
