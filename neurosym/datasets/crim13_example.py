from .load_data import DatasetFromNpy, DatasetWrapper


def crim13_example(train_seed, **kwargs) -> DatasetWrapper:
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
