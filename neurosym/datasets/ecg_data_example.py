from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper


def ecg_data_example(
    train_seed,
    label_mode="multi",
    is_regression=None,
    data_dir="data/ecg_classification/ecg",
    **kwargs,
):
    """Creates a dataset for ECG classification experiments.

    This follows the same standardized file layout as the basketball dataset:
    train/valid/test splits with *_data.npz and *_labels_{single|multi}.npz files.

    Args:
        train_seed (int): Seed for dataset shuffling.
        label_mode (str): Which label set to use. One of {"single", "multi"}.
            - "single": integer class labels (N, 1)
            - "multi": multi-hot labels (N, C)
        is_regression (bool | None): Whether to treat outputs as regression vectors.
            Defaults to True for multi-label and False for single-label.
        data_dir (str): Base directory containing the standardized ECG files.

    Returns:
        DatasetWrapper: An instance containing train/valid/test datasets.
    """
    if label_mode not in {"single", "multi"}:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    if is_regression is None:
        is_regression = label_mode == "multi"

    label_suffix = "single" if label_mode == "single" else "multi"

    return DatasetWrapper(
        DatasetFromNpy(
            f"{data_dir}/train_ecg_data.npz",
            f"{data_dir}/train_ecg_labels_{label_suffix}.npz",
            seed=train_seed,
            is_regression=is_regression,
        ),
        DatasetFromNpy(
            f"{data_dir}/test_ecg_data.npz",
            f"{data_dir}/test_ecg_labels_{label_suffix}.npz",
            seed=0,
            is_regression=is_regression,
        ),
        val=DatasetFromNpy(
            f"{data_dir}/valid_ecg_data.npz",
            f"{data_dir}/valid_ecg_labels_{label_suffix}.npz",
            seed=0,
            is_regression=is_regression,
        ),
        **kwargs,
    )
