# pylint: disable=too-many-locals
"""ECG dataset loader with ECGDeli pre-extracted features.

Downloads PTB-XL and PTB-XL+ from PhysioNet, parses ECGDeli feature CSVs,
and builds structured feature groups for use with the attention ECG DSL.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from neurosym.datasets.load_data import DatasetFromNpy, DatasetWrapper
from neurosym.utils.logging import log

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PTB_XL_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# 14 per-lead feature types
_PER_LEAD_FEATURES = (
    "P_Amp",
    "Q_Amp",
    "R_Amp",
    "S_Amp",
    "T_Amp",
    "PQ_Int",
    "PR_Int",
    "QRS_Dur",
    "QT_Int",
    "QT_IntCorr",
    "P_DurFull",
    "T_DurFull",
    "ST_Elev",
    "P_Morph",
)

_GLOBAL_FEATURES = (
    "PQ_Int_Global",
    "PR_Int_Global",
    "P_Dur_Global",
    "QRS_Dur_Global",
    "QT_Int_Global",
    "QT_IntFramingham_Global",
    "RR_Mean_Global",
    "T_Dur_Global",
    "HA__Global",
)

NUM_LEADS = len(LEAD_NAMES)
NUM_PER_LEAD_FEATURES = len(_PER_LEAD_FEATURES)
NUM_GLOBAL_FEATURES = len(_GLOBAL_FEATURES)
NUM_CHANNELS = NUM_LEADS + NUM_GLOBAL_FEATURES


# --------------------------------------------------------------------------- #
# Lazy pandas import
# --------------------------------------------------------------------------- #


def _import_pandas():
    try:
        import pandas as pd  # pylint: disable=import-outside-toplevel

        return pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for ECG dataset loading. "
            "Install it with `pip install pandas`."
        ) from exc


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #


def _download_ptb_xl_plus(data_dir: str, verbose: int = 1) -> None:
    """Download PTB-XL and PTB-XL+ from PhysioNet if not already present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ptbxl_dir = data_path / "ptb-xl" / "1.0.3"
    ptbxl_plus_dir = data_path / "ptb-xl-plus" / "1.0.1"

    if not ptbxl_dir.exists():
        if verbose:
            log("Downloading PTB-XL 1.0.3 from PhysioNet...")
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-c",
                "-np",
                "-nH",
                "--cut-dirs=1",
                "-P",
                str(data_path),
                "https://physionet.org/files/ptb-xl/1.0.3/",
            ],
            check=True,
        )

    if not ptbxl_plus_dir.exists():
        if verbose:
            log("Downloading PTB-XL+ 1.0.1 from PhysioNet...")
        subprocess.run(
            [
                "wget",
                "-r",
                "-N",
                "-c",
                "-np",
                "-nH",
                "--cut-dirs=1",
                "-P",
                str(data_path),
                "https://physionet.org/files/ptb-xl-plus/1.0.1/",
            ],
            check=True,
        )

    if verbose:
        log("PTB-XL and PTB-XL+ data ready.")


# --------------------------------------------------------------------------- #
# Metadata & feature loading
# --------------------------------------------------------------------------- #


def _load_metadata(data_dir: str):
    """Parse ptbxl_database.csv and scp_statements.csv, return metadata DataFrame."""
    pd = _import_pandas()
    data_path = Path(data_dir)

    ptbxl_csv = data_path / "ptb-xl" / "1.0.3" / "ptbxl_database.csv"
    scp_csv = data_path / "ptb-xl" / "1.0.3" / "scp_statements.csv"

    meta = pd.read_csv(ptbxl_csv, index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(scp_csv, index_col=0)
    return meta, scp


def _load_ecgdeli_features(data_dir: str, record_ids):
    """Load ECGDeli feature CSV and extract median-only columns (177 features)."""
    pd = _import_pandas()
    data_path = Path(data_dir)

    # Try multiple known paths for the ECGDeli features CSV
    candidates = [
        data_path / "ptb-xl-plus" / "1.0.1" / "features" / "ecgdeli_features.csv",
        data_path / "ptb-xl-plus" / "1.0.1" / "labels" / "ptbxl_ecgdeli.csv",
        data_path
        / "ptb-xl-plus"
        / "1.0.1"
        / "features"
        / "ecgdeli"
        / "ecgdeli_features.csv",
    ]
    feat_csv = None
    for candidate in candidates:
        if candidate.exists():
            feat_csv = candidate
            break
    if feat_csv is None:
        raise FileNotFoundError(
            f"ECGDeli features CSV not found. Searched: {[str(c) for c in candidates]}"
        )

    feat_df = pd.read_csv(feat_csv, index_col="ecg_id")

    # Keep only bare columns (no _iqr, _count suffixes)
    median_cols = [
        c
        for c in feat_df.columns
        if not c.endswith("_iqr") and not c.endswith("_count")
    ]
    feat_df = feat_df[median_cols]

    # Align with requested record_ids
    feat_df = feat_df.loc[feat_df.index.intersection(record_ids)]
    return feat_df


def _make_labels(metadata, scp_statements, label_mode: str):
    """Build label arrays and return (labels, valid_mask).

    For single-label mode, records with != 1 superclass label are masked out.
    For multi-label mode, all records are kept.
    """
    superclass_map = {}
    for code in scp_statements.index:
        row = scp_statements.loc[code]
        sc = row.get("diagnostic_class", None)
        if sc in PTB_XL_SUPERCLASSES:
            superclass_map[code] = sc

    n_records = len(metadata)
    n_classes = len(PTB_XL_SUPERCLASSES)
    class_to_idx = {c: i for i, c in enumerate(PTB_XL_SUPERCLASSES)}

    multi_hot = np.zeros((n_records, n_classes), dtype=np.float32)
    for i, (_, row) in enumerate(metadata.iterrows()):
        scp_codes = row["scp_codes"]
        for code, likelihood in scp_codes.items():
            if likelihood > 0 and code in superclass_map:
                cls_idx = class_to_idx[superclass_map[code]]
                multi_hot[i, cls_idx] = 1.0

    if label_mode == "single":
        # Filter to records with exactly 1 superclass
        label_counts = multi_hot.sum(axis=1)
        valid_mask = label_counts == 1
        single_labels = multi_hot[valid_mask].argmax(axis=1).astype(np.int64)
        return single_labels, valid_mask
    # multi
    return multi_hot, np.ones(n_records, dtype=bool)


# --------------------------------------------------------------------------- #
# Imputation & normalization
# --------------------------------------------------------------------------- #


def _impute_and_normalize(x_train, x_val, x_test, normalize):
    """Impute NaNs with training-set column median, optionally z-score normalize."""
    train_medians = np.nanmedian(x_train, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    for arr in (x_train, x_val, x_test):
        nan_mask = np.isnan(arr)
        arr[nan_mask] = np.take(train_medians, np.where(nan_mask)[1])

    if normalize:
        train_mean = x_train.mean(axis=0)
        train_std = x_train.std(axis=0)
        train_std[train_std == 0] = 1.0
        x_train = (x_train - train_mean) / train_std
        x_val = (x_val - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std

    return x_train, x_val, x_test


def _reshape_to_channels(x):
    """Reshape flat (B, 177) features to (B, 21, 14) channel layout.

    Channels 0-11 are the 12 leads (14 features each, already contiguous).
    Channels 12-20 are the 9 global features, each a scalar zero-padded to 14.
    """
    n = x.shape[0]
    n_lead_feats = NUM_LEADS * NUM_PER_LEAD_FEATURES  # 168
    leads = x[:, :n_lead_feats].reshape(n, NUM_LEADS, NUM_PER_LEAD_FEATURES)
    globals_raw = x[:, n_lead_feats:]  # (B, 9)
    globals_padded = np.zeros(
        (n, NUM_GLOBAL_FEATURES, NUM_PER_LEAD_FEATURES), dtype=x.dtype
    )
    globals_padded[:, :, 0] = globals_raw
    return np.concatenate([leads, globals_padded], axis=1)


def _build_datamodule(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    train_seed,
    is_regression,
    columns,
    **kwargs,
):
    """Build a DatasetWrapper with attached metadata."""
    train_ds = DatasetFromNpy.from_arrays(x_train, y_train, seed=train_seed)
    val_ds = DatasetFromNpy.from_arrays(x_val, y_val, seed=0)
    test_ds = DatasetFromNpy.from_arrays(x_test, y_test, seed=0)
    train_ds.is_regression = is_regression
    val_ds.is_regression = is_regression
    test_ds.is_regression = is_regression

    datamodule = DatasetWrapper(train_ds, test_ds, val=val_ds, **kwargs)

    datamodule.class_names = list(PTB_XL_SUPERCLASSES)
    datamodule.lead_names = list(LEAD_NAMES)
    datamodule.feature_columns = columns
    datamodule.num_leads = NUM_LEADS
    datamodule.features_per_lead = NUM_PER_LEAD_FEATURES
    datamodule.num_global_features = NUM_GLOBAL_FEATURES
    return datamodule


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #


def ecg_data_example(
    train_seed: int,
    label_mode: str = "single",
    is_regression: bool | None = None,
    data_dir: str = "data/ecg",
    normalize: bool = True,
    verbose: int = 1,
    **kwargs: Any,
) -> DatasetWrapper:
    """Load the ECG dataset with ECGDeli pre-extracted features.

    Downloads data from PhysioNet if not present. Uses the standard PTB-XL
    evaluation split: folds 1-8 train, fold 9 validation, fold 10 test.

    :param train_seed: Seed for dataset shuffling.
    :param label_mode: ``'single'`` (filter multi-label) or ``'multi'`` (multi-hot).
    :param is_regression: Whether outputs are regression targets.
        Defaults to True for multi-label, False for single-label.
    :param data_dir: Base directory for downloaded data.
    :param normalize: Whether to z-score normalize features (fit on train).
    :param verbose: Verbosity level (0=silent, 1=progress).
    :param kwargs: Extra keyword arguments passed to ``DatasetWrapper``.
    :return: A ``DatasetWrapper`` with train/val/test splits and attached metadata.
    """
    if label_mode not in {"single", "multi"}:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    if is_regression is None:
        is_regression = label_mode == "multi"

    _download_ptb_xl_plus(data_dir, verbose=verbose)

    if verbose:
        log("Loading ECG metadata...")
    metadata, scp_statements = _load_metadata(data_dir)

    if verbose:
        log("Loading ECGDeli features...")
    feat_df = _load_ecgdeli_features(data_dir, metadata.index)

    # Align metadata to available features
    common_ids = metadata.index.intersection(feat_df.index)
    metadata = metadata.loc[common_ids]
    feat_df = feat_df.loc[common_ids]

    labels, valid_mask = _make_labels(metadata, scp_statements, label_mode)

    # Reorder columns to lead-major layout:
    # [lead1_feat1, ..., lead1_feat14, lead2_feat1, ..., lead12_feat14, g1, ..., g9]
    ordered_cols = []
    for lead in LEAD_NAMES:
        for feat in _PER_LEAD_FEATURES:
            ordered_cols.append(f"{feat}_{lead}")
    ordered_cols.extend(_GLOBAL_FEATURES)
    feat_df = feat_df[ordered_cols]

    columns = list(feat_df.columns)
    features = feat_df.values.astype(np.float32)[valid_mask]
    metadata_filtered = metadata[valid_mask]

    # Standard PTB-XL split
    folds = metadata_filtered["strat_fold"].values
    splits = {
        "train": folds <= 8,
        "val": folds == 9,
        "test": folds == 10,
    }
    x_train, x_val, x_test = (features[splits[k]] for k in ("train", "val", "test"))
    y_train, y_val, y_test = (labels[splits[k]] for k in ("train", "val", "test"))

    x_train, x_val, x_test = _impute_and_normalize(x_train, x_val, x_test, normalize)

    # Reshape from flat (B, 177) to channelised (B, 21, 14):
    #   channels 0-11: 12 leads, each with 14 per-lead features
    #   channels 12-20: 9 global features, each zero-padded to 14
    x_train, x_val, x_test = (
        _reshape_to_channels(arr) for arr in (x_train, x_val, x_test)
    )

    if verbose:
        log(
            f"ECG dataset loaded: {x_train.shape[0]} train, "
            f"{x_val.shape[0]} val, {x_test.shape[0]} test, "
            f"{x_train.shape[1:]} per sample "
            f"({NUM_CHANNELS} channels x {NUM_PER_LEAD_FEATURES} features), "
            f"label_mode={label_mode}"
        )

    return _build_datamodule(
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        train_seed,
        is_regression,
        columns,
        **kwargs,
    )
