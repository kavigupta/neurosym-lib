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

LEAD_GROUPS = {
    "inferior": ("II", "III", "aVF"),
    "lateral_limb": ("I", "aVL"),
    "septal": ("V1", "V2"),
    "anterior": ("V3", "V4"),
    "lateral_precordial": ("V5", "V6"),
}
LEAD_GROUP_NAMES = tuple(LEAD_GROUPS.keys())
NUM_LEAD_GROUPS = len(LEAD_GROUPS)
NUM_GROUPED_CHANNELS = NUM_LEAD_GROUPS + 1  # 6 (5 lead groups + 1 global)

# Feature-type groupings (Phase 1-style): group features by type across all leads
_AMP_FEATURES = ("P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp")
_INT_FEATURES = (
    "PQ_Int",
    "PR_Int",
    "QRS_Dur",
    "QT_Int",
    "QT_IntCorr",
    "P_DurFull",
    "T_DurFull",
)
_ST_FEATURES = ("ST_Elev",)
_MORPH_FEATURES = ("P_Morph",)
# Feature-major slice sizes: amp=60, int=84, st=12, morph=12, global=9. Total = 177.
NUM_AMP_FEATURES = len(_AMP_FEATURES) * NUM_LEADS  # 60
NUM_INT_FEATURES = len(_INT_FEATURES) * NUM_LEADS  # 84
NUM_ST_FEATURES = len(_ST_FEATURES) * NUM_LEADS  # 12
NUM_MORPH_FEATURES = len(_MORPH_FEATURES) * NUM_LEADS  # 12
# NUM_GLOBAL_FEATURES already defined = 9
NUM_FLAT_FEATURES = (
    NUM_AMP_FEATURES
    + NUM_INT_FEATURES
    + NUM_ST_FEATURES
    + NUM_MORPH_FEATURES
    + NUM_GLOBAL_FEATURES
)  # 177


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


def _reshape_to_grouped_channels(x):
    """Reshape flat (B, 177) features to (B, 6, 14) grouped channel layout.

    Channels 0-4: 5 anatomical lead groups, each averaging 14 per-lead features
                   across member leads.
    Channel 5: 9 global features placed in first 9 positions, zero-padded to 14.

    aVR is excluded (right-sided, less diagnostic for the 5 superclasses).
    """
    n = x.shape[0]
    lead_to_idx = {name: i for i, name in enumerate(LEAD_NAMES)}

    grouped = np.zeros((n, NUM_GROUPED_CHANNELS, NUM_PER_LEAD_FEATURES), dtype=x.dtype)

    for g_idx, group_name in enumerate(LEAD_GROUP_NAMES):
        lead_names = LEAD_GROUPS[group_name]
        lead_indices = [lead_to_idx[name] for name in lead_names]
        group_features = np.stack(
            [
                x[:, idx * NUM_PER_LEAD_FEATURES : (idx + 1) * NUM_PER_LEAD_FEATURES]
                for idx in lead_indices
            ],
            axis=0,
        )  # (num_leads_in_group, B, 14)
        grouped[:, g_idx, :] = group_features.mean(axis=0)

    # Global features: last 9 columns, zero-padded to 14
    n_lead_feats = NUM_LEADS * NUM_PER_LEAD_FEATURES  # 168
    globals_raw = x[:, n_lead_feats:]  # (B, 9)
    grouped[:, NUM_LEAD_GROUPS, :NUM_GLOBAL_FEATURES] = globals_raw

    return grouped


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


def _ordered_feature_columns(feature_major, feature_groups):
    """Return feature-column order for the requested layout.

    Feature-type-major groups all leads per feature type; otherwise lead-major.
    """
    ordered_cols = []
    if feature_major or feature_groups:
        for feat_group in (_AMP_FEATURES, _INT_FEATURES, _ST_FEATURES, _MORPH_FEATURES):
            for feat in feat_group:
                for lead in LEAD_NAMES:
                    ordered_cols.append(f"{feat}_{lead}")
    else:
        for lead in LEAD_NAMES:
            for feat in _PER_LEAD_FEATURES:
                ordered_cols.append(f"{feat}_{lead}")
    ordered_cols.extend(_GLOBAL_FEATURES)
    return ordered_cols


def _reshape_features(arrays, grouped, feature_major, feature_groups):
    """Reshape flat (B, 177) arrays into the layout requested by the flags."""
    if feature_groups:
        # Keep flat (B, 177); FeatureGroupUnpackModule slices into 5 typed args.
        return arrays
    if feature_major:
        # Single channel containing all 177 features (B, 1, 177).
        return tuple(arr.reshape(arr.shape[0], 1, -1) for arr in arrays)
    if grouped:
        return tuple(_reshape_to_grouped_channels(arr) for arr in arrays)
    return tuple(_reshape_to_channels(arr) for arr in arrays)


def _attach_layout_metadata(datamodule, grouped, feature_major, feature_groups):
    """Attach layout-specific attributes to the datamodule."""
    if grouped:
        datamodule.num_leads = NUM_LEAD_GROUPS
        datamodule.num_global_features = 1
        datamodule.lead_names = list(LEAD_GROUP_NAMES)
    if feature_major:
        # Single flat channel of all 177 features
        datamodule.num_leads = 1
        datamodule.num_global_features = 0
        datamodule.features_per_lead = NUM_FLAT_FEATURES  # 177
        datamodule.lead_names = ["flat_177"]
    if feature_groups:
        # Heterogeneous-typed lambda mode: 5 distinct feature groups.
        # Data stays as flat (B, 177); FeatureGroupUnpackModule slices into 5 args.
        datamodule.feature_group_slices = {
            "amp": (0, 60),
            "int": (60, 144),
            "st": (144, 156),
            "morph": (156, 168),
            "global": (168, 177),
        }
        datamodule.feature_group_sizes = (60, 84, 12, 12, 9)
        datamodule.num_feature_groups = 5


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
    grouped: bool = False,
    feature_major: bool = False,
    feature_groups: bool = False,
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

    feat_df = feat_df[_ordered_feature_columns(feature_major, feature_groups)]

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

    # Reshape from flat (B, 177) to the channelised layout requested by the flags
    x_train, x_val, x_test = _reshape_features(
        (x_train, x_val, x_test), grouped, feature_major, feature_groups
    )

    if verbose:
        log(
            f"ECG dataset loaded: {x_train.shape[0]} train, "
            f"{x_val.shape[0]} val, {x_test.shape[0]} test, "
            f"shape={x_train.shape[1:]} per sample, "
            f"label_mode={label_mode}, grouped={grouped}, "
            f"feature_major={feature_major}, feature_groups={feature_groups}"
        )

    datamodule = _build_datamodule(
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
    _attach_layout_metadata(datamodule, grouped, feature_major, feature_groups)
    return datamodule
