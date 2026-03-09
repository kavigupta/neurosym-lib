from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurosym.datasets.torch_ecg_data_example import (
    _compute_random_split_indices,
    _extract_records_with_optional_split,
    _signal_to_flat_features,
    torch_ecg_data_example,
)


def test_signal_to_flat_features_shape():
    signal = np.random.randn(12, 500).astype(np.float32)
    features = _signal_to_flat_features(signal)
    assert features.shape == (144,)
    assert features.dtype == np.float32


def test_signal_to_flat_features_single_channel_input():
    signal = np.random.randn(500).astype(np.float32)
    features = _signal_to_flat_features(signal)
    assert features.shape == (144,)


def test_random_split_indices_non_overlapping():
    train_idx, val_idx, test_idx = _compute_random_split_indices(
        n_records=100,
        split_seed=42,
        val_fraction=0.15,
        test_fraction=0.15,
    )
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 100


def test_extract_records_with_explicit_train_test_split():
    records = {
        "train": ["A0001", "A0002"],
        "val": ["A0003"],
        "test": ["A0004", "A0005"],
    }
    train_records, val_records, test_records = _extract_records_with_optional_split(
        records
    )
    assert train_records == ["A0001", "A0002"]
    assert val_records == ["A0003"]
    assert test_records == ["A0004", "A0005"]


def test_torch_ecg_data_example_missing_dependency():
    if importlib.util.find_spec("torch_ecg") is not None:
        pytest.skip("torch_ecg is installed; missing-dependency path not applicable.")
    with pytest.raises(ImportError, match="torch-ecg is required"):
        torch_ecg_data_example(
            train_seed=0,
            dataset_name="CPSC2018",
            max_records=100,
        )
