#!/usr/bin/env python3
"""
Rebuild and verify ECG dataset provenance.

This script reproduces `data/ecg_classification/ecg/*.npz` from
`data/ecg_classification/ecg_process/{x,y}_{train,test}.npy` using the exact
logic in `notebooks/get_ecg_datasets.ipynb`, checks fingerprint evidence
for CPSC2018 origin, and computes explicit similarity diagnostics.

Primary references:
- CPSC2018 dataset overview (6877 records, 9 classes):
  https://torch-ecg.readthedocs.io/en/latest/api/generated/torch_ecg.databases.CPSC2018.html
- Multi-label class totals (Table V):
  https://openreview.net/attachment?id=vwzHeWFM4Q&name=supplementary_material
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


# Table V (supplementary) total multi-label counts for CPSC2018
# class order: SNR, AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE
EXPECTED_CPSC2018_MULTI_COUNTS = np.array([918, 1221, 722, 236, 1857, 616, 700, 869, 220])
EXPECTED_CPSC2018_TOTAL_RECORDS = 6877
EXPECTED_CPSC2018_NUM_CLASSES = 9
EXPECTED_CPSC2018_SPLIT = {"train_raw": 4813, "test_raw": 2064}

DATA_SIMILARITY_THRESHOLDS = {
    "atol": 1e-6,
    "rtol": 1e-5,
    "min_close_rate": 0.999,
    "max_rmse": 1e-5,
    "min_cosine_similarity": 0.999,
}
LABEL_SIMILARITY_THRESHOLDS = {
    "atol": 0.0,
    "rtol": 0.0,
    "min_close_rate": 1.0,
    "max_rmse": 0.0,
    "min_cosine_similarity": 1.0,
}
CPSC_SIMILARITY_THRESHOLDS = {
    "min_class_count_cosine_similarity": 0.999,
    "max_class_count_relative_mae": 0.01,
    "max_total_records_relative_error": 0.001,
    "max_split_relative_error": 0.01,
    "max_num_classes_abs_error": 0.0,
}


@dataclass
class StandardizedDataset:
    train_x: np.ndarray
    valid_x: np.ndarray
    test_x: np.ndarray
    train_y_multi: np.ndarray
    valid_y_multi: np.ndarray
    test_y_multi: np.ndarray
    train_y_single: np.ndarray
    valid_y_single: np.ndarray
    test_y_single: np.ndarray


def _build_standardized_from_raw(
    raw_dir: Path,
    normalize_per_split: bool = True,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> StandardizedDataset:
    x_train = np.load(raw_dir / "x_train.npy")
    y_train = np.load(raw_dir / "y_train.npy")
    x_test = np.load(raw_dir / "x_test.npy")
    y_test = np.load(raw_dir / "y_test.npy")

    if normalize_per_split:
        train_min = x_train.min(0)
        train_max = x_train.max(0)
        train_den = train_max - train_min
        train_den[train_den == 0] = 1.0
        x_train = (x_train - train_min) / train_den

        test_min = x_test.min(0)
        test_max = x_test.max(0)
        test_den = test_max - test_min
        test_den[test_den == 0] = 1.0
        x_test = (x_test - test_min) / test_den

    rs = np.random.RandomState(seed)
    indices = rs.permutation(len(x_train))
    val_size = int(len(x_train) * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    y_train_multi = y_train.astype(np.float32)
    y_val_multi = y_val.astype(np.float32)
    y_test_multi = y_test.astype(np.float32)

    y_train_single = y_train_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
    y_val_single = y_val_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
    y_test_single = y_test_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)

    return StandardizedDataset(
        train_x=x_train.astype(np.float32),
        valid_x=x_val.astype(np.float32),
        test_x=x_test.astype(np.float32),
        train_y_multi=y_train_multi,
        valid_y_multi=y_val_multi,
        test_y_multi=y_test_multi,
        train_y_single=y_train_single,
        valid_y_single=y_val_single,
        test_y_single=y_test_single,
    )


def _write_standardized(ds: StandardizedDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train_ecg_data.npz", ds.train_x)
    np.savez_compressed(out_dir / "valid_ecg_data.npz", ds.valid_x)
    np.savez_compressed(out_dir / "test_ecg_data.npz", ds.test_x)
    np.savez_compressed(out_dir / "train_ecg_labels_multi.npz", ds.train_y_multi)
    np.savez_compressed(out_dir / "valid_ecg_labels_multi.npz", ds.valid_y_multi)
    np.savez_compressed(out_dir / "test_ecg_labels_multi.npz", ds.test_y_multi)
    np.savez_compressed(out_dir / "train_ecg_labels_single.npz", ds.train_y_single)
    np.savez_compressed(out_dir / "valid_ecg_labels_single.npz", ds.valid_y_single)
    np.savez_compressed(out_dir / "test_ecg_labels_single.npz", ds.test_y_single)


def _expected_standardized_arrays(ds: StandardizedDataset) -> dict[str, np.ndarray]:
    return {
        "train_ecg_data.npz": ds.train_x,
        "valid_ecg_data.npz": ds.valid_x,
        "test_ecg_data.npz": ds.test_x,
        "train_ecg_labels_multi.npz": ds.train_y_multi,
        "valid_ecg_labels_multi.npz": ds.valid_y_multi,
        "test_ecg_labels_multi.npz": ds.test_y_multi,
        "train_ecg_labels_single.npz": ds.train_y_single,
        "valid_ecg_labels_single.npz": ds.valid_y_single,
        "test_ecg_labels_single.npz": ds.test_y_single,
    }


def _safe_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 and y_norm == 0.0:
        return 1.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def _compute_array_similarity(
    expected: np.ndarray,
    observed: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, object]:
    exact_match = bool(np.array_equal(expected, observed))
    shape_equal = expected.shape == observed.shape
    result: dict[str, object] = {
        "exact_match": exact_match,
        "shape_equal": shape_equal,
        "shape_expected": list(expected.shape),
        "shape_observed": list(observed.shape),
        "dtype_expected": str(expected.dtype),
        "dtype_observed": str(observed.dtype),
        "thresholds": dict(thresholds),
    }

    if not shape_equal:
        result.update(
            {
                "num_values": 0,
                "close_rate": 0.0,
                "mae": None,
                "rmse": None,
                "max_abs_error": None,
                "cosine_similarity": None,
                "centered_cosine_similarity": None,
                "similar_enough": False,
            }
        )
        return result

    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    observed_flat = observed.astype(np.float64, copy=False).reshape(-1)
    num_values = int(expected_flat.size)

    if num_values == 0:
        result.update(
            {
                "num_values": 0,
                "close_rate": 1.0,
                "mae": 0.0,
                "rmse": 0.0,
                "max_abs_error": 0.0,
                "cosine_similarity": 1.0,
                "centered_cosine_similarity": 1.0,
                "similar_enough": True,
            }
        )
        return result

    diff = observed_flat - expected_flat
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs_error = float(abs_diff.max())
    close_rate = float(
        np.mean(
            np.isclose(
                expected_flat,
                observed_flat,
                atol=thresholds["atol"],
                rtol=thresholds["rtol"],
            )
        )
    )
    cosine_similarity = _safe_cosine_similarity(expected_flat, observed_flat)
    centered_cosine_similarity = _safe_cosine_similarity(
        expected_flat - expected_flat.mean(),
        observed_flat - observed_flat.mean(),
    )
    similar_enough = exact_match or (
        close_rate >= thresholds["min_close_rate"]
        and rmse <= thresholds["max_rmse"]
        and cosine_similarity >= thresholds["min_cosine_similarity"]
    )

    result.update(
        {
            "num_values": num_values,
            "close_rate": close_rate,
            "mae": mae,
            "rmse": rmse,
            "max_abs_error": max_abs_error,
            "cosine_similarity": cosine_similarity,
            "centered_cosine_similarity": centered_cosine_similarity,
            "similar_enough": bool(similar_enough),
        }
    )
    return result


def _compare_with_standard_dir(ds: StandardizedDataset, standard_dir: Path) -> dict[str, object]:
    expected = _expected_standardized_arrays(ds)
    exact_matches: dict[str, bool] = {}
    similarity_details: dict[str, dict[str, object]] = {}
    similarity_checks: dict[str, bool] = {}

    for filename, arr in expected.items():
        loaded = np.load(standard_dir / filename)["arr_0"]
        exact_matches[filename] = bool(np.array_equal(arr, loaded))
        thresholds = LABEL_SIMILARITY_THRESHOLDS if "labels" in filename else DATA_SIMILARITY_THRESHOLDS
        similarity = _compute_array_similarity(arr, loaded, thresholds)
        similarity_details[filename] = similarity
        similarity_checks[f"{filename}_similar_enough"] = bool(similarity["similar_enough"])

    return {
        "exact_matches": exact_matches,
        "similarity_details": similarity_details,
        "similarity_checks": similarity_checks,
    }


def _compute_raw_fingerprint(raw_dir: Path) -> dict[str, object]:
    y_train = np.load(raw_dir / "y_train.npy")
    y_test = np.load(raw_dir / "y_test.npy")
    y_all = np.vstack([y_train, y_test]).astype(np.int64)
    per_record_label_count, per_record_label_count_n = np.unique(
        y_all.sum(axis=1), return_counts=True
    )

    return {
        "raw_shapes": {
            "x_train": list(np.load(raw_dir / "x_train.npy").shape),
            "y_train": list(y_train.shape),
            "x_test": list(np.load(raw_dir / "x_test.npy").shape),
            "y_test": list(y_test.shape),
        },
        "records_total": int(y_all.shape[0]),
        "num_classes": int(y_all.shape[1]),
        "multi_label_column_sums": y_all.sum(axis=0).astype(int).tolist(),
        "labels_per_record_histogram": {
            str(int(k)): int(v)
            for k, v in zip(per_record_label_count.tolist(), per_record_label_count_n.tolist())
        },
    }


def _validate_cpsc_fingerprint(fp: dict[str, object]) -> dict[str, bool]:
    raw_shapes = fp["raw_shapes"]
    assert isinstance(raw_shapes, dict)
    checks = {
        "split_matches_4813_2064": (
            raw_shapes["y_train"][0] == EXPECTED_CPSC2018_SPLIT["train_raw"]
            and raw_shapes["y_test"][0] == EXPECTED_CPSC2018_SPLIT["test_raw"]
        ),
        "total_records_6877": fp["records_total"] == EXPECTED_CPSC2018_TOTAL_RECORDS,
        "num_classes_9": fp["num_classes"] == EXPECTED_CPSC2018_NUM_CLASSES,
        "multi_counts_match_cpsc_table_v": np.array_equal(
            np.array(fp["multi_label_column_sums"], dtype=np.int64),
            EXPECTED_CPSC2018_MULTI_COUNTS.astype(np.int64),
        ),
    }
    return checks


def _compute_cpsc_similarity(fp: dict[str, object]) -> dict[str, float]:
    raw_shapes = fp["raw_shapes"]
    assert isinstance(raw_shapes, dict)
    observed_counts = np.array(fp["multi_label_column_sums"], dtype=np.float64)
    expected_counts = EXPECTED_CPSC2018_MULTI_COUNTS.astype(np.float64)
    counts_diff = observed_counts - expected_counts

    class_count_mae = float(np.mean(np.abs(counts_diff)))
    class_count_relative_mae = float(class_count_mae / max(1.0, float(np.mean(expected_counts))))
    total_records = int(fp["records_total"])
    num_classes = int(fp["num_classes"])
    train_count = int(raw_shapes["y_train"][0])
    test_count = int(raw_shapes["y_test"][0])

    return {
        "class_count_cosine_similarity": _safe_cosine_similarity(observed_counts, expected_counts),
        "class_count_mae": class_count_mae,
        "class_count_relative_mae": class_count_relative_mae,
        "total_records_relative_error": float(
            abs(total_records - EXPECTED_CPSC2018_TOTAL_RECORDS) / EXPECTED_CPSC2018_TOTAL_RECORDS
        ),
        "train_split_relative_error": float(
            abs(train_count - EXPECTED_CPSC2018_SPLIT["train_raw"]) / EXPECTED_CPSC2018_SPLIT["train_raw"]
        ),
        "test_split_relative_error": float(
            abs(test_count - EXPECTED_CPSC2018_SPLIT["test_raw"]) / EXPECTED_CPSC2018_SPLIT["test_raw"]
        ),
        "num_classes_abs_error": float(abs(num_classes - EXPECTED_CPSC2018_NUM_CLASSES)),
    }


def _validate_cpsc_similarity(similarity: dict[str, float]) -> dict[str, bool]:
    return {
        "class_count_cosine_similarity_ge_threshold": (
            similarity["class_count_cosine_similarity"]
            >= CPSC_SIMILARITY_THRESHOLDS["min_class_count_cosine_similarity"]
        ),
        "class_count_relative_mae_le_threshold": (
            similarity["class_count_relative_mae"]
            <= CPSC_SIMILARITY_THRESHOLDS["max_class_count_relative_mae"]
        ),
        "total_records_relative_error_le_threshold": (
            similarity["total_records_relative_error"]
            <= CPSC_SIMILARITY_THRESHOLDS["max_total_records_relative_error"]
        ),
        "train_split_relative_error_le_threshold": (
            similarity["train_split_relative_error"]
            <= CPSC_SIMILARITY_THRESHOLDS["max_split_relative_error"]
        ),
        "test_split_relative_error_le_threshold": (
            similarity["test_split_relative_error"]
            <= CPSC_SIMILARITY_THRESHOLDS["max_split_relative_error"]
        ),
        "num_classes_abs_error_le_threshold": (
            similarity["num_classes_abs_error"]
            <= CPSC_SIMILARITY_THRESHOLDS["max_num_classes_abs_error"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/ecg_classification/ecg_process"),
        help="Directory containing raw x/y train/test .npy files.",
    )
    parser.add_argument(
        "--standard-dir",
        type=Path,
        default=Path("data/ecg_classification/ecg"),
        help="Directory containing standardized .npz files to compare against.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory to write regenerated standardized .npz files.",
    )
    args = parser.parse_args()

    standardized = _build_standardized_from_raw(args.raw_dir)
    fp = _compute_raw_fingerprint(args.raw_dir)
    cpsc_checks = _validate_cpsc_fingerprint(fp)
    cpsc_similarity = _compute_cpsc_similarity(fp)
    cpsc_similarity_checks = _validate_cpsc_similarity(cpsc_similarity)
    compare = _compare_with_standard_dir(standardized, args.standard_dir)
    compare_exact_checks = compare["exact_matches"]
    assert isinstance(compare_exact_checks, dict)
    compare_similarity_checks = compare["similarity_checks"]
    assert isinstance(compare_similarity_checks, dict)

    strict_checks_passed = all(cpsc_checks.values()) and all(compare_exact_checks.values())
    similarity_checks_passed = all(cpsc_similarity_checks.values()) and all(
        compare_similarity_checks.values()
    )
    all_ok = strict_checks_passed and similarity_checks_passed

    if args.out_dir is not None:
        _write_standardized(standardized, args.out_dir)
    else:
        with TemporaryDirectory() as tmp:
            _write_standardized(standardized, Path(tmp))

    result = {
        "source_candidate": "CPSC2018",
        "fingerprint": fp,
        "cpsc_checks": cpsc_checks,
        "cpsc_similarity_scores": cpsc_similarity,
        "cpsc_similarity_checks": cpsc_similarity_checks,
        "rebuild_matches_existing_standardized_files": compare_exact_checks,
        "rebuild_similarity_details": compare["similarity_details"],
        "rebuild_similarity_checks": compare_similarity_checks,
        "strict_checks_passed": strict_checks_passed,
        "similarity_checks_passed": similarity_checks_passed,
        "all_checks_passed": all_ok,
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
