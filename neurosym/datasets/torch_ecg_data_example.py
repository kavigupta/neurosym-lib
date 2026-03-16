from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .load_data import DatasetFromNpy, DatasetWrapper


DEFAULT_NUM_CHANNELS = 12
DEFAULT_FEATURE_DIM = 6


@dataclass
class _PreparedTorchECGDataset:
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y_multi: np.ndarray
    val_y_multi: np.ndarray
    test_y_multi: np.ndarray
    class_names: list[str]


def _import_torch_ecg_databases():
    try:
        from torch_ecg import databases
    except ImportError as exc:
        raise ImportError(
            "torch-ecg is required for `torch_ecg_data_example`. "
            "Install it with `pip install torch-ecg`."
        ) from exc
    return databases


def _flatten_record_container(value: Any) -> list[str]:
    if isinstance(value, dict):
        output: list[str] = []
        for nested in value.values():
            output.extend(_flatten_record_container(nested))
        return output
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _extract_records_with_optional_split(all_records: Any) -> tuple[list[str], list[str], list[str]]:
    if isinstance(all_records, dict):
        lowered = {str(k).lower(): k for k in all_records}
        train_keys = [lowered[k] for k in lowered if "train" in k]
        test_keys = [lowered[k] for k in lowered if "test" in k]
        val_keys = [lowered[k] for k in lowered if ("val" in k or "valid" in k)]

        if train_keys and test_keys:
            train_records: list[str] = []
            test_records: list[str] = []
            val_records: list[str] = []
            for key in train_keys:
                train_records.extend(_flatten_record_container(all_records[key]))
            for key in test_keys:
                test_records.extend(_flatten_record_container(all_records[key]))
            for key in val_keys:
                val_records.extend(_flatten_record_container(all_records[key]))
            return train_records, val_records, test_records

        flattened = _flatten_record_container(all_records)
        return flattened, [], []

    flattened = _flatten_record_container(all_records)
    return flattened, [], []


def _compute_random_split_indices(
    n_records: int,
    split_seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_records < 3:
        raise ValueError(
            f"Need at least 3 records for train/val/test split, got {n_records}."
        )
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1, got {val_fraction + test_fraction}"
        )

    rs = np.random.RandomState(split_seed)
    ordering = rs.permutation(n_records)

    n_test = int(n_records * test_fraction)
    if n_test == 0:
        n_test = 1

    n_remaining = n_records - n_test
    n_val = int(n_remaining * val_fraction)
    if n_val == 0 and n_remaining > 1:
        n_val = 1

    n_train = n_records - n_test - n_val
    if n_train <= 0:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        n_train = n_records - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes for n_records={n_records}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    test_idx = ordering[:n_test]
    val_idx = ordering[n_test : n_test + n_val]
    train_idx = ordering[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _normalize_minmax_per_split(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=True)
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    den = maxs - mins
    den[den == 0] = 1.0
    return ((x - mins) / den).astype(np.float32)


def _ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        x = np.squeeze(x)
        if x.ndim == 1:
            return x[None, :]
        if x.ndim != 2:
            raise ValueError(f"Expected 1D/2D ECG signal, got shape {signal.shape}")
    # Heuristic: if axis 0 looks like time and axis 1 looks like channels, transpose.
    if x.shape[1] <= DEFAULT_NUM_CHANNELS and x.shape[0] > x.shape[1]:
        x = x.T
    return x


def _dominant_autocorr_lag_norm(lead_signal: np.ndarray, max_lag: int = 200) -> float:
    x = np.asarray(lead_signal, dtype=np.float64)
    if x.size < 4:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    lag_bound = min(max_lag, x.size - 1)
    if lag_bound <= 1:
        return 0.0
    # Use FFT-based autocorrelation — O(n log n) instead of O(n^2).
    n = x.size
    fft_x = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))
    acf = acf_full[:lag_bound + 1]
    if acf.size <= 1:
        return 0.0
    lag = int(np.argmax(acf[1:]) + 1)
    return float(lag) / float(max(1, x.size - 1))


def _spectral_entropy_norm(lead_signal: np.ndarray) -> float:
    x = np.asarray(lead_signal, dtype=np.float64)
    if x.size < 4:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    spec = np.abs(np.fft.rfft(x))
    power = np.sum(spec)
    if power <= 0:
        return 0.0
    p = spec / power
    ent = -np.sum(p * np.log(p + 1e-12))
    max_ent = np.log(max(2, p.size))
    return float(ent / max_ent)


def _extract_single_lead_features(lead_signal: np.ndarray) -> np.ndarray:
    x = np.asarray(lead_signal, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros((DEFAULT_FEATURE_DIM, 2), dtype=np.float32)

    dx = np.diff(x)
    zero_cross_rate = (
        float(np.mean((x[:-1] * x[1:]) < 0.0)) if x.size > 1 else 0.0
    )
    abs_diff_mean = float(np.mean(np.abs(dx))) if dx.size > 0 else 0.0
    diff_std = float(np.std(dx)) if dx.size > 0 else 0.0
    dom_lag = _dominant_autocorr_lag_norm(x)
    spec_ent = _spectral_entropy_norm(x)
    mean_abs = float(np.mean(np.abs(x)))

    interval_features = np.array(
        [zero_cross_rate, abs_diff_mean, diff_std, dom_lag, spec_ent, mean_abs],
        dtype=np.float32,
    )
    amplitude_features = np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
        ],
        dtype=np.float32,
    )
    return np.stack([interval_features, amplitude_features], axis=-1)


def _signal_to_flat_features(
    signal: np.ndarray,
    num_channels: int = DEFAULT_NUM_CHANNELS,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> np.ndarray:
    if feature_dim != DEFAULT_FEATURE_DIM:
        raise ValueError(
            f"Only feature_dim={DEFAULT_FEATURE_DIM} is supported, got {feature_dim}"
        )

    x = _ensure_channel_first(signal)
    n_channels, _ = x.shape
    if n_channels >= num_channels:
        x = x[:num_channels]
    else:
        padded = np.zeros((num_channels, x.shape[1]), dtype=np.float32)
        padded[:n_channels] = x
        x = padded

    per_channel = np.stack([_extract_single_lead_features(ch) for ch in x], axis=0)
    return per_channel.reshape(-1).astype(np.float32)


def _parse_labels(raw_labels: Any) -> list[str]:
    if raw_labels is None:
        return []

    if isinstance(raw_labels, dict):
        for key in ("labels", "diagnoses", "diagnosis", "classes", "class"):
            if key in raw_labels:
                return _parse_labels(raw_labels[key])
        return [str(x) for x in raw_labels.values()]

    if isinstance(raw_labels, str):
        cleaned = raw_labels.strip()
        return [cleaned] if cleaned else []

    if isinstance(raw_labels, (list, tuple, set, np.ndarray)):
        out = [str(x).strip() for x in raw_labels]
        return [x for x in out if x]

    return [str(raw_labels)]


def _call_with_supported_kwargs(fn, kwargs: dict[str, Any]):
    try:
        sig = inspect.signature(fn)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**supported)
    except (TypeError, ValueError):
        return fn()


def _load_record_data(db, record: str, *, _format_cache: dict | None = None) -> np.ndarray:
    """Load ECG waveform data for a single record.

    Uses a format cache to avoid repeatedly failing through torch_ecg's
    ``load_data`` when the .mat files use PhysioNet format (``val`` key)
    instead of the original CPSC2018 format (``ECG`` key).  On the first
    call the loader probes the format; subsequent calls skip directly to
    the working path.
    """
    if _format_cache is None:
        _format_cache = {}

    cached_fmt = _format_cache.get("fmt")

    # Fast path: format already detected as direct-read.
    if cached_fmt in ("val", "ECG"):
        return _load_mat_direct(db, record, cached_fmt)

    # Try torch_ecg's load_data first (only if we haven't already failed).
    if cached_fmt != "direct_fallback":
        load_data = getattr(db, "load_data", None)
        if load_data is not None:
            kwargs = {
                "rec": record,
                "record": record,
                "data_format": "channel_first",
                "units": "mV",
                "return_fs": False,
            }
            try:
                sig = inspect.signature(load_data)
                supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
                data = load_data(**supported)
                if isinstance(data, tuple):
                    data = data[0]
                _format_cache["fmt"] = "torch_ecg"
                return np.asarray(data, dtype=np.float32)
            except (TypeError, ValueError, KeyError):
                pass

    # Direct .mat fallback — detect format on first record, cache it.
    fmt = _detect_and_load_mat(db, record, _format_cache)
    return _load_mat_direct(db, record, fmt)


def _detect_and_load_mat(db, record: str, _format_cache: dict) -> str:
    """Probe a single .mat file to determine its format and cache the result."""
    from scipy.io import loadmat

    rec_path = db.get_absolute_path(record, db.rec_ext)
    raw = loadmat(str(rec_path))
    if "val" in raw:
        _format_cache["fmt"] = "val"
        return "val"
    if "ECG" in raw:
        _format_cache["fmt"] = "ECG"
        return "ECG"
    raise ValueError(
        f"Unrecognized .mat format for record {record}: keys={list(raw.keys())}"
    )


def _load_mat_direct(db, record: str, fmt: str) -> np.ndarray:
    """Read a .mat file directly, using the pre-detected format."""
    from scipy.io import loadmat

    rec_path = db.get_absolute_path(record, db.rec_ext)
    raw = loadmat(str(rec_path))
    if fmt == "val":
        # PhysioNet WFDB format: val is (n_leads, n_samples) in ADC units.
        # Standard gain for CinC 2020 CPSC2018 data is 1000 ADC/mV.
        data = np.asarray(raw["val"], dtype=np.float32) / 1000.0
    elif fmt == "ECG":
        data = np.asarray(raw["ECG"]["data"][0, 0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown .mat format: {fmt}")
    return data


def _load_record_labels(db, record: str) -> list[str]:
    get_labels = getattr(db, "get_labels", None)
    if get_labels is not None:
        kwargs = {
            "rec": record,
            "record": record,
            "ann_format": "a",
            "fmt": "a",
            "scored_only": True,
            "normalize": True,
        }
        try:
            sig = inspect.signature(get_labels)
            supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
            raw = get_labels(**supported)
        except (TypeError, ValueError):
            try:
                raw = get_labels(record)
            except TypeError:
                raw = _call_with_supported_kwargs(get_labels, kwargs)
        labels = _parse_labels(raw)
        if labels:
            return labels

    load_ann = getattr(db, "load_ann", None)
    if load_ann is not None:
        try:
            raw_ann = load_ann(record)
        except TypeError:
            raw_ann = load_ann(rec=record)
        labels = _parse_labels(raw_ann)
        if labels:
            return labels

    return []


# PhysioNet bulk ZIP for the CinC 2020 challenge (open access, ~4.5 GB).
_PHYSIONET_CINC2020_ZIP = (
    "https://physionet.org/static/published-projects/challenge-2020/"
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-"
    "challenge-2020-1.0.2.zip"
)

# Fallback download specifications for datasets whose upstream URLs are dead.
# Each entry maps a torch-ecg class name to a dict with:
#   "zip_prefix": path prefix inside the CinC 2020 ZIP to extract
#   "label_urls": list of (url, extract) pairs for annotation files
_FALLBACK_SPECS: dict[str, dict] = {
    "CPSC2018": {
        "zip_prefix": "training/cpsc_2018/",
        "label_urls": [("http://2018.icbeb.org/file/REFERENCE.csv", False)],
    },
}


def _download_from_physionet(
    spec: dict,
    db_path: Path,
    verbose: int,
    http_get,
) -> None:
    """Download record files from PhysioNet (open-access).

    The CinC 2020 challenge hosts CPSC2018 data as individual ``.mat`` /
    ``.hea`` files under subdirectories ``g1/`` … ``g7/``.  We crawl the
    directory listings to build a file list, then download in parallel
    using a thread pool with ``requests.Session`` for HTTP keep-alive.
    Both ``.mat`` and ``.hea`` files are fetched (the ``.hea`` headers
    are needed by CINC2021's ``load_ann`` for annotation reading).
    """
    import re
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    from requests.adapters import HTTPAdapter

    zip_prefix = spec["zip_prefix"]  # e.g. "training/cpsc_2018/"
    # Derive the per-file base URL from the zip prefix.
    base_url = (
        "https://physionet.org/files/challenge-2020/1.0.2/" + zip_prefix
    )

    if verbose:
        print(
            f"[torch_ecg_data_example] Downloading from PhysioNet: {base_url}"
        )

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64)
    session.mount("https://", adapter)

    # Discover group subdirectories (g1/, g2/, …).
    listing = session.get(base_url, timeout=30).text
    group_dirs = sorted(re.findall(r'href="(g\d+/)"', listing))

    # Collect (url, dest_path) pairs — .mat and .hea files.
    items: list[tuple[str, Path]] = []
    for gdir in group_dirs:
        group_url = base_url + gdir
        group_listing = session.get(group_url, timeout=30).text
        files = re.findall(r'href="([^"]+\.(?:mat|hea))"', group_listing)
        dest = db_path / gdir.rstrip("/")
        dest.mkdir(parents=True, exist_ok=True)
        for fname in files:
            out_path = dest / fname
            if not out_path.exists():
                items.append((group_url + fname, out_path))

    if verbose:
        print(
            f"[torch_ecg_data_example]   {len(items)} files to download "
            f"across {len(group_dirs)} groups (~1.5 GB)..."
        )

    lock = threading.Lock()
    done = [0]

    def _fetch(item: tuple[str, Path]) -> None:
        url, out = item
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        with lock:
            done[0] += 1
            if verbose and done[0] % 500 == 0:
                print(
                    f"[torch_ecg_data_example]   "
                    f"{done[0]}/{len(items)} files..."
                )

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(_fetch, item) for item in items]
        for f in as_completed(futures):
            f.result()

    session.close()

    # Download label/annotation files.
    for label_url, extract in spec.get("label_urls", []):
        if verbose:
            print(f"[torch_ecg_data_example]   labels: {label_url}")
        http_get(label_url, str(db_path), extract=extract)

    if verbose:
        n_mat = sum(1 for _ in db_path.rglob("*.mat"))
        print(
            f"[torch_ecg_data_example] PhysioNet download complete. "
            f"{n_mat} record file(s)."
        )


def _auto_download_torch_ecg(
    db_cls: type,
    db_path: Path,
    dataset_name: str,
    verbose: int,
) -> None:
    """Download dataset files using torch_ecg's ``http_get`` utility.

    torch_ecg <= 0.0.31 crashes during ``__init__`` on pandas >= 2.0 when the
    data directory is empty (the ``_ls_rec`` method uses a ``.at`` pattern
    incompatible with modern pandas).  We work around this by fetching the
    files *before* the database object is constructed so ``_ls_rec`` finds
    real records and never enters the broken code-path.

    The function first tries the URLs declared by the torch-ecg database class.
    If those fail (e.g. Alibaba Cloud mirrors are offline), it falls back to
    known Google Cloud Storage / PhysioNet mirrors listed in
    ``_FALLBACK_MIRRORS``.
    """
    try:
        from torch_ecg.utils.download import http_get
    except ImportError as exc:
        raise ImportError(
            "torch-ecg is required for automatic dataset download. "
            "Install it with `pip install torch-ecg`."
        ) from exc

    # Resolve download URLs from the class.  ``url`` is typically a @property
    # that returns static URLs without referencing instance state, so we can
    # read it from a bare (un-initialised) instance to avoid __init__.
    urls: list[str] = []
    try:
        bare = object.__new__(db_cls)
        url_val = bare.url
        if isinstance(url_val, str):
            urls = [url_val]
        elif isinstance(url_val, (list, tuple)):
            urls = list(url_val)
    except Exception:  # noqa: BLE001
        pass

    # Also collect label URLs from ``_download_labels`` source.
    label_urls: list[str] = []
    if hasattr(db_cls, "_download_labels"):
        try:
            import re

            src = inspect.getsource(db_cls._download_labels)
            label_urls = re.findall(r'"(https?://[^"]+)"', src)
        except Exception:  # noqa: BLE001
            pass

    # Build (url, extract) download plan from the primary sources.
    download_plan: list[tuple[str, bool]] = [(u, True) for u in urls] + [
        (u, False) for u in label_urls
    ]

    if not download_plan and dataset_name not in _FALLBACK_SPECS:
        raise FileNotFoundError(
            f"torch-ecg data directory is empty: {db_path}. "
            f"Could not automatically resolve download URLs for "
            f"`{dataset_name}`. Please download the dataset files manually."
        )

    if verbose:
        print(
            f"[torch_ecg_data_example] Downloading {dataset_name} "
            f"into {db_path} ..."
        )

    # Attempt primary URLs; fall back to mirrors on failure.
    success = False
    if download_plan:
        try:
            for url, extract in download_plan:
                if verbose:
                    print(f"[torch_ecg_data_example]   {url}")
                http_get(url, str(db_path), extract=extract)
            success = True
        except Exception as primary_exc:  # noqa: BLE001
            if verbose:
                print(
                    f"[torch_ecg_data_example] Primary URLs failed "
                    f"({primary_exc}), trying fallback mirrors..."
                )

    if not success:
        fallback = _FALLBACK_SPECS.get(dataset_name)
        if not fallback:
            raise FileNotFoundError(
                f"torch-ecg data directory is empty: {db_path}. "
                f"Primary download URLs for `{dataset_name}` are unreachable "
                f"and no fallback mirrors are configured."
            )
        _download_from_physionet(fallback, db_path, verbose, http_get)

    if verbose:
        n_files = sum(1 for _ in db_path.iterdir())
        print(
            f"[torch_ecg_data_example] Download complete. "
            f"{n_files} file(s) in {db_path}."
        )


def _prepare_torch_ecg_dataset(
    dataset_name: str,
    db_dir: str | None,
    working_dir: str | None,
    max_records: int | None,
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
    normalize_per_split: bool,
    verbose: int,
    dataset_kwargs: dict[str, Any] | None,
) -> _PreparedTorchECGDataset:
    dataset_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    db_module = _import_torch_ecg_databases()
    db_cls = getattr(db_module, dataset_name, None)
    if db_cls is None:
        available = sorted(
            name
            for name in dir(db_module)
            if not name.startswith("_") and isinstance(getattr(db_module, name), type)
        )
        raise ValueError(
            f"Unknown torch-ecg dataset `{dataset_name}`. "
            f"Available classes include: {available}"
        )

    init_kwargs = dict(dataset_kwargs)
    if db_dir is not None:
        init_kwargs.setdefault("db_dir", db_dir)
    if working_dir is not None:
        init_kwargs.setdefault("working_dir", working_dir)
    if "verbose" in inspect.signature(db_cls).parameters:
        init_kwargs.setdefault("verbose", verbose)

    # Early check: verify the data directory exists and contains data files.
    # torch_ecg <= 0.0.31 has a pandas 2.x compatibility bug that crashes
    # during __init__ when the directory is empty or annotations are missing,
    # so we must download files *before* constructing the database object.
    #
    # We also detect partial downloads (data files present but no annotation
    # file) since those cause a "No labels found" error downstream.
    effective_db_dir = init_kwargs.get("db_dir")
    if effective_db_dir is not None:
        db_path = Path(effective_db_dir)
        if not db_path.exists():
            db_path.mkdir(parents=True, exist_ok=True)
        # Determine the expected file extension from the class.
        rec_ext = "mat"
        try:
            bare = object.__new__(db_cls)
            rec_ext = getattr(bare, "rec_ext", rec_ext)
        except Exception:  # noqa: BLE001
            pass
        has_data = any(db_path.rglob(f"*.{rec_ext}"))
        has_annotations = any(db_path.rglob("*.csv"))
        if db_path.is_dir() and (not has_data or not has_annotations):
            _auto_download_torch_ecg(db_cls, db_path, dataset_name, verbose)

    try:
        db = db_cls(**init_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to initialize torch-ecg dataset `{dataset_name}` with "
            f"db_dir={db_dir!r}. Ensure the dataset files and annotations are "
            "downloaded and accessible in this directory. "
            "Note: torch-ecg <= 0.0.31 has a known compatibility issue with "
            "pandas >= 2.0 when annotation files are missing."
        ) from exc

    all_records = getattr(db, "all_records", None)
    if all_records is None:
        raise AttributeError(
            f"{dataset_name} instance has no `all_records`; cannot build benchmark dataset."
        )

    train_records, val_records, test_records = _extract_records_with_optional_split(all_records)
    if train_records and test_records:
        if not val_records:
            rs = np.random.RandomState(split_seed)
            original_train = list(train_records)
            perm = rs.permutation(len(original_train))
            n_val = int(len(original_train) * val_fraction)
            if n_val == 0 and len(original_train) > 1:
                n_val = 1
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            train_records = [original_train[i] for i in train_idx]
            val_records = [original_train[i] for i in val_idx] if n_val > 0 else []
    else:
        records = train_records
        if max_records is not None:
            records = records[: max_records]
        train_idx, val_idx, test_idx = _compute_random_split_indices(
            len(records), split_seed=split_seed, val_fraction=val_fraction, test_fraction=test_fraction
        )
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]

    combined_records = train_records + val_records + test_records
    if (
        max_records is not None
        and max_records > 0
        and max_records < len(combined_records)
    ):
        # If max_records truncates the dataset, re-split deterministically.
        combined = combined_records[:max_records]
        train_idx, val_idx, test_idx = _compute_random_split_indices(
            len(combined),
            split_seed=split_seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        train_records = [combined[i] for i in train_idx]
        val_records = [combined[i] for i in val_idx]
        test_records = [combined[i] for i in test_idx]

    records_by_split = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    split_features: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[list[str]]] = {"train": [], "val": [], "test": []}

    # Shared format cache avoids re-probing .mat format on every record.
    _fmt_cache: dict = {}

    total_records = sum(len(v) for v in records_by_split.values())
    loaded = 0

    for split_name, split_records in records_by_split.items():
        for record in split_records:
            labels = _load_record_labels(db, record)
            if not labels:
                loaded += 1
                continue
            signal = _load_record_data(db, record, _format_cache=_fmt_cache)
            features = _signal_to_flat_features(signal)
            split_features[split_name].append(features)
            split_labels[split_name].append(labels)
            loaded += 1
            if verbose and loaded % 500 == 0:
                print(
                    f"[torch_ecg_data_example] Loaded {loaded}/{total_records} records..."
                )

    all_label_names: set[str] = set()
    for split_name in ("train", "val", "test"):
        for labels in split_labels[split_name]:
            all_label_names.update(labels)
    if not all_label_names:
        raise ValueError(
            f"No labels found for dataset `{dataset_name}`. "
            "This loader requires a classification ECG dataset."
        )
    class_names = sorted(all_label_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def build_arrays(split_name: str) -> tuple[np.ndarray, np.ndarray]:
        x_list = split_features[split_name]
        y_list = split_labels[split_name]
        if not x_list:
            raise ValueError(f"Split `{split_name}` is empty after filtering.")
        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.zeros((len(y_list), len(class_names)), dtype=np.float32)
        for i, labels in enumerate(y_list):
            for label in labels:
                y[i, class_to_idx[label]] = 1.0
        return x, y

    train_x, train_y = build_arrays("train")
    val_x, val_y = build_arrays("val")
    test_x, test_y = build_arrays("test")

    if normalize_per_split:
        train_x = _normalize_minmax_per_split(train_x)
        val_x = _normalize_minmax_per_split(val_x)
        test_x = _normalize_minmax_per_split(test_x)

    return _PreparedTorchECGDataset(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y_multi=train_y,
        val_y_multi=val_y,
        test_y_multi=test_y,
        class_names=class_names,
    )


def torch_ecg_data_example(
    train_seed: int,
    dataset_name: str,
    db_dir: str | None = None,
    working_dir: str | None = None,
    label_mode: str = "multi",
    is_regression: bool | None = None,
    max_records: int | None = 5000,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_seed: int = 42,
    normalize_per_split: bool = True,
    verbose: int = 1,
    dataset_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> DatasetWrapper:
    """
    Build a NEAR-compatible ECG datamodule from a torch-ecg dataset.

    This loader converts per-record raw ECG waveforms into the same flattened
    structure expected by the ECG DSLs: ``(N, 12 * 6 * 2) = (N, 144)``,
    where each lead contributes six "interval-like" and six "amplitude-like"
    engineered features.

    :param train_seed: seed used for train split ordering in DatasetFromNpy.
    :param dataset_name: torch-ecg database class name, e.g. ``CPSC2018``.
    :param db_dir: local path to the dataset files.
    :param working_dir: optional torch-ecg working/cache directory.
    :param label_mode: ``"single"`` (argmax labels) or ``"multi"`` (multi-hot).
    :param is_regression: overrides regression mode. Defaults to ``label_mode == "multi"``.
    :param max_records: maximum number of records to load after split extraction.
    :param val_fraction: validation fraction when a split is not provided by the dataset.
    :param test_fraction: test fraction when a split is not provided by the dataset.
    :param split_seed: seed for split creation.
    :param normalize_per_split: min-max normalize each split independently.
    :param verbose: verbosity forwarded to torch-ecg dataset when supported.
    :param dataset_kwargs: additional kwargs forwarded to the torch-ecg dataset constructor.
    :param kwargs: forwarded to ``DatasetWrapper`` (e.g. batch_size).
    :return: ``DatasetWrapper`` containing train/val/test ``DatasetFromNpy`` datasets.
    """
    if label_mode not in {"single", "multi"}:
        raise ValueError(f"Unknown label_mode `{label_mode}`, expected `single` or `multi`.")
    if is_regression is None:
        is_regression = label_mode == "multi"

    prepared = _prepare_torch_ecg_dataset(
        dataset_name=dataset_name,
        db_dir=db_dir,
        working_dir=working_dir,
        max_records=max_records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        normalize_per_split=normalize_per_split,
        verbose=verbose,
        dataset_kwargs=dataset_kwargs,
    )

    if label_mode == "single":
        train_y = prepared.train_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
        val_y = prepared.val_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
        test_y = prepared.test_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
    else:
        train_y = prepared.train_y_multi
        val_y = prepared.val_y_multi
        test_y = prepared.test_y_multi

    datamodule = DatasetWrapper(
        DatasetFromNpy(prepared.train_x, train_y, seed=train_seed, is_regression=is_regression),
        DatasetFromNpy(prepared.test_x, test_y, seed=0, is_regression=is_regression),
        val=DatasetFromNpy(prepared.val_x, val_y, seed=0, is_regression=is_regression),
        **kwargs,
    )
    datamodule.class_names = prepared.class_names
    datamodule.dataset_name = dataset_name
    return datamodule
