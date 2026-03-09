# ECG Dataset Provenance

## Conclusion
`data/ecg_classification/ecg/` originates from a preprocessed subset of **CPSC2018** (China Physiological Signal Challenge 2018), not PTB-XL or MIMIC-IV-ECG.

## Why this is the source
- Raw split sizes in `data/ecg_classification/ecg_process/` are:
  - `x_train.npy`: 4813 records
  - `x_test.npy`: 2064 records
  - total: `6877`, which matches CPSC2018.
- Label dimensionality is exactly `9` classes, matching CPSC2018.
- Multi-label marginal totals from `y_train.npy + y_test.npy` are:
  - `[918, 1221, 722, 236, 1857, 616, 700, 869, 220]`
  - This exactly matches published CPSC2018 multi-label totals (Table V; class order: SNR, AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE).
- The standardized files in `data/ecg_classification/ecg/*.npz` are reproducibly regenerated from `ecg_process` using the logic in `notebooks/get_ecg_datasets.ipynb` (per-split min-max normalization, `val_fraction=0.15`, `seed=42`).

## Reproduce and verify
Run:

```bash
python notebooks/ecg_reproduction/verify_ecg_provenance.py
```

The script checks:
- CPSC2018 fingerprint matches
- CPSC2018 similarity scores (class-count cosine, relative errors on totals/splits)
- regenerated standardized arrays are byte-identical to current `data/ecg_classification/ecg/*.npz`
- regenerated standardized arrays are also evaluated with explicit similarity metrics
  (`close_rate`, `MAE`, `RMSE`, `cosine_similarity`) and thresholded pass/fail

## Sources
- CPSC2018 dataset overview (6877 records, 9 classes):
  - https://torch-ecg.readthedocs.io/en/latest/api/generated/torch_ecg.databases.CPSC2018.html
- CPSC2018 multi-label class totals (Table V, supplementary):
  - https://openreview.net/attachment?id=vwzHeWFM4Q&name=supplementary_material
