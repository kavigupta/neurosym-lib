# ECG NEAR Experiment Reproduction

## Classification Tasks

We evaluate two classification formulations on 12-lead ECG data:

- **Single-label** (`--label-mode single`): Each recording is assigned one
  diagnostic class (argmax of multi-hot labels). Trained with cross-entropy loss
  and softmax output. This is a simplification -- ~6.9% of CPSC2018 records have
  multiple diagnoses -- but serves as a standard multi-class baseline.

- **Multi-label** (`--label-mode multi`): Each recording can carry multiple
  simultaneous diagnoses (multi-hot encoding). Trained with BCE loss and sigmoid
  output. This matches how CPSC2018 and the CinC 2020/2021 challenges were
  originally framed, and is considered more clinically relevant since cardiac
  conditions commonly co-occur.

Both formulations are evaluated with **macro F1** (average of per-class F1 scores),
consistent with the CPSC2018 challenge scoring.

See [ECG_RELATED_WORK.md](ECG_RELATED_WORK.md) for detailed literature context on
these task formulations.

## Dataset Provenance

The ECG dataset originates from **CPSC2018** (China Physiological Signal Challenge 2018):
- **6,877 records** across **9 diagnostic classes**
- Classes: SNR (Normal Sinus Rhythm), AF (Atrial Fibrillation), I-AVB (First-degree AV Block), LBBB (Left Bundle Branch Block), RBBB (Right Bundle Branch Block), PAC (Premature Atrial Contraction), PVC (Premature Ventricular Contraction), STD (ST Depression), STE (ST Elevation)
- Train/test split: 4,813/2,064 records
- Multi-label class totals: [918, 1221, 722, 236, 1857, 616, 700, 869, 220]

Reference: https://torch-ecg.readthedocs.io/en/latest/api/generated/torch_ecg.databases.CPSC2018.html

## 1. Dataset Preparation

To generate standardized train/val/test splits (plus single/multi label sets), run:
```bash
uv run jupyter nbconvert --execute --to notebook notebooks/get_ecg_datasets.ipynb
```
This exports:
- `data/ecg_classification/ecg/train_ecg_data.npz`
- `data/ecg_classification/ecg/valid_ecg_data.npz`
- `data/ecg_classification/ecg/test_ecg_data.npz`
- `data/ecg_classification/ecg/*_ecg_labels_single.npz`
- `data/ecg_classification/ecg/*_ecg_labels_multi.npz`

## 2. Running All Experiments (Recommended)

Run the full suite of experiments across local ECG data and optional
torch-ecg datasets (CPSC2018, CINC2021):

```bash
bash notebooks/ecg_reproduction/run_all_ecg_experiments.sh \
  --device cuda:0 \
  --extra-datasets CPSC2018,CINC2021 \
  --db-dir data/ecg_classification/ecg_db/
```

This runs, for each dataset and each label mode (single/multi):
1. **NEAR attention ECG DSL** (`benchmark_attention_ecg.py`)
2. **NEAR attention drop-eg DSL** (`benchmark_attention_drop_eg.py`)
3. **MLP baseline** (`baseline_nn.py`)
4. **Decision tree baseline** (`baseline_tree.py --model decision_tree`)
5. **Random forest baseline** (`baseline_tree.py --model random_forest`)
6. **torch-ecg CNN baselines** (`baseline_torch_ecg.py`)

For extra torch-ecg datasets (CPSC2018, CINC2021), the script first prepares
standardized `.npz` splits via `torch_ecg_data_example()`, then runs the
same experiment suite on those splits.

### Hyperparameters (defaults used by `run_all_ecg_experiments.sh`)

| Parameter | NEAR | MLP | Tree | TorchECG |
|-----------|------|-----|------|----------|
| num_programs | 200 | — | — | — |
| hidden_dim | 32 | 64 | — | — |
| neural_hidden_size | 32 | — | — | — |
| batch_size | 1024 | 256 | — | 256 |
| epochs (search) | 30 | 20 | — | 20 |
| epochs (final) | 60 | — | — | — |
| lr | 1e-4 | 1e-3 | — | 1e-3 |
| structural_cost | 0.1 | — | — | — |
| n_estimators | — | — | 200 | — |
| max_depth | — | — | 15 | — |
| train_seed | 0 | 0 | 0 | 0 |
| split_seed | 42 | — | — | 42 |
| max_records | 5000 | — | — | — |

### Output Structure

```
outputs/ecg_results/all_runs/
├── local_ecg/
│   ├── single/
│   │   ├── near_attention.pkl
│   │   ├── near_attention_summary.json
│   │   ├── near_attention_drop_eg.pkl
│   │   ├── near_attention_drop_eg_summary.json
│   │   ├── baseline_nn.json
│   │   ├── baseline_tree_decision_tree.json
│   │   ├── baseline_tree_random_forest.json
│   │   └── baseline_torch_ecg.json
│   └── multi/
│       └── (same files)
├── cpsc2018/
│   ├── single/
│   └── multi/
├── cinc2021/
│   ├── single/
│   └── multi/
└── prepared_data/
    ├── cpsc2018/
    │   ├── *.npz
    │   └── source_metadata.json
    └── cinc2021/
        └── ...
```

## 2b. Running Individual Experiments

Run individual benchmarks directly:

```bash
# NEAR attention DSL
uv run python notebooks/ecg_reproduction/benchmark_attention_ecg.py \
  --num-programs 200 --epochs 30 --final-epochs 60 --device cuda:0

# NEAR attention drop-eg DSL
uv run python notebooks/ecg_reproduction/benchmark_attention_drop_eg.py \
  --num-programs 200 --epochs 30 --final-epochs 60 --device cuda:0

# MLP baseline
uv run python notebooks/ecg_reproduction/baseline_nn.py --epochs 20 --device cuda:0

# Tree baselines
uv run python notebooks/ecg_reproduction/baseline_tree.py --model decision_tree
uv run python notebooks/ecg_reproduction/baseline_tree.py --model random_forest

# torch-ecg CNN baselines
uv run python notebooks/ecg_reproduction/baseline_torch_ecg.py --device cuda:0
```

All scripts accept `--data-dir` to point at a different standardized split
directory, and `--label-mode single|multi`.

### Arguments (common across scripts)

- `--output` — path for output file
- `--data-dir` (default: `data/ecg_classification/ecg`)
- `--label-mode` (`single` or `multi`, default: `single`)
- `--device` (default: `cuda:0`)

> Note: `label-mode=multi` uses BCE loss and regression-style validation
> metrics for search heuristics. Final evaluation still reports multilabel
> metrics.

## 3. Expected Results (Macro F1)

### Single-label

| Dataset | NEAR-Att | NEAR-Drop | MLP | DTree | RForest | TorchECG |
|---------|----------|-----------|-----|-------|---------|----------|
| local_ecg | **0.4025** | 0.2437 | 0.3191 | 0.1907 | 0.3218 | 0.1915 |
| cpsc2018 | 0.3836 | 0.2925 | 0.3214 | 0.1774 | 0.3627 | **0.3642** |
| cinc2021 | 0.2880 | 0.0890 | 0.3593 | 0.1940 | **0.4420** | 0.3333 |

### Multi-label

| Dataset | NEAR-Att | NEAR-Drop | MLP | DTree | RForest | TorchECG |
|---------|----------|-----------|-----|-------|---------|----------|
| local_ecg | **0.2653** | 0.1854 | 0.0000 | 0.2005 | 0.2297 | 0.1661 |
| cpsc2018 | 0.2202 | 0.2455 | 0.0000 | 0.1933 | **0.3360** | 0.3118 |
| cinc2021 | 0.1437 | 0.1250 | 0.0000 | 0.1432 | 0.2096 | **0.2661** |

> These results were produced on a single NVIDIA GPU with `--device cuda:0`.
> Minor floating-point variations are expected across hardware.

## 4. Analyzing Results

Open `analyze_ecg_results.ipynb` (single-label) or
`analyze_ecg_results_multi.ipynb` (multi-label) to inspect and compare
results. The notebooks load from `outputs/ecg_results/all_runs/` and write:
- `outputs/ecg_results/comparison_single.csv` / `comparison_single.md`
- `outputs/ecg_results/comparison_multi.csv` / `comparison_multi.md`
