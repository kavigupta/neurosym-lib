# ECG NEAR Experiment Reproduction

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

## 2. Running the Experiment

Run the benchmark script (uses the channel/interval ECG DSL):
```bash
uv run python notebooks/ecg_reproduction/benchmark_ecg.py --num-programs 20 --epochs 30 --device cpu
```

### Arguments
- `--output` (default: `outputs/ecg_results/reproduction.pkl`)
- `--num-programs` (default: `20`)
- `--hidden-dim` (default: `16`)
- `--neural-hidden-size` (default: `16`)
- `--batch-size` (default: `1024`)
- `--epochs` (default: `30`)
- `--final-epochs` (default: `40`)
- `--lr` (default: `1e-4`)
- `--structural-cost-penalty` (default: `0.1`)
- `--device` (default: `cuda:0`)
- `--label-mode` (`single` or `multi`, default: `single`)

> Note: `label-mode=multi` uses a BCE loss and a regression-style validation metric
> for search heuristics. Final evaluation still reports multilabel metrics.

## 2b. Baseline Models

These baselines use the same standardized ECG splits and label modes as the NEAR run.

MLP baseline (single-label by default):
```bash
uv run python notebooks/ecg_reproduction/baseline_nn.py --epochs 100 --device cpu
```

Tree baseline (decision tree or random forest):
```bash
uv run python notebooks/ecg_reproduction/baseline_tree.py --model decision_tree --label-mode single
uv run python notebooks/ecg_reproduction/baseline_tree.py --model random_forest --label-mode multi
```

Each baseline writes metrics to `outputs/ecg_results/baseline_*.json`.

## 3. Analyzing Results

Open `analyze_ecg_results.ipynb` to inspect and compare discovered programs. It
loads `outputs/ecg_results/reproduction.pkl` and writes:
- `outputs/ecg_results/comparison.csv`
- `outputs/ecg_results/comparison.md`
