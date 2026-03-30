# ECG NEAR Experiment Reproduction

This directory contains scripts to reproduce the NEAR experiments on the ECG dataset for ECG diagnostic classification using pre-extracted ECGDeli features.

## 1. Dataset

The experiments use the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/) datasets from PhysioNet, with pre-extracted ECGDeli features. The dataset contains:

- **21,799** 12-lead ECG records with **177 ECGDeli features** (14 per-lead types x 12 leads + 9 global)
- **5 diagnostic superclasses**: NORM, MI, STTC, CD, HYP
- **Standard split**: Folds 1-8 train, fold 9 validation, fold 10 test

### Downloading the data

The data is downloaded automatically on first run via `wget`. You can also download it manually:

```bash
# Download PTB-XL (~2 GB)
wget -r -N -c -np -nH --cut-dirs=1 -P data/ecg \
  https://physionet.org/files/ptb-xl/1.0.3/

# Download PTB-XL+ (~200 MB)
wget -r -N -c -np -nH --cut-dirs=1 -P data/ecg \
  https://physionet.org/files/ptb-xl-plus/1.0.1/
```

After downloading, the directory structure should be:

```
data/ecg/
  ptb-xl/1.0.3/       # ECG records + metadata
    ptbxl_database.csv
    scp_statements.csv
    records*/
  ptb-xl-plus/1.0.1/  # ECGDeli pre-extracted features
    features/
```

Data is cached in `data/ecg/` after the first download. No further preparation is needed.

## 2. Running the Baselines

### Random Forest

```bash
uv run python notebooks/ecg_reproduction/baseline_tree.py \
  --model random_forest --label-mode single
```

### Decision Tree

```bash
uv run python notebooks/ecg_reproduction/baseline_tree.py \
  --model decision_tree --label-mode single
```

#### Tree Baseline Arguments

* `--model` (str, default: `random_forest`, choices: `random_forest`, `decision_tree`): Which tree model to use.
* `--label-mode` (str, default: `single`, choices: `single`, `multi`): Label mode (`single` = filtered to records with exactly 1 superclass; `multi` = multi-hot).
* `--data-dir` (str, default: `data/ecg`): Path to ECG data directory.
* `--seed` (int, default: `42`): Random seed.
* `--n-estimators` (int, default: `100`): Number of estimators for Random Forest.
* `--max-depth` (int, default: `None`): Max depth for tree (None = unlimited, matching sklearn default).
* `--output` (str): Optional JSON path for metrics output.

### MLP Baseline

```bash
uv run python notebooks/ecg_reproduction/baseline_nn.py \
  --label-mode single --device cuda:0
```

#### MLP Baseline Arguments

* `--label-mode` (str, default: `single`, choices: `single`, `multi`): Label mode.
* `--data-dir` (str, default: `data/ecg`): Path to ECG data directory.
* `--batch-size` (int, default: `256`): Training batch size.
* `--epochs` (int, default: `300`): Maximum training epochs.
* `--lr` (float, default: `1e-3`): Learning rate.
* `--hidden-dim` (int, default: `256`): Hidden layer dimension (3 layers).
* `--seed` (int, default: `42`): Random seed.
* `--device` (str, default: `cuda:0`): Device for training.
* `--output` (str): Optional JSON path for metrics output.

## 3. Running the NEAR Experiment

To run the NEAR experiment with the attention ECG DSL:

```bash
uv run python notebooks/ecg_reproduction/benchmark_attention_ecg.py \
  --label-mode single --device cuda:0
```

Recommended hyperparameters (from experiment tuning):

```bash
uv run python notebooks/ecg_reproduction/benchmark_attention_ecg.py \
  --label-mode single --hidden-dim 32 --lr 1e-3 --batch-size 256 \
  --epochs 50 --final-epochs 200 --num-programs 15 \
  --structural-cost-penalty 0.1 --device cuda:0
```

### NEAR Arguments

* `--output` (str, default: `outputs/ecg_results/reproduction_attention.pkl`): Path to save results.
* `--data-dir` (str, default: `data/ecg`): Directory for ECG data.
* `--num-programs` (int, default: `200`): Number of programs to discover during search.
* `--hidden-dim` (int, default: `16`): Hidden dimension for the DSL.
* `--neural-hidden-size` (int, default: `32`): Hidden size for the neural hole filler.
* `--batch-size` (int, default: `1024`): Training batch size.
* `--epochs` (int, default: `30`): Number of epochs for training during the search phase.
* `--final-epochs` (int, default: `60`): Number of epochs for final training of discovered programs.
* `--lr` (float, default: `1e-4`): Learning rate.
* `--structural-cost-penalty` (float, default: `0.1`): Penalty multiplier for structural cost in search.
* `--device` (str, default: `cuda:0`): Device to use for training (e.g., `cuda:0` or `cpu`).
* `--label-mode` (str, default: `single`, choices: `single`, `multi`): Label mode.
* `--use-shields` (flag, default: off): Enable shield productions in the DSL.

### Attention ECG DSL

The DSL reshapes the flat 177-dim input into 21 channels (12 leads + 9 global), each with 14 features. Each channel is bound as a separate lambda variable (`$0` through `$20`).

**Productions:**

| Production | Type Signature | Description |
|---|---|---|
| `embed` | `$fInp -> $fHid` | Project raw features to hidden dim |
| `linear` | `$fHid -> $fHid` | Hidden-to-hidden linear layer |
| `linear_bool` | `$fHid -> {f, 1}` | Project to scalar (for `ite` condition) |
| `output` | `$fHid -> $fOut` | Project hidden to class logits |
| `add` | `(%num, %num) -> %num` | Element-wise addition |
| `mul` | `(%num, %num) -> %num` | Element-wise multiplication |
| `sub` | `(%num, %num) -> %num` | Element-wise subtraction |
| `ite` | `({f, 1}, %num, %num) -> %num` | Differentiable if-then-else (sigmoid-gated) |
| `lam` | binds 21 channel variables | Lambda over all input channels |
| `shield_k` | removes variable `k` from scope | Only with `--use-shields` |

**Sample programs** (shown with 3 channels / 4 features / 2 classes for clarity):

```
P1 - Simple pipeline (embed one channel, classify):
  (lam (output (embed $0_0)))

P2 - Add two channels before embedding:
  (lam (output (embed (add $0_0 $1_0))))

P3 - If-then-else (condition on channel 0, branch on channels 1 vs 2):
  (lam (output (ite (linear_bool (embed $0_0))
                    (embed $1_0)
                    (embed $2_0))))

P4 - Complex nested (add + mul + linear):
  (lam (output (linear (add (embed $0_0)
                            (mul (embed $1_0)
                                 (embed $2_0))))))

P5 - If-then-else with subtraction in else branch:
  (lam (output (ite (linear_bool (embed $0_0))
                    (linear (embed $1_0))
                    (linear (sub (embed $1_0)
                                 (embed $2_0))))))

P6 - Shield (drop channel 2 from scope, classify from channel 0):
  (lam (shield2 (output (embed $0_0))))

P7 - Shield + ite (drop channel 0, branch on remaining channels):
  (lam (shield0 (output (ite (linear_bool (embed $0_0))
                              (embed $0_0)
                              (embed $1_0)))))
```

### Run All Experiments

To run all baselines and NEAR for both label modes:

```bash
bash notebooks/ecg_reproduction/run_all_ecg_experiments.sh --device cuda:0
```

## 4. Analyzing the Results

The `analyze_ecg_results.ipynb` notebook loads baseline and NEAR results, generates comparison tables, and computes per-class AUC breakdowns.

Results are saved to:
- `outputs/ecg_results/comparison.csv`
- `outputs/ecg_results/comparison.md`

### Results

```
Macro AUC Comparison (best per method, fold 10 test set)
================================================================================
              NEAR-Attention    MLP  DecisionTree  RandomForest
multi                 0.8726 0.9002        0.7373        0.9047
single                0.8998 0.8974        0.6771        0.8876
================================================================================

Reference: RF on ECGDeli features (Mehari 2023) = 0.899 macro AUC
```

### Key Findings

1. **NEAR matches baselines on single-label**: The best discovered program achieves 0.900 macro AUC, matching the MLP baseline (0.897) and exceeding the single-label Random Forest (0.888).
2. **Shields don't seem to help.**
