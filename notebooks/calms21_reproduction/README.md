# CALMS21 (Investigation) NEAR Experiment Reproduction

This directory contains scripts to reproduce the NEAR experiments on the CALMS21 dataset for investigation vs. not investigation classification.

## 1. Dataset Preparation

The pre-processed CALMS21 Task 1 investigation dataset is located in `data/mice_classification/calms21_task1/`. No further preparation is needed if the data is already present.

If you need to download the dataset, see the `tutorial/near_demo_behavior_classification.ipynb` notebook for the `gdown` download command that produces the `calms21_task1` folder.

## 2. Running the Experiment

To run the NEAR experiment, execute the `benchmark_calms21.py` script. You can customize the experiment using various command-line arguments.

```bash
uv run python notebooks/calms21_reproduction/benchmark_calms21.py --num-programs 20 --epochs 12 --device cpu
```

### Available Arguments

*   `--output` (str, default: `outputs/calms21_results/reproduction.pkl`): Path to save results.
*   `--num-programs` (int, default: `40`): Number of programs to discover during the search.
*   `--hidden-dim` (int, default: `16`): Hidden dimension for the Domain-Specific Language (DSL).
*   `--neural-hidden-size` (int, default: `16`): Hidden size for the neural hole filler.
*   `--batch-size` (int, default: `1024`): Training batch size.
*   `--epochs` (int, default: `12`): Number of epochs for training during the search phase.
*   `--final-epochs` (int, default: `40`): Number of epochs for final training of discovered programs.
*   `--lr` (float, default: `1e-4`): Learning rate for training.
*   `--structural-cost-penalty` (float, default: `0.05`): Penalty multiplier for the structural cost in search.
*   `--max-depth` (int, default: `6`): Maximum program depth.
*   `--device` (str, default: `cuda:0`): Device to use for training (e.g., `cuda:0` or `cpu`).

## 3. Results

The script saves:
- Raw programs to `outputs/calms21_results/reproduction_raw.pkl`
- Final evaluated programs to `outputs/calms21_results/reproduction.pkl`
- A summary JSON to `outputs/calms21_results/reproduction_summary.json`

## 4. Analyzing the Results

The `analyze_calms21_results.ipynb` notebook can be used to analyze the results from the experiment. It loads the `reproduction.pkl` file (and any available baseline results) and generates a comparison of the discovered programs, including summary statistics and a table of the top programs.

The analysis notebook saves its results to:
- `outputs/calms21_results/comparison.csv`
- `outputs/calms21_results/comparison.md`
