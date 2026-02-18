# Fly-v-Fly NEAR Experiment Reproduction

This directory contains scripts to reproduce the NEAR experiments on the Fly-v-Fly dataset for behavior classification.

## 1. Dataset Preparation

The raw dataset is located in `data/fruitflies_classification/fly_process/`. To prepare the data for the experiment, you need to run the relevant cell in the `notebooks/get_near_datasets.ipynb` notebook.

Specifically, execute the cell that contains the following code:
```python
# Collate Fly-vs.-Fly sequences into fixed-length tensors and export train/val/test splits.
fruitflies_folder = Path("./data/fruitflies_classification/fly_process/")
output_folder = fruitflies_folder.parent / "flyvfly"
output_folder.mkdir(parents=True, exist_ok=True)

MAX_LEN = 300
FEATURE_DIM = 53

# ... (rest of the cell)
```
This will process the raw data and save it in the correct format in the `data/fruitflies_classification/flyvfly/` directory.

## 2. Running the Experiment

To run the NEAR experiment, execute the `benchmark_flyvfly.py` script. You can customize the experiment using various command-line arguments.

```bash
uv run python notebooks/flyvfly_reproduction/benchmark_flyvfly.py --num-programs 20 --epochs 50 --device cpu
```

### Available Arguments

*   `--output` (str, default: `outputs/flyvfly_results/reproduction.pkl`): Path to save results.
*   `--num-programs` (int, default: `40`): Number of programs to discover during the search.
*   `--hidden-dim` (int, default: `16`): Hidden dimension for the Domain-Specific Language (DSL).
*   `--neural-hidden-size` (int, default: `16`): Hidden size for the neural hole filler.
*   `--batch-size` (int, default: `2048`): Training batch size.
*   `--epochs` (int, default: `30`): Number of epochs for training during the search phase.
*   `--final-epochs` (int, default: `40`): Number of epochs for final training of discovered programs.
*   `--lr` (float, default: `1e-4`): Learning rate for training.
*   `--structural-cost-penalty` (float, default: `0.1`): Penalty multiplier for the structural cost in search.
*   `--device` (str, default: `cuda:0`): Device to use for training (e.g., `cuda:0` or `cpu`).

## 3. Analyzing the Results

The `analyze_flyvfly_results.ipynb` notebook can be used to analyze the results from the experiment. It loads the `reproduction.pkl` file and generates a comparison of the discovered programs, including summary statistics and a table of the top programs.

The analysis notebook saves its results to:
- `outputs/flyvfly_results/comparison.csv`
- `outputs/flyvfly_results/comparison.md`

### Expected Results Snippet

After running the analysis notebook, you can expect to see a table summarizing the performance of the discovered programs, sorted by Hamming accuracy:

```
================================================================================
RESULTS COMPARISON
================================================================================
               experiment  precision   recall  f1_score  support  hamming_accuracy        time
neurosym_reproduction_036   0.231719 0.168844  0.131285   1050.0          0.458095 1997.639672
neurosym_reproduction_005   0.065306 0.142857  0.089636   1050.0          0.457143  175.275669
...
================================================================================
```

> [!IMPORTANT]
> The reproduction of this experiment is not complete due to missing evaluation signals in the original codebase. The results may not match the reported results in the NEAR paper.
