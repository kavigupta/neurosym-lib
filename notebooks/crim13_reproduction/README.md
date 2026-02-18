# CRIM-13 (Mice) NEAR Experiment Reproduction

This directory contains scripts to reproduce the NEAR experiments on the CRIM-13 dataset for mice behavior classification.

## 1. Dataset Preparation

The CRIM-13 dataset is pre-processed and located in `data/mice_classification/crim13/`. No further preparation is needed.

## 2. Running the Experiment

To run the NEAR experiment, execute the `benchmark_crim13.py` script. You can customize the experiment using various command-line arguments.

```bash
uv run python notebooks/crim13_reproduction/benchmark_crim13.py --num-programs 20 --epochs 50 --device cpu
```

### Available Arguments

*   `--output` (str, default: `outputs/crim13_results/reproduction.pkl`): Path to save results.
*   `--num-programs` (int, default: `40`): Number of programs to discover during the search.
*   `--hidden-dim` (int, default: `16`): Hidden dimension for the Domain-Specific Language (DSL).
*   `--neural-hidden-size` (int, default: `16`): Hidden size for the neural hole filler.
*   `--batch-size` (int, default: `2000`): Training batch size.
*   `--epochs` (int, default: `30`): Number of epochs for training during the search phase.
*   `--final-epochs` (int, default: `40`): Number of epochs for final training of discovered programs.
*   `--lr` (float, default: `1e-4`): Learning rate for training.
*   `--structural-cost-penalty` (float, default: `0.01`): Penalty multiplier for the structural cost in search.
*   `--device` (str, default: `cuda:0`): Device to use for training (e.g., `cuda:0` or `cpu`).
*   `--behavior` (str, default: `sniff`, choices: `sniff`, `other`): The specific behavior to use for training and evaluation.
*   `--evaluate-reported` (action: `store_true`): If set, evaluates the reported program from the MICE-DSL paper instead of running a new search.

To evaluate the reported program from the MICE-DSL paper:

```bash
uv run python notebooks/crim13_reproduction/benchmark_crim13.py --evaluate-reported
```

This will save the results to `outputs/crim13_results/reported_program.pkl`.

## 3. Analyzing the Results

The `analyze_crim13_results.ipynb` notebook can be used to analyze the results from the experiment. It loads the `reproduction_sniff.pkl` and `reported_program.pkl` files, as well as baseline results, and generates a comparison of the discovered programs, including summary statistics and a table of the top programs.

The analysis notebook saves its results to:
- `outputs/crim13_results/comparison.csv`
- `outputs/crim13_results/comparison.md`

### Expected Results Snippet

After running the analysis notebook, you can expect to see a table summarizing the performance of the discovered programs, sorted by Hamming accuracy:

```
================================================================================
RESULTS COMPARISON
================================================================================
                     experiment  precision   recall  f1_score  support  hamming_accuracy         time
         crim13_enumeration_010   0.446849 0.500000  0.471933 295300.0          0.893698 81125.326581
          crim13_astar-near_001   0.446849 0.500000  0.471933 295300.0          0.893698  1563.009056
neurosym_reproduction_sniff_015   0.446849 0.500000  0.471933 295300.0          0.893698   137.301126
...
================================================================================
```
