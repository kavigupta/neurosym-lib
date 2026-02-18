# Basketball NEAR Experiment Reproduction

This directory contains scripts to reproduce the NEAR experiments on the Basketball dataset for offensive versus defensive behavior classification.

## 1. Dataset Preparation

The raw dataset is located in `data/basketball_classification/bball_process/`. To prepare the data for the experiment, you need to run the relevant cell in the `notebooks/get_near_datasets.ipynb` notebook.

Specifically, execute the cell that contains the following code:
```python
# get the basketball data in the correct format.
bball_folder = Path("./data/basketball_classification/bball_process/")
map_types = {
    "data" : "data",
    "label" : "labels",
    "labels" : "labels",
}
# ballhandler

for file_path in sorted(bball_folder.glob("*.npy")):
    split, typ = file_path.stem.replace("_basket", "").split("_")
    typ = map_types[typ]
    save_path = bball_folder.parent / "bball" / f"{split}_bball_{typ}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if typ == 'data':
        data = np.load(file_path, allow_pickle=True).astype(np.float32)
        print(save_path.stem, data.shape, data.dtype)
        np.savez_compressed(save_path, data)
    elif typ in "labels":
        labels = np.load(file_path, allow_pickle=True).astype(np.int64)
        is_ballhandler = (labels == 1).astype(np.int64)
        print(save_path.stem, labels.shape, len(np.unique(labels)), labels.dtype)
        np.savez_compressed(save_path, labels)
```
This will process the raw data and save it in the correct format in the `data/basketball_classification/bball/` directory.

## 2. Running the Experiment

To run the NEAR experiment, execute the `benchmark_bball.py` script. You can customize the experiment using various command-line arguments.

```bash
uv run python notebooks/bball_reproduction/benchmark_bball.py --num-programs 20 --epochs 50 --device cpu
```

### Available Arguments

*   `--output` (str, default: `outputs/bball_results/reproduction.pkl`): Path to save results.
*   `--num-programs` (int, default: `40`): Number of programs to discover during the search.
*   `--hidden-dim` (int, default: `16`): Hidden dimension for the Domain-Specific Language (DSL).
*   `--neural-hidden-size` (int, default: `16`): Hidden size for the neural hole filler.
*   `--batch-size` (int, default: `4000`): Training batch size.
*   `--epochs` (int, default: `30`): Number of epochs for training during the search phase.
*   `--final-epochs` (int, default: `40`): Number of epochs for final training of discovered programs.
*   `--lr` (float, default: `1e-4`): Learning rate for training.
*   `--structural-cost-penalty` (float, default: `0.1`): Penalty multiplier for the structural cost in search.
*   `--device` (str, default: `cuda:0`): Device to use for training (e.g., `cuda:0` or `cpu`).

## 3. Analyzing the Results

The `analyze_bball_results.ipynb` notebook can be used to analyze the results from the experiment. It loads the `reproduction.pkl` file and generates a comparison of the discovered programs, including summary statistics and a table of the top programs.

The analysis notebook saves its results to:
- `outputs/bball_results/comparison.csv`
- `outputs/bball_results/comparison.md`

### Expected Results Snippet

After running the analysis notebook, you can expect to see a table summarizing the performance of the discovered programs, sorted by Hamming accuracy:

```
================================================================================
RESULTS COMPARISON
================================================================================
               experiment  precision   recall  f1_score  support  hamming_accuracy        time
neurosym_reproduction_000   0.101749 0.239139  0.142236  67325.0          0.323713  241.053499
neurosym_reproduction_039   0.067619 0.163255  0.036533  67325.0          0.115084 1004.492519
neurosym_reproduction_033   0.067554 0.160154  0.037228  67325.0          0.113450  829.661858
...
================================================================================
```

> [!IMPORTANT]
> The reproduction of this experiment is not complete due to missing evaluation signals in the original codebase. The results may not match the reported results in the NEAR paper.
