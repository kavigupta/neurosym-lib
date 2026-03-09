#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ecg_results/all_runs}"

NUM_PROGRAMS="${NUM_PROGRAMS:-200}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
NEURAL_HIDDEN_SIZE="${NEURAL_HIDDEN_SIZE:-32}"
NEAR_BATCH_SIZE="${NEAR_BATCH_SIZE:-1024}"
NEAR_EPOCHS="${NEAR_EPOCHS:-30}"
NEAR_FINAL_EPOCHS="${NEAR_FINAL_EPOCHS:-60}"
NEAR_LR="${NEAR_LR:-1e-4}"
STRUCTURAL_COST_PENALTY="${STRUCTURAL_COST_PENALTY:-0.1}"

MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-256}"
MLP_EPOCHS="${MLP_EPOCHS:-20}"
MLP_LR="${MLP_LR:-1e-3}"

TREE_N_ESTIMATORS="${TREE_N_ESTIMATORS:-200}"
TREE_MAX_DEPTH="${TREE_MAX_DEPTH:-15}"

TORCH_BASELINE_BATCH_SIZE="${TORCH_BASELINE_BATCH_SIZE:-256}"
TORCH_BASELINE_EPOCHS="${TORCH_BASELINE_EPOCHS:-20}"
TORCH_BASELINE_LR="${TORCH_BASELINE_LR:-1e-3}"

TRAIN_SEED="${TRAIN_SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"

EXTRA_DATASETS="${EXTRA_DATASETS:-}"
DB_DIR="${DB_DIR:-}"
WORKING_DIR="${WORKING_DIR:-}"
DATASET_KWARGS_JSON="${DATASET_KWARGS_JSON:-{}}"
MAX_RECORDS="${MAX_RECORDS:-5000}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
TEST_FRACTION="${TEST_FRACTION:-0.15}"

RUN_PROVENANCE_CHECK=1
RUN_GENERAL_DIRECT=1

usage() {
  cat <<'EOF'
Run all ECG DSL and baseline experiments for:
1) local standardized ECG dataset
2) optional extra torch-ecg datasets

Usage:
  bash notebooks/ecg_reproduction/run_all_ecg_experiments.sh [options]

Options:
  --extra-datasets CSV         Comma-separated torch-ecg dataset names (e.g. CPSC2018,CINC2021)
  --db-dir PATH                torch-ecg db_dir for extra datasets
  --working-dir PATH           torch-ecg working_dir for extra datasets
  --dataset-kwargs-json JSON   Extra kwargs for torch-ecg dataset constructors
  --output-root PATH           Output root directory
  --device DEVICE              Torch device (default: cpu)
  --num-programs N             NEAR programs to discover per run
  --near-epochs N              Search-training epochs for NEAR
  --near-final-epochs N        Final-training epochs for NEAR
  --mlp-epochs N               Epochs for baseline_nn.py
  --torch-baseline-epochs N    Epochs for baseline_torch_ecg.py
  --max-records N              Max records when preparing extra datasets
  --val-fraction F             Validation fraction when preparing extra datasets
  --test-fraction F            Test fraction when preparing extra datasets
  --split-seed N               Split seed for extra datasets
  --train-seed N               Train seed used by all scripts
  --skip-provenance            Skip verify_ecg_provenance.py on local dataset
  --skip-general-direct        Skip benchmark_ecg_general.py runs for extra datasets
  -h, --help                   Show this help

Env vars with same names are also supported (e.g. DEVICE, EXTRA_DATASETS).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra-datasets) EXTRA_DATASETS="$2"; shift 2 ;;
    --db-dir) DB_DIR="$2"; shift 2 ;;
    --working-dir) WORKING_DIR="$2"; shift 2 ;;
    --dataset-kwargs-json) DATASET_KWARGS_JSON="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-programs) NUM_PROGRAMS="$2"; shift 2 ;;
    --near-epochs) NEAR_EPOCHS="$2"; shift 2 ;;
    --near-final-epochs) NEAR_FINAL_EPOCHS="$2"; shift 2 ;;
    --mlp-epochs) MLP_EPOCHS="$2"; shift 2 ;;
    --torch-baseline-epochs) TORCH_BASELINE_EPOCHS="$2"; shift 2 ;;
    --max-records) MAX_RECORDS="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --test-fraction) TEST_FRACTION="$2"; shift 2 ;;
    --split-seed) SPLIT_SEED="$2"; shift 2 ;;
    --train-seed) TRAIN_SEED="$2"; shift 2 ;;
    --skip-provenance) RUN_PROVENANCE_CHECK=0; shift ;;
    --skip-general-direct) RUN_GENERAL_DIRECT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"

run_cmd() {
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  "$@"
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//'
}

prepare_extra_dataset() {
  local dataset_name="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing standardized splits for ${dataset_name} -> ${out_dir}"

  DATASET_NAME="$dataset_name" \
  OUT_DIR="$out_dir" \
  DB_DIR="${DB_DIR}" \
  WORKING_DIR="${WORKING_DIR}" \
  DATASET_KWARGS_JSON="${DATASET_KWARGS_JSON}" \
  MAX_RECORDS="${MAX_RECORDS}" \
  VAL_FRACTION="${VAL_FRACTION}" \
  TEST_FRACTION="${TEST_FRACTION}" \
  SPLIT_SEED="${SPLIT_SEED}" \
  TRAIN_SEED="${TRAIN_SEED}" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import neurosym as ns

dataset_name = os.environ["DATASET_NAME"]
out_dir = Path(os.environ["OUT_DIR"])
db_dir = os.environ.get("DB_DIR") or None
working_dir = os.environ.get("WORKING_DIR") or None
max_records_env = int(os.environ.get("MAX_RECORDS", "0"))
max_records = max_records_env if max_records_env > 0 else None
val_fraction = float(os.environ.get("VAL_FRACTION", "0.15"))
test_fraction = float(os.environ.get("TEST_FRACTION", "0.15"))
split_seed = int(os.environ.get("SPLIT_SEED", "42"))
train_seed = int(os.environ.get("TRAIN_SEED", "0"))
dataset_kwargs = json.loads(os.environ.get("DATASET_KWARGS_JSON", "{}"))
if not isinstance(dataset_kwargs, dict):
    raise ValueError("DATASET_KWARGS_JSON must decode to an object.")

dm = ns.datasets.torch_ecg_data_example(
    train_seed=train_seed,
    dataset_name=dataset_name,
    db_dir=db_dir,
    working_dir=working_dir,
    label_mode="multi",
    is_regression=True,
    max_records=max_records,
    val_fraction=val_fraction,
    test_fraction=test_fraction,
    split_seed=split_seed,
    normalize_per_split=True,
    batch_size=1024,
    dataset_kwargs=dataset_kwargs,
)

train_x = dm.train.inputs.astype(np.float32)
valid_x = dm.val.inputs.astype(np.float32)
test_x = dm.test.inputs.astype(np.float32)
train_y_multi = dm.train.outputs.astype(np.float32)
valid_y_multi = dm.val.outputs.astype(np.float32)
test_y_multi = dm.test.outputs.astype(np.float32)

train_y_single = train_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
valid_y_single = valid_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)
test_y_single = test_y_multi.argmax(axis=-1).astype(np.int64).reshape(-1, 1)

out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_dir / "train_ecg_data.npz", train_x)
np.savez_compressed(out_dir / "valid_ecg_data.npz", valid_x)
np.savez_compressed(out_dir / "test_ecg_data.npz", test_x)
np.savez_compressed(out_dir / "train_ecg_labels_multi.npz", train_y_multi)
np.savez_compressed(out_dir / "valid_ecg_labels_multi.npz", valid_y_multi)
np.savez_compressed(out_dir / "test_ecg_labels_multi.npz", test_y_multi)
np.savez_compressed(out_dir / "train_ecg_labels_single.npz", train_y_single)
np.savez_compressed(out_dir / "valid_ecg_labels_single.npz", valid_y_single)
np.savez_compressed(out_dir / "test_ecg_labels_single.npz", test_y_single)

metadata = {
    "dataset_name": dataset_name,
    "db_dir": db_dir,
    "working_dir": working_dir,
    "max_records": max_records,
    "val_fraction": val_fraction,
    "test_fraction": test_fraction,
    "split_seed": split_seed,
    "train_seed": train_seed,
    "dataset_kwargs": dataset_kwargs,
    "n_train": int(train_x.shape[0]),
    "n_val": int(valid_x.shape[0]),
    "n_test": int(test_x.shape[0]),
    "input_dim": int(train_x.shape[-1]),
    "num_classes": int(train_y_multi.shape[-1]),
    "class_names": getattr(dm, "class_names", []),
}
with open(out_dir / "source_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Wrote standardized dataset to: {out_dir}")
PY
}

run_all_for_dataset() {
  local dataset_key="$1"
  local data_dir="$2"
  local dataset_output_root="$OUTPUT_ROOT/$dataset_key"
  mkdir -p "$dataset_output_root"

  for mode in single multi; do
    local mode_dir="$dataset_output_root/$mode"
    mkdir -p "$mode_dir"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/benchmark_ecg.py \
      --data-dir "$data_dir" \
      --label-mode "$mode" \
      --output "$mode_dir/near_simple.pkl" \
      --num-programs "$NUM_PROGRAMS" \
      --hidden-dim "$HIDDEN_DIM" \
      --neural-hidden-size "$NEURAL_HIDDEN_SIZE" \
      --batch-size "$NEAR_BATCH_SIZE" \
      --epochs "$NEAR_EPOCHS" \
      --final-epochs "$NEAR_FINAL_EPOCHS" \
      --lr "$NEAR_LR" \
      --structural-cost-penalty "$STRUCTURAL_COST_PENALTY" \
      --device "$DEVICE"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/benchmark_attention_ecg.py \
      --data-dir "$data_dir" \
      --label-mode "$mode" \
      --output "$mode_dir/near_attention.pkl" \
      --num-programs "$NUM_PROGRAMS" \
      --hidden-dim "$HIDDEN_DIM" \
      --neural-hidden-size "$NEURAL_HIDDEN_SIZE" \
      --batch-size "$NEAR_BATCH_SIZE" \
      --epochs "$NEAR_EPOCHS" \
      --final-epochs "$NEAR_FINAL_EPOCHS" \
      --lr "$NEAR_LR" \
      --structural-cost-penalty "$STRUCTURAL_COST_PENALTY" \
      --device "$DEVICE"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/benchmark_attention_drop_eg.py \
      --data-dir "$data_dir" \
      --label-mode "$mode" \
      --output "$mode_dir/near_attention_drop_eg.pkl" \
      --num-programs "$NUM_PROGRAMS" \
      --hidden-dim "$HIDDEN_DIM" \
      --neural-hidden-size "$NEURAL_HIDDEN_SIZE" \
      --batch-size "$NEAR_BATCH_SIZE" \
      --epochs "$NEAR_EPOCHS" \
      --final-epochs "$NEAR_FINAL_EPOCHS" \
      --lr "$NEAR_LR" \
      --structural-cost-penalty "$STRUCTURAL_COST_PENALTY" \
      --device "$DEVICE"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_nn.py \
      --data-dir "$data_dir" \
      --label-mode "$mode" \
      --batch-size "$MLP_BATCH_SIZE" \
      --epochs "$MLP_EPOCHS" \
      --lr "$MLP_LR" \
      --seed "$TRAIN_SEED" \
      --device "$DEVICE" \
      --output "$mode_dir/baseline_nn.json"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_tree.py \
      --data-dir "$data_dir" \
      --model decision_tree \
      --label-mode "$mode" \
      --seed "$TRAIN_SEED" \
      --max-depth "$TREE_MAX_DEPTH" \
      --n-estimators "$TREE_N_ESTIMATORS" \
      --output "$mode_dir/baseline_tree_decision_tree.json"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_tree.py \
      --data-dir "$data_dir" \
      --model random_forest \
      --label-mode "$mode" \
      --seed "$TRAIN_SEED" \
      --max-depth "$TREE_MAX_DEPTH" \
      --n-estimators "$TREE_N_ESTIMATORS" \
      --output "$mode_dir/baseline_tree_random_forest.json"

    run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_torch_ecg.py \
      --data-dir "$data_dir" \
      --label-mode "$mode" \
      --train-seed "$TRAIN_SEED" \
      --split-seed "$SPLIT_SEED" \
      --batch-size "$TORCH_BASELINE_BATCH_SIZE" \
      --epochs "$TORCH_BASELINE_EPOCHS" \
      --lr "$TORCH_BASELINE_LR" \
      --device "$DEVICE" \
      --output "$mode_dir/baseline_torch_ecg.json"
  done
}

if [[ "$RUN_PROVENANCE_CHECK" -eq 1 ]]; then
  run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/verify_ecg_provenance.py
fi

# 1) Local standardized ECG dataset
run_all_for_dataset "local_ecg" "data/ecg_classification/ecg"

# 2) Optional extra torch-ecg datasets
if [[ -n "$EXTRA_DATASETS" ]]; then
  IFS=',' read -r -a DATASET_ARRAY <<< "$EXTRA_DATASETS"
  for raw_name in "${DATASET_ARRAY[@]}"; do
    dataset_name="$(echo "$raw_name" | xargs)"
    if [[ -z "$dataset_name" ]]; then
      continue
    fi

    dataset_key="$(slugify "$dataset_name")"
    prepared_dir="$OUTPUT_ROOT/prepared_data/$dataset_key"
    prepare_extra_dataset "$dataset_name" "$prepared_dir"
    run_all_for_dataset "$dataset_key" "$prepared_dir"

    if [[ "$RUN_GENERAL_DIRECT" -eq 1 ]]; then
      for mode in single multi; do
        mode_dir="$OUTPUT_ROOT/$dataset_key/$mode"
        mkdir -p "$mode_dir"
        cmd=(
          "$PYTHON_BIN" notebooks/ecg_reproduction/benchmark_ecg_general.py
          --dataset-name "$dataset_name"
          --dataset-kwargs-json "$DATASET_KWARGS_JSON"
          --max-records "$MAX_RECORDS"
          --val-fraction "$VAL_FRACTION"
          --test-fraction "$TEST_FRACTION"
          --split-seed "$SPLIT_SEED"
          --output "$mode_dir/near_simple_general_direct.pkl"
          --num-programs "$NUM_PROGRAMS"
          --hidden-dim "$HIDDEN_DIM"
          --neural-hidden-size "$NEURAL_HIDDEN_SIZE"
          --batch-size "$NEAR_BATCH_SIZE"
          --epochs "$NEAR_EPOCHS"
          --final-epochs "$NEAR_FINAL_EPOCHS"
          --lr "$NEAR_LR"
          --structural-cost-penalty "$STRUCTURAL_COST_PENALTY"
          --device "$DEVICE"
          --label-mode "$mode"
        )
        if [[ -n "$DB_DIR" ]]; then
          cmd+=(--db-dir "$DB_DIR")
        fi
        if [[ -n "$WORKING_DIR" ]]; then
          cmd+=(--working-dir "$WORKING_DIR")
        fi
        run_cmd "${cmd[@]}"
      done
    fi
  done
fi

echo ""
echo "All ECG experiments completed."
echo "Outputs written under: $OUTPUT_ROOT"
