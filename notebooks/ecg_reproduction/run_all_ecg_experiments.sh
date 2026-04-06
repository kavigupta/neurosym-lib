#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ecg_results/all_runs}"
DATA_DIR="${DATA_DIR:-data/ecg}"

NUM_PROGRAMS="${NUM_PROGRAMS:-200}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
NEURAL_HIDDEN_SIZE="${NEURAL_HIDDEN_SIZE:-32}"
NEAR_BATCH_SIZE="${NEAR_BATCH_SIZE:-1024}"
NEAR_EPOCHS="${NEAR_EPOCHS:-30}"
NEAR_FINAL_EPOCHS="${NEAR_FINAL_EPOCHS:-60}"
NEAR_LR="${NEAR_LR:-1e-4}"
STRUCTURAL_COST_PENALTY="${STRUCTURAL_COST_PENALTY:-0.1}"

MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-256}"
MLP_EPOCHS="${MLP_EPOCHS:-300}"
MLP_LR="${MLP_LR:-1e-3}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"

TREE_N_ESTIMATORS="${TREE_N_ESTIMATORS:-100}"

TRAIN_SEED="${TRAIN_SEED:-0}"

usage() {
  cat <<'EOF'
Run all ECG experiments (NEAR + baselines) for single and multi-label modes.

Usage:
  bash notebooks/ecg_reproduction/run_all_ecg_experiments.sh [options]

Options:
  --data-dir PATH            ECG data directory (default: data/ecg)
  --output-root PATH         Output root directory
  --device DEVICE            Torch device (default: cpu)
  --num-programs N           NEAR programs to discover per run
  --near-epochs N            Search-training epochs for NEAR
  --near-final-epochs N      Final-training epochs for NEAR
  --mlp-epochs N             Epochs for baseline MLP
  --train-seed N             Train seed used by all scripts
  -h, --help                 Show this help

Env vars with same names are also supported (e.g. DEVICE, DATA_DIR).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-programs) NUM_PROGRAMS="$2"; shift 2 ;;
    --near-epochs) NEAR_EPOCHS="$2"; shift 2 ;;
    --near-final-epochs) NEAR_FINAL_EPOCHS="$2"; shift 2 ;;
    --mlp-epochs) MLP_EPOCHS="$2"; shift 2 ;;
    --train-seed) TRAIN_SEED="$2"; shift 2 ;;
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

for mode in single multi; do
  mode_dir="$OUTPUT_ROOT/$mode"
  mkdir -p "$mode_dir"

  run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/benchmark_attention_ecg.py \
    --data-dir "$DATA_DIR" \
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

  run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_nn.py \
    --data-dir "$DATA_DIR" \
    --label-mode "$mode" \
    --batch-size "$MLP_BATCH_SIZE" \
    --epochs "$MLP_EPOCHS" \
    --lr "$MLP_LR" \
    --hidden-dim "$MLP_HIDDEN_DIM" \
    --seed "$TRAIN_SEED" \
    --device "$DEVICE" \
    --output "$mode_dir/baseline_nn.json"

  run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_tree.py \
    --data-dir "$DATA_DIR" \
    --model decision_tree \
    --label-mode "$mode" \
    --seed "$TRAIN_SEED" \
    --n-estimators "$TREE_N_ESTIMATORS" \
    --output "$mode_dir/baseline_tree_decision_tree.json"

  run_cmd "$PYTHON_BIN" notebooks/ecg_reproduction/baseline_tree.py \
    --data-dir "$DATA_DIR" \
    --model random_forest \
    --label-mode "$mode" \
    --seed "$TRAIN_SEED" \
    --n-estimators "$TREE_N_ESTIMATORS" \
    --output "$mode_dir/baseline_tree_random_forest.json"
done

echo ""
echo "All ECG experiments completed."
echo "Outputs written under: $OUTPUT_ROOT"
