#!/bin/bash
# Patient-level disease/medication detection via attention MIL on frozen
# backbone embeddings. Demonstrates the pipeline on placeholder data; see the
# README ("Disease detection") for the NHANES pointer.
#
# Required:
#   CKPT_PATH  path to a pretrained checkpoint (optional but recommended)
#
# Optional:
#   METHOD (ar_transformer) PRESET (small) TASK_KIND (binary) NUM_CLASSES (2)
#   DATA_DIR (./data/disease) BAG_SIZE (256) EPOCHS (20)
#
# Quick start (synthetic placeholder data):
#   python scripts/disease_detection/make_placeholder_data.py --out ./data/disease
#   CKPT_PATH=./outputs/.../last.ckpt bash scripts/disease_detection/eval_disease.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

METHOD="${METHOD:-ar_transformer}"
PRESET="${PRESET:-small}"
TASK="${TASK:-disease}"
TASK_KIND="${TASK_KIND:-binary}"
NUM_CLASSES="${NUM_CLASSES:-2}"
DATA_DIR="${DATA_DIR:-./data/disease}"
BAG_SIZE="${BAG_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
OUT_DIR="${OUT_DIR:-./logs/disease}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_ARG=()
if [[ -n "${CKPT_PATH:-}" ]]; then CKPT_ARG=(--ckpt-path "$CKPT_PATH"); fi

"$PYTHON_BIN" -m inertia1.scripts.eval_nhanes_labels \
  --data-root "$DATA_DIR/windows" \
  --labels-csv "$DATA_DIR/labels.csv" \
  --split-csv "$DATA_DIR/split.csv" \
  --task "$TASK" --task-kind "$TASK_KIND" --num-classes "$NUM_CLASSES" \
  --method "$METHOD" --preset "$PRESET" \
  "${CKPT_ARG[@]}" \
  --bag-size "$BAG_SIZE" --epochs "$EPOCHS" \
  --class-balanced \
  --out-dir "$OUT_DIR"
