#!/bin/bash
# Evaluate a pretrained backbone on HAR / FoG downstream datasets.
#
# Runs a per-dataset linear probe (or full finetune) on frozen SSL embeddings
# using frozen, subject-grouped splits. The heavy lifting (windowing, splits,
# label mapping for each dataset) lives in motion_dataloader.py +
# inertia1/eval/motion_linear_probe.py.
#
# Frozen splits are NOT shipped (they are derived from non-redistributable
# datasets). After preprocessing your own data copies, generate splits with
# inertia1/eval/motion_split_analysis.py and pass the directory via SPLIT_DIR.
#
# Required:
#   METHOD     ar_transformer | patchtst
#   MODE       linear_probe | full_finetune
#   CKPT_PATH  path to a pretrained checkpoint (last.ckpt)
#   SPLIT_DIR  directory of generated split JSONs
#
# Optional:
#   PRESET (small) DATA_ROOT (./data/downstream) DATASETS (see below)
#   PRECISION (32-true) OUT_ROOT (./logs)
#
# Usage:
#   METHOD=ar_transformer MODE=linear_probe CKPT_PATH=./outputs/.../last.ckpt \
#     DATA_ROOT=./data/downstream SPLIT_DIR=./splits bash scripts/eval/eval_motion.sh
set -euo pipefail

: "${METHOD:?METHOD must be ar_transformer|patchtst}"
: "${MODE:?MODE must be linear_probe|full_finetune}"
: "${CKPT_PATH:?CKPT_PATH is required}"
: "${SPLIT_DIR:?SPLIT_DIR (dir of generated split JSONs) is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PRESET="${PRESET:-small}"
DATA_ROOT="${DATA_ROOT:-./data/downstream}"
PRECISION="${PRECISION:-32-true}"
OUT_ROOT="${OUT_ROOT:-./logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-[wisdm,FoGTurning,OdayFoG,daphnet_fog,HHAR,OpportunityUCIDataset,Recofit,wear_dataset,harth,har70plus,MHEALTHDATASET,PAMAP2_Dataset,capture24_willetts]}"

OUT_BASE="$OUT_ROOT/motion_eval__${METHOD}__${MODE}"
mkdir -p "$OUT_BASE"

"$PYTHON_BIN" -m inertia1.run \
  --config "$REPO_ROOT/inertia1/config/default.yaml" \
  --config "$REPO_ROOT/inertia1/config/methods/${METHOD}.yaml" \
  --config "$REPO_ROOT/inertia1/config/presets/${PRESET}.yaml" \
  run.stage=finetune \
  run.ckpt_path="$CKPT_PATH" \
  logging.wandb.enabled=false \
  data.window_sec=30 data.hz=20 data.axes=3 \
  eval_motion.window_sec=30 eval_motion.sampling_rate=20 \
  eval_motion.extra_eval_window_secs=[] \
  eval_motion.split_path_or_dir="$SPLIT_DIR" \
  eval_motion.mode=overlap \
  eval_motion.overlap_stride_samples=20 \
  eval_motion.allow_oob_splits=true \
  eval_motion.datasets="$DATASETS" \
  eval_motion.preload=true \
  eval_motion.data_root="$DATA_ROOT" \
  eval_motion.finetune.mode="$MODE" \
  trainer.accelerator=cuda trainer.devices=1 trainer.precision="$PRECISION" \
  trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 \
  trainer.use_distributed_sampler=false \
  checkpointing.run_dir="$OUT_BASE"

echo "outputs: $OUT_BASE"
