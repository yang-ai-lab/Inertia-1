#!/bin/bash
# Pretrain the autoregressive Transformer (AR-Transformer) on accelerometry.
#
# The pretraining "exploration axes" are all controlled by `data.*` overrides:
#   - WINDOW_SEC : window length in seconds        (data.window_sec)
#   - HZ         : target sampling rate in Hz       (data.hz; must divide native_hz)
#   - AXES       : 3 = triaxial, 1 = uniaxial (L2)  (data.axes)
# Sweep these to reproduce the uniaxial-vs-triaxial / sampling-rate / window
# ablations. Model size is set by the preset (small | medium | large).
#
# Usage:
#   DATA_ROOT=./data/pretrain bash scripts/pretrain/pretrain_ar_transformer.sh
#   WINDOW_SEC=60 HZ=10 AXES=1 PRESET=small bash scripts/pretrain/pretrain_ar_transformer.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PRESET="${PRESET:-small}"
WINDOW_SEC="${WINDOW_SEC:-30}"
HZ="${HZ:-20}"
AXES="${AXES:-3}"
DATA_ROOT="${DATA_ROOT:-./data/pretrain}"
OUT_ROOT="${OUT_ROOT:-./outputs}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR="${LR:-1e-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "AR-Transformer pretrain | preset=$PRESET window=${WINDOW_SEC}s hz=${HZ} axes=${AXES}"

"$PYTHON_BIN" -m inertia1.run \
    --config inertia1/config/default.yaml \
    --config inertia1/config/methods/ar_transformer.yaml \
    --config inertia1/config/presets/${PRESET}.yaml \
    run.stage=pretrain \
    trainer.precision=bf16-mixed \
    trainer.max_steps=${MAX_STEPS} \
    optim.lr=${LR} \
    data.data_root=${DATA_ROOT} \
    data.window_sec=${WINDOW_SEC} \
    data.hz=${HZ} \
    data.axes=${AXES} \
    checkpointing.root_dir=${OUT_ROOT}
