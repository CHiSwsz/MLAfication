#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$MODEL_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"
# Activate your environment before running this script.
export WANDB_MODE="${WANDB_MODE:-offline}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --nproc_per_node "${NPROC_PER_NODE:-1}" \
    "$MODEL_DIR/train_stage2.py" \
    --cfg_file "${CFG_FILE:-$MODEL_DIR/cfgs/stage2.yaml}"
