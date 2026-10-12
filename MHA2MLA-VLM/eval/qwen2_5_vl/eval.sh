#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src/mha2mla-vlm/qwen2_5_vl:$PROJECT_DIR/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
num_processes="${NPROC_PER_NODE:-8}"

model_path="${MODEL_PATH:-qwen2_5_vl/checkpoints/stage2_ckpt}"



accelerate launch \
    --num_processes=${num_processes} \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=${model_path},max_pixels=12845056 \
    --tasks ai2d,gqa,pope,seedbench,realworldqa,mmbench_en,chartqa,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path eval/qwen2_5_vl/logs
