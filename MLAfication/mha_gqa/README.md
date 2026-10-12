# MLAfication for MHA/GQA Models

This directory contains the MLAfication implementation for pretrained MHA and
GQA models, including Llama, Qwen2/Qwen2.5, and SmolLM. The implementation for
QKNorm-based models is provided in the sibling `qknorm/` directory.

[← MLAfication guide](../README.md)

Run the commands below from this implementation directory (`MLAfication/mha_gqa/` in the unified repository).

## Repository layout

- `gen/convert.py`: DeltaRoPE selection and SVDLLM V2 initialization.
- `gen/patch_func.py` and `gen/svdllm.py`: conversion primitives.
- `src/mha2mla/train.py`: logit-distillation recovery training.
- `src/mha2mla/patching_*.py`: MLA parameterization and forward pass.
- `eval/`: short-context and LongBench evaluation entry points.

## Workflow

1. Install the environment from the parent-project [`requirements.txt`](../requirements.txt) after
   selecting the CUDA-enabled PyTorch build for the host.
2. Set `MHA2MLA_DATA_PATH` and generate MLA initialization tensors. SVDLLM V2
   is the default; model paths and experimental parameters are command-line
   options rather than source edits:

   ```bash
   torchrun --nproc_per_node 4 gen/convert.py \
     --model_name_or_path Qwen/Qwen2.5-7B \
     --output_path artifacts/qwen2.5-7b-rank256.pt \
     --lora_rank 256 --topk 8 --num_samples 128 --sequence_length 4096
   ```

3. Create a training YAML using the fields in `src/mha2mla/arguments.py`.
   Recovery training preserves the original required `--cfg_file` interface:

   ```bash
   torchrun --nproc_per_node 4 src/mha2mla/train.py \
     --cfg_file /path/to/train.yml
   ```

4. Use `eval/eval_v2.py` and `eval/longbench.py` for evaluation. LongBench
   likewise requires `--cfg_file`. QKV-only
   checkpoints are loaded on top of the original frozen model. Use
   `src/mha2mla/export_mla_checkpoint.py --adapter_checkpoint ...` to export a
   self-contained full checkpoint when needed.

Supply credentials for gated Hugging Face resources through the environment
when needed:

```bash
export HF_TOKEN=your_token
```

Do not commit credentials, model weights, generated tensors, checkpoints,
datasets, logs, or evaluation outputs. See the [parent-project README](../README.md) for
LongBench data overrides, quantized evaluation, and decode benchmarking.
