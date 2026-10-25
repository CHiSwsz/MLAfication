# MLAfication for QKNorm Models

This directory contains the MLAfication implementation for QKNorm-based GQA
and MoE models, including Qwen3 and Qwen3-MoE. The implementation for standard
MHA/GQA models is provided in the sibling `mha_gqa/` directory.

[← MLAfication guide](../README.md)

Run the commands below from this implementation directory (`MLAfication/qknorm/` in the unified repository).

## Repository layout

- `gen/convert.py`: DeltaRoPE selection, QKNorm absorption, and SVDLLM V2
  initialization.
- `gen/patch_func.py` and `gen/svdllm.py`: conversion primitives.
- `src/mha2mla/train.py`: logit-distillation recovery training.
- `src/mha2mla/patching_*.py`: QKNorm-compatible MLA and latent-cache forward
  passes.
- `eval/`: short-context, LongBench, QKNorm, and MoE evaluation entry points.

## Workflow

1. Follow the parent-project [environment installation](../README.md#environment),
   including its security overrides, after selecting the CUDA-enabled PyTorch
   build for the host.
2. Set `MHA2MLA_DATA_PATH` and generate MLA initialization tensors. The same
   entry point detects dense Qwen3 and Qwen3-MoE from the model config:

   ```bash
   torchrun --nproc_per_node 4 gen/convert.py \
     --model_name_or_path Qwen/Qwen3-8B-Base \
     --output_path artifacts/qwen3-8b-rank512.pt \
     --lora_rank 512 --topk 8 --num_samples 128 --sequence_length 4096
   ```

3. Create a training YAML using the fields in `src/mha2mla/arguments.py`.
   Recovery training preserves the original required `--cfg_file` interface:

   ```bash
   torchrun --nproc_per_node 4 src/mha2mla/train.py \
     --cfg_file /path/to/train.yml
   ```

4. Use `eval/eval_v2.py` and `eval/longbench.py` for the main evaluation.
   LongBench likewise requires `--cfg_file`.
   `collect_attn_logits_max.py` and `collect_moe_vio.py` reproduce the QKNorm
   and MoE diagnostics. QKV-only checkpoints bundle their MLA tensors and load
   on top of the original base model; `src/mha2mla/export_mla_checkpoint.py`
   merges a delta into a full portable checkpoint.

Supply credentials for gated Hugging Face resources through the environment
when needed:

```bash
export HF_TOKEN=your_token
```

LongBench downloads `THUDM/LongBench` directly. Quantized and FP8 latent-cache
evaluation are controlled with `cache_implementation`, `backend`, `nbits`, and
`residual_length`; [`benchmark_decode.py`](../benchmark_decode.py) in the parent project directory measures decode
latency, throughput, and peak memory without a YAML file.

Do not commit credentials, model weights, generated tensors, checkpoints,
datasets, logs, virtual environments, or evaluation outputs.

## Token-wise KNorm statistics (Table 1)

`alpha` is the mean of the pre-KNorm key RMS denominator for a fixed layer and
KV head. To measure how well this constant represents individual tokens, report
the **standard deviation across token-wise norms at each context length**. This
is different from the standard deviation of the five context-length means or
the standard error of the estimated mean.

Run the diagnostic on one GPU, using the same model and calibration data as the
table. For example, after setting `MHA2MLA_DATA_PATH` to the packed calibration
corpus (or `MHA2MLA_HF_DATASET` to a local/Hugging Face text dataset):

```bash
python eval/collect_knorm_stats.py \
  --model_name_or_path Qwen/Qwen3-8B-Base \
  --context_lengths 1024 2048 4096 8192 16384 \
  --num_samples 64 --batch_size 1 --dtype bfloat16 \
  --output_file artifacts/qwen3-8b-knorm-stats.json
```

The script observes keys immediately before native KNorm, runs only the decoder
forward, and uses prefixes of the **same 64 calibration rows** for every length.
It truncates the extra next-token position returned by Nanoset. Text datasets
can contain shorter, padded sequences: those pads are excluded, and
`full_length_samples` records how many rows actually reach the requested length.
Use packed samples to measure full-length contexts. Reproducing the original
Table 1 values requires its original checkpoint, calibration samples and dtype;
this script does not assume those samples are bundled with the repository.

The JSON contains `mean`, `std`, `count` and `cv = std / mean` for **every layer
and KV head**, plus a `selected` record for the table. By default the selected
zero-based indices are `num_hidden_layers // 2` and `num_key_value_heads // 2`;
use `--layer` and `--head` to specify the exact head used in the original table.
The norm is `sqrt(mean(k.float() ** 2) + rms_norm_eps)`, matching the conversion
code. Standard deviations use the population definition (`correction=0`).
Statistics are accumulated as FP64 central moments, without retaining all keys.

To collect the same statistics for the **reconstructed MLA keys** during an
existing conversion, add `--knorm_stats_path artifacts/converted-knorm-stats.json`
to `gen/convert.py`. The optional diagnostic pools token counts and central
moments across ranks and excludes masked positions from both the mean and std.
It leaves the existing conversion's `Ek` scaling calculation unchanged and
also records it as `conversion_alpha`. For unpadded inputs the diagnostic mean
has the same definition, with FP64 accumulation. With masked positions it can
differ: the legacy `Ek` numerator includes masked positions, whereas this
diagnostic counts only tokens included by the conversion's attention mask.
The report records the actual input width as `context_length` and the CLI
setting separately as `requested_context_length` (Nanoset can return an extra
next-token position). Original and reconstructed key statistics are explicitly
labeled and should be reported separately.
