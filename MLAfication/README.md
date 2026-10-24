# MLAfication: Efficiently Enabling High-Fidelity MLA for Modern LLMs

**Zile Wang, Zhide Xie, and Tao Ji**

[← All MLAfication projects](../README.md)

**Paper:** [arXiv:2610.17202](https://arxiv.org/abs/2610.17202). Under review. Citation metadata is available in [CITATIONS.bib](../CITATIONS.bib).

Run the commands below from this `MLAfication/` project directory inside the unified repository. Use a separate environment from the ACL 2025 and AAAI 2026 implementations.

This repository contains the unified code used by **MLAfication**:

- `mha_gqa/`: pretrained MHA/GQA models without QKNorm (Llama, Qwen2/2.5,
  and SmolLM-style attention).
- `qknorm/`: dense and MoE Qwen3 models with QKNorm.

The conversion entry points implement contribution-aware Partial-RoPE and use
activation-aware `scaled_svdllm` (SVDLLM V2) by default. Alternative SVD modes
remain available from the command line for ablations.

## Checkpoints

The MLAfication checkpoints are available in the
[MLAfication collection on Hugging Face](https://huggingface.co/collections/Gnaudzzx/mlafication-6a91b14d4fb0e7dec2e1fc10).

## Environment

Use Python 3.11 and install a CUDA-enabled PyTorch build matching the host
driver. The paper's original serving environment used PyTorch 2.11 with CUDA 12.8;
the requirements now select patched versions. See [the dependency policy](../security/dependency-policy.json)
for the validated Linux resolution and security-check scope.

```bash
python -m pip install "uv>=0.13.5"
uv pip install --overrides ../security/lighteval-overrides.txt -r requirements.txt
```

The patched runtime targets PyTorch 2.13, Transformers 5.10.4, NLTK 3.10.4,
and Lighteval 0.6.2. Lighteval 0.6.2 is
installed using explicit security overrides because its historical package
metadata pins vulnerable NLTK/Protobuf releases and caps PyTorch below 2.5.
`requirements.txt` supplies the patched runtime dependencies. Hugging Face
tokens for gated models should be provided through `HF_TOKEN`; credentials
are never read from source files.

## Calibration data

Conversion consumes the MHA2MLA data mixture through Nanotron `Nanoset` files.
Set `MHA2MLA_DATA_PATH` to a directory containing these five preprocessed
subdirectories:

```text
fineweb-edu-dedup/
cosmopedia-v2/
python-edu/
open-web-math/
stackoverflow/
```

Recovery training can instead use either the same Nanoset folders or a Hugging
Face dataset supplied with `--hf_dataset_name_or_path`.

For a lightweight conversion check without preprocessing Nanoset files, set
`MHA2MLA_HF_DATASET` to a Hugging Face dataset ID or a directory created by
`datasets.save_to_disk`. The default split and text column are `train` and
`text`; override them with `MHA2MLA_HF_SPLIT` and `MHA2MLA_TEXT_COLUMN`.

LongBench v1 and v2 are downloaded directly from `THUDM/LongBench` and
`THUDM/LongBench-v2`. Set `LONGBENCH_DATASET` or `LONGBENCH_V2_DATASET` to a
local dataset path for offline evaluation.

## 1. Generate MLA tensors

For an MHA/GQA model:

```bash
torchrun --nproc_per_node 4 mha_gqa/gen/convert.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --output_path artifacts/qwen2.5-7b-rank256.pt \
  --lora_rank 256 --topk 8 --num_samples 128 --sequence_length 4096
```

For either dense Qwen3 or Qwen3-MoE, use the same QKNorm entry point; the model
type is detected from its config:

```bash
torchrun --nproc_per_node 4 qknorm/gen/convert.py \
  --model_name_or_path Qwen/Qwen3-8B-Base \
  --output_path artifacts/qwen3-8b-rank512.pt \
  --lora_rank 512 --topk 8 --num_samples 128 --sequence_length 4096
```

Use `--svd_method original_svd` only for the vanilla-SVD ablation. Distributed
calibration reduces and synchronizes all statistics before rank 0 writes a
single portable CPU tensor file.

## 2. Recovery training

Both training entry points preserve the original configuration semantics and
require `--cfg_file`. Create the YAML outside this repository using the fields
defined in each implementation's `src/mha2mla/arguments.py`, then run:

```bash
torchrun --nproc_per_node 4 mha_gqa/src/mha2mla/train.py \
  --cfg_file /path/to/train.yml
```

Replace the entry point with `qknorm/src/mha2mla/train.py` for Qwen3. The paper
reports the exact rank, token budget, learning rate, and sequence length for
each model in its appendix; compute `max_steps` from the target token budget,
global batch size, and sequence length.

QKV-only checkpoints are deltas. Each checkpoint bundles its MLA tensor file,
and evaluation loads the frozen weights from `model_name_or_path`
automatically. To create a full portable MHA/GQA checkpoint:

```bash
python mha_gqa/src/mha2mla/export_mla_checkpoint.py \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --mla_tensor_path artifacts/qwen2.5-7b-rank256.pt \
  --adapter_checkpoint outputs/qwen2.5-7b-rank256/checkpoint-500 \
  --output_dir exported/qwen2.5-7b-rank256
```

Use `qknorm/src/mha2mla/export_mla_checkpoint.py` with the same arguments for
dense or MoE Qwen3.

## 3. Evaluation

`eval/eval_v2.py` is the Lighteval-compatible short-context entry point. For
example, to run HellaSwag through the QKNorm checkpoint loader:

```bash
accelerate launch qknorm/eval/eval_v2.py accelerate \
  --model_args "pretrained=/path/to/checkpoint,dtype=bfloat16,max_length=4096" \
  --tasks "leaderboard|hellaswag|0|0" \
  --override_batch_size 8 --output_dir /path/to/results
```

LongBench also preserves the original configuration semantics and requires
`--cfg_file`:

```bash
python qknorm/eval/longbench.py --cfg_file /path/to/longbench.yml
```

In the YAML, set `cache_implementation: quantized`, `backend: quanto`, and
`nbits: 4` for a quantized LongBench run, or use `cache_implementation: fp8`
and `nbits: 8` for the custom latent FP8 cache. Set
`lb_max_samples_per_subset` only for a bounded smoke test; leaving it unset
evaluates every example in all 21 LongBench v1 subsets. The default
`q_group_size` is 64; choose a divisor of the MLA latent and RoPE dimensions
(for example, 16 for a rank-16 checkpoint).
If `nvcc` is unavailable, the evaluator uses Quanto's portable PyTorch CUDA
fallback instead of its optional compiled unpack kernel.

`benchmark_decode.py` provides a configuration-free latency, throughput, and
peak-memory check for baseline and MLA checkpoints:

```bash
python benchmark_decode.py \
  --model_name_or_path Qwen/Qwen3-8B-Base \
  --implementation qknorm \
  --mla_tensor_path artifacts/qwen3-8b-rank512.pt \
  --checkpoint outputs/qwen3-8b-rank512/checkpoint-500 \
  --cache_implementation fp8 --batch_sizes 1,4 --input_lengths 512,4096
```

The benchmark is a Hugging Face correctness/performance check; serving-engine
numbers should additionally record the engine version, tensor parallelism,
context length, and concurrency used for the run.

## Provenance and scope

The standard-attention implementation was consolidated from `real_v2` at
`d3a3811df47d1e67c3b98b57651e862b2d8f4d2c`. The Qwen3 implementation was
consolidated from `feat/qwen3` at
`fd9ab67ba48faf5fe46540ce673cd08f6e646057`.

Model weights, preprocessed datasets, generated tensors, checkpoints, logs,
and evaluation outputs are intentionally not committed.

## License

Except where otherwise noted, the original source code in this repository is
licensed under the Apache License, Version 2.0. See `LICENSE` for the complete
terms. Third-party and adapted code retains its original copyright and license;
see `THIRD_PARTY_NOTICES.md` for attribution and applicable terms.

The repository license does not grant rights to model weights, datasets, the
paper manuscript, or third-party figures and assets. Those materials remain
subject to their respective terms.
