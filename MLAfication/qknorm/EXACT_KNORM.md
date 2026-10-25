# Exact-KNorm quality control

This control retains token-dependent KNorm after reconstructing the low-rank
key. It measures the quality cost of calibrated KNorm while retaining the same
Partial-RoPE selection, joint KV rank, initializer, and QKV-only distillation.
It is an intended quality upper-bound baseline, **not a guaranteed numerical
upper bound** on downstream scores.

For each token and KV head, the forward computes

```text
c = W_down x
k = concat(W_kr x, W_uk c)             # complete head, [RoPE, NoPE] order
k_hat = gamma_k * k / sqrt(mean(k^2) + eps)
k_rope = RoPE(k_hat[..., :rope_dim])
k_nope = k_hat[..., rope_dim:]
v = W_uv c
```

The RMS is computed in FP32 across the **entire** reconstructed head before
rotation. Gamma is permuted with the projection rows; neither gamma nor a
calibration-average RMS is absorbed into either key projection. QNorm is
unchanged. As with the existing reordered QNorm, gamma is stored per head;
the retained KNorm gains are trainable under the QKV-only policy. The original
model, with its original QKNorm, remains the frozen distillation teacher.

## Qwen3-8B, the nominal −68.75% setting

Qwen3-8B has 8 KV heads of dimension 128. Use rank 512 and top-k 8 (16 rotary
dimensions per KV head). The calibrated model's latent cache contains
`512 + 8 * 16 = 640` elements per token per layer, versus `2 * 8 * 128 = 2048`
for GQA: a 68.75% reduction.

**The exact control uses expanded GQA-shaped K/V caches, not the 640-element
latent cache.** Report it at the same rank/Partial-RoPE setting, and mark
standard latent-cache/MQA serving as unsupported. Its runtime cache is still
2048 elements per token per layer. The latent forward patcher explicitly
rejects exact checkpoints, including LongBench's latent FP8/quantized paths.

## Conversion and training

Run from `MLAfication/qknorm/`, in the environment described in the
[parent guide](../README.md#environment). Set `MHA2MLA_DATA_PATH` to the existing
preprocessed calibration mixture. To emit both controls with **identical**
frequency selection and SVD factors in one conversion:

```bash
torchrun --nproc_per_node 4 gen/convert.py \
  --model_name_or_path Qwen/Qwen3-8B-Base \
  --output_path artifacts/qwen3-8b-rank512-calibrated.pt \
  --exact_knorm_output_path artifacts/qwen3-8b-rank512-exact-knorm.pt \
  --lora_rank 512 --topk 8 --svd_method scaled_svdllm \
  --num_samples 128 --sequence_length 4096
```

To produce only the exact control, omit `--exact_knorm_output_path`, add
`--knorm_mode exact`, and set `--output_path` to the exact artifact path. This
skips calibration-average KNorm collection and absorption. Do not attach a
KNorm module to an old calibrated artifact: those weights already include
the gain and average-RMS scaling. Mode mismatches raise an error.

Copy [the training example](cfgs/qwen3-8b-rank512-exact-knorm.example.yml), fill
in the local model/data paths, and set `output_dir` to a **new** directory on
the experiment checkpoint volume. It follows the existing Qwen3-8B rank512
recipe: global batch 64, 4800 updates, sequence-length setting 4096, KL
temperature 1, LR 1e-4, 144 warmup steps and 480 final decay steps. Nanoset's
existing extra-token convention is unchanged; use the same loader and budget
for the calibrated reference. With 8 ranks, the example uses microbatch 2 and
accumulation 4, ZeRO-2 and activation checkpointing. If adjusting microbatch,
keep `world_size * microbatch * accumulation = 64`.

```bash
torchrun --nproc_per_node 8 src/mha2mla/train.py \
  --cfg_file /path/to/qwen3-8b-exact-knorm.yml
```

The existing trainer saves attention-only checkpoints with bundled MLA tensors.
These are model snapshots; its current `resume_from_checkpoint` handling loads
weights and starts a fresh optimizer/schedule, rather than resuming an exact
training state. Use a fresh run for this comparison. This implementation does
not submit a training job.

## Evaluation and checkpoints

Use the ordinary `eval/eval_v2.py` Commonsense and `eval/longbench.py` entry
points as described in [the implementation guide](README.md). For LongBench,
set `knorm_mode: exact`, point `mla_tensor_path` to the checkpoint's bundled
`mla_tensors.pt`, set `ckpt_path`, and use `cache_implementation: dynamic`.
The standard forward supports both prefill and cached decoding. Step-zero
evaluation also works with the exact initialization artifact and no checkpoint.

Exact tensor files contain `gamma_k`; legacy files without it retain calibrated
behavior. Checkpoint config records `mla_knorm_mode: exact`, and QKV-only saves
include `self_attn.k_norm.weight`. The existing full-checkpoint export command
also preserves this mode and the bundled tensors. Loading requires this
repository's patchers, as for other MLAfication checkpoints.

Compare Commonsense, LongBench, and attention-logit diagnostics using the same
evaluation environment, prompts and context limits as the calibrated control.
Use the same training data, budget and trainable-module policy for both runs;
any comparison to a historical checkpoint should state remaining differences.
The quality gap after recovery includes the effect of learning with exact
versus calibrated KNorm, not just a single-forward approximation error.

## Validation

`tests/test_exact_knorm.py` checks conversion against the legacy calibrated
formula, exact attention against an independent reconstruction in the original
head order, gradients, cached decoding, QKV-only updates, checkpoint round trips,
mode mismatch rejection, rejection of latent decoding, and training-recipe compatibility. Run it with:

```bash
python -m unittest discover -s tests -p 'test_exact_knorm.py' -v
```

Run the test command from the repository root, not this implementation directory.
