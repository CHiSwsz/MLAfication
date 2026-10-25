"""Collect Table 1 token-wise key-norm mean/std across context lengths."""

import argparse
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from knorm_stats import TokenNormStats, context_record, key_rms, write_report


def collate_tokens(rows):
    fields = ["input_ids"]
    if "attention_mask" in rows[0]:
        fields.append("attention_mask")
    return {field: torch.stack([torch.as_tensor(row[field], dtype=torch.long) for row in rows])
            for field in fields}


def collect_context(model, dataloader, sequence_length, device):
    """Observe native keys before KNorm without recomputing or modifying attention."""
    if model.config.model_type not in ("qwen3", "qwen3_moe") or getattr(model.config, "mha2mla", None):
        raise ValueError("Expected an unconverted Qwen3 or Qwen3-MoE model")
    stats = [TokenNormStats(model.config.num_key_value_heads, device)
             for _ in model.model.layers]
    token_mask = None
    handles = []
    samples = full_length_samples = 0

    def hook(layer_index):
        def observe(module, args):
            stats[layer_index].update(key_rms(args[0], module.variance_epsilon), token_mask)
        return observe

    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.k_norm.register_forward_pre_hook(hook(i)))
        with torch.inference_mode():
            for batch in dataloader:
                ids = batch["input_ids"][:, :sequence_length].to(device)
                if ids.shape[1] != sequence_length:
                    raise ValueError("Dataset rows must be padded or packed to the requested context length")
                mask = batch.get("attention_mask")
                # Packed Nanoset rows have no padding. EOS is a real token even
                # when a tokenizer uses the same ID for EOS and padding.
                token_mask = torch.ones_like(ids) if mask is None else mask[:, :sequence_length].to(device)
                samples += ids.shape[0]
                full_length_samples += int((token_mask.sum(-1) == sequence_length).sum().item())
                # Skip the LM head: collecting key norms needs no vocabulary logits.
                model.model(input_ids=ids, attention_mask=token_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return stats, samples, full_length_samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layer", type=int, help="Selected layer for Table 1 (zero-based; default: middle)")
    parser.add_argument("--head", type=int, help="Selected KV head for Table 1 (zero-based; default: middle)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    if min(args.context_lengths + [args.num_samples, args.batch_size]) <= 0:
        parser.error("Context lengths, num_samples and batch_size must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This diagnostic runs on one device; launch with python, not multi-rank torchrun")
    return args


def main():
    args = parse_args()
    # Keep the statistics helpers importable in CPU tests without dataset dependencies.
    from mha2mla_datasets import MHA2MLADataArguments, load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    if config.model_type not in ("qwen3", "qwen3_moe") or getattr(config, "mha2mla", None):
        raise ValueError("Use the original, unconverted Qwen3/Qwen3-MoE checkpoint")
    layer = config.num_hidden_layers // 2 if args.layer is None else args.layer
    head = config.num_key_value_heads // 2 if args.head is None else args.head
    if not 0 <= layer < config.num_hidden_layers or not 0 <= head < config.num_key_value_heads:
        raise ValueError("Selected layer/head is outside this model's zero-based index range")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_args = MHA2MLADataArguments(
        sequence_length=max(args.context_lengths), per_gpu_batchsize=args.batch_size,
        max_batches=math.ceil(args.num_samples / args.batch_size),
    )
    # Nanoset's logger needs a process group even for single-device diagnostics.
    owns_group = not os.getenv("MHA2MLA_HF_DATASET") and not dist.is_initialized()
    try:
        if owns_group:
            dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        dataset = load_dataset(tokenizer, data_args)
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()
    if len(dataset) < args.num_samples:
        raise ValueError(f"Requested {args.num_samples} samples; dataset only has {len(dataset)}")
    if os.getenv("MHA2MLA_HF_DATASET"):
        # These are numeric tokens only. Collate plain Python values without
        # datasets' tensor formatters importing optional video-decoding packages.
        dataset = dataset.with_format(None, columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(Subset(dataset, range(args.num_samples)), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    ).to(device=args.device).eval()
    contexts = []
    print("context_length\tlayer\thead\tvalid_tokens\tmean(alpha)\ttoken_std\tCV")
    for length in args.context_lengths:
        stats, samples, full = collect_context(model, dataloader, length, args.device)
        record = context_record(stats, length, args.layer, args.head)
        record.update(num_samples=samples, full_length_samples=full)
        contexts.append(record)
        selected = record["selected"]
        cv = "undefined" if selected["cv"] is None else f"{selected['cv']:.6f}"
        print(f"{length}\t{selected['layer']}\t{selected['head']}\t{selected['count']}\t"
              f"{selected['mean']:.6f}\t{selected['std']:.6f}\t{cv}", flush=True)
    write_report(
        args.output_file, contexts, model_name_or_path=args.model_name_or_path,
        key_source="original_pre_knorm", rms_norm_eps=model.config.rms_norm_eps,
        dtype=args.dtype, num_samples=args.num_samples,
        sampling="prefixes of the same calibration rows, loaded once at the maximum context length",
        dataset={"hf_dataset": os.getenv("MHA2MLA_HF_DATASET"),
                 "hf_split": os.getenv("MHA2MLA_HF_SPLIT", "train"),
                 "text_column": os.getenv("MHA2MLA_TEXT_COLUMN", "text"),
                 "nanoset_folders": data_args.dataset_folders,
                 "nanoset_weights": data_args.dataset_weights},
    )


if __name__ == "__main__":
    main()
