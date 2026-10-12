#!/usr/bin/env python3
"""Small, reproducible Hugging Face decode benchmark for baseline and MLA models."""

import argparse
import csv
import inspect
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def comma_separated_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--mla_tensor_path")
    parser.add_argument("--implementation", choices=("mha_gqa", "qknorm"))
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--batch_sizes", type=comma_separated_ints, default=[1, 4])
    parser.add_argument("--input_lengths", type=comma_separated_ints, default=[512, 2048])
    parser.add_argument("--new_tokens", type=int, default=128)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--cache_implementation",
        choices=("dynamic", "quantized", "fp8"),
        default="dynamic",
    )
    parser.add_argument("--backend", default="quanto")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--output", default="decode_benchmark.csv")
    args = parser.parse_args()
    if bool(args.implementation) != bool(args.mla_tensor_path):
        parser.error("--implementation and --mla_tensor_path must be provided together")
    if args.checkpoint and not args.mla_tensor_path:
        parser.error("--checkpoint requires an MLA implementation and tensor file")
    if args.cache_implementation == "fp8" and args.implementation != "qknorm":
        parser.error("the custom FP8 latent cache is available for qknorm MLA only")
    return args


def load_model(args, device):
    dtype = DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.implementation:
        source_dir = Path(__file__).resolve().parent / args.implementation / "src" / "mha2mla"
        sys.path.insert(0, str(source_dir))
        from helpers import load_checkpoint
        from patching_mlafication import patch_mlafication

        if args.implementation == "qknorm" and args.cache_implementation in {
            "quantized",
            "fp8",
        }:
            from patching_mla_latent_fwd import patch_mla_fwd_instance
        else:
            from patching_mla_fwd import patch_mla_fwd_instance

        mla_args = types.SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            mla_tensor_path=args.mla_tensor_path,
            output_mla_dir=None,
            is_baseline=False,
        )
        model = patch_mlafication(model, model.config, mla_args)
        patch_mla_fwd_instance(model)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, strict=False)

    return model.to(device=device, dtype=dtype).eval(), tokenizer


def cache_kwargs(args, model):
    if args.cache_implementation == "dynamic":
        return {"cache_implementation": "dynamic"}
    if args.cache_implementation == "quantized":
        from transformers.cache_utils import QuantizedCache

        accepted = set(inspect.signature(QuantizedCache.__init__).parameters)
        cache_config = {
            "backend": args.backend,
            "nbits": args.nbits,
            "residual_length": args.residual_length,
        }
        optional = {"compute_dtype": model.dtype, "device": model.device}
        cache_config.update(
            {name: value for name, value in optional.items() if name in accepted}
        )
        return {
            "cache_implementation": "quantized",
            "cache_config": cache_config,
        }

    from fp8_cache import QuantoFP8Cache

    return {
        "past_key_values": QuantoFP8Cache(
            config=model.config,
            residual_length=args.residual_length,
        )
    }


@torch.inference_mode()
def run_once(model, input_ids, attention_mask, args):
    kwargs = cache_kwargs(args, model)
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        min_new_tokens=args.new_tokens,
        max_new_tokens=args.new_tokens,
        pad_token_id=model.generation_config.pad_token_id,
        **kwargs,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_decode.py requires a CUDA GPU")
    device = torch.device("cuda:0")
    model, tokenizer = load_model(args, device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    rows = []

    for batch_size in args.batch_sizes:
        for input_length in args.input_lengths:
            input_ids = torch.randint(
                0,
                model.config.vocab_size,
                (batch_size, input_length),
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            for _ in range(args.warmup_runs):
                run_once(model, input_ids, attention_mask, args)

            elapsed = []
            peak_memory = []
            for _ in range(args.runs):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                elapsed.append(run_once(model, input_ids, attention_mask, args))
                peak_memory.append(torch.cuda.max_memory_allocated(device))

            mean_seconds = sum(elapsed) / len(elapsed)
            row = {
                "implementation": args.implementation or "baseline",
                "cache": args.cache_implementation,
                "batch_size": batch_size,
                "input_length": input_length,
                "new_tokens": args.new_tokens,
                "seconds": mean_seconds,
                "tokens_per_second": batch_size * args.new_tokens / mean_seconds,
                "peak_memory_mib": sum(peak_memory) / len(peak_memory) / 1024**2,
            }
            rows.append(row)
            print(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
