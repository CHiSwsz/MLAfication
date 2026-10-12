#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
import types

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helpers import load_checkpoint
from patching_mlafication import patch_mlafication


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a full QKNorm-compatible MLA checkpoint."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--mla_tensor_path", required=True)
    parser.add_argument("--adapter_checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP),
    )
    return parser.parse_args()


def init_single_process_group():
    if dist.is_initialized():
        return None
    temp_dir = tempfile.TemporaryDirectory()
    init_file = os.path.join(temp_dir.name, "dist_init")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    return temp_dir


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dist_tmpdir = init_single_process_group()
    try:
        dtype = DTYPE_MAP[args.dtype]
        model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        mla_args = types.SimpleNamespace(
            model_name_or_path=args.model_name_or_path,
            mla_tensor_path=args.mla_tensor_path,
            output_mla_dir=args.output_dir,
            is_baseline=False,
        )
        model = patch_mlafication(model, model_config, mla_args)
        if args.adapter_checkpoint:
            load_checkpoint(model, args.adapter_checkpoint, strict=False)

        bundled_tensor_name = "mla_tensors.pt"
        bundled_tensor_path = os.path.join(args.output_dir, bundled_tensor_name)
        if os.path.abspath(args.mla_tensor_path) != os.path.abspath(bundled_tensor_path):
            shutil.copy2(args.mla_tensor_path, bundled_tensor_path)
        model.config.mha2mla = vars(mla_args).copy()
        model.config.mha2mla["mla_tensor_path"] = bundled_tensor_name
        model.config.mha2mla["checkpoint_type"] = "full"

        state_dict = model.state_dict()
        if model.config.tie_word_embeddings:
            state_dict["lm_head.weight"] = state_dict[
                "model.embed_tokens.weight"
            ].clone()
        model.save_pretrained(args.output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if dist_tmpdir is not None:
            dist_tmpdir.cleanup()


if __name__ == "__main__":
    main()
