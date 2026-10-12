import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset as load_hf_dataset
from datasets import load_from_disk


def _default_dataset_folders():
    root = os.getenv("MHA2MLA_DATA_PATH")
    if not root:
        return []
    return [
        f"{root.rstrip('/')}/{name}/"
        for name in (
            "fineweb-edu-dedup",
            "cosmopedia-v2",
            "python-edu",
            "open-web-math",
            "stackoverflow",
        )
    ]


def _load_hf_calibration_dataset(tokenizer, dataset_args, dataset_name):
    split = os.getenv("MHA2MLA_HF_SPLIT", "train")
    text_column = os.getenv("MHA2MLA_TEXT_COLUMN", "text")
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
    else:
        dataset = load_hf_dataset(dataset_name, split=split)
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column {text_column!r} is absent; available columns: "
            f"{dataset.column_names}"
        )

    max_examples = dataset_args.max_batches * dataset_args.per_gpu_batchsize
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=dataset_args.sequence_length,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset


def load_dataset(tokenizer, dataset_args):
    """Load dataset from configuration."""
    hf_dataset = os.getenv("MHA2MLA_HF_DATASET")
    if hf_dataset:
        return _load_hf_calibration_dataset(tokenizer, dataset_args, hf_dataset)
    if not dataset_args.dataset_folders:
        raise EnvironmentError(
            "Set MHA2MLA_DATA_PATH for the preprocessed Nanoset mixture or "
            "MHA2MLA_HF_DATASET for a Hugging Face/local text dataset."
        )
    from nanotron.data.nanoset import Nanoset

    origin_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = dataset_args.sequence_length
    token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # multi GPUs
    global_batch_size = dataset_args.per_gpu_batchsize * world_size
    tokenizer.model_max_length = origin_model_max_length
    return Nanoset(
        dataset_folders=dataset_args.dataset_folders,
        sequence_length=dataset_args.sequence_length,
        dataset_weights=dataset_args.dataset_weights,
        token_size=token_size,
        train_split_num_samples=global_batch_size * dataset_args.max_batches,
    )


@dataclass
class MHA2MLADataArguments:
    dataset_folders: list[str] = field(default_factory=_default_dataset_folders)
    dataset_weights: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.08, 0.06, 0.01])
    sequence_length: int = field(default=4096)
    per_gpu_batchsize: int = field(default=1)
    max_batches: int = field(default=128)
