import os
import argparse
from dataclasses import asdict
from types import SimpleNamespace

import torch
import gc
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    Trainer,
    get_scheduler,
)
from transformers.utils import logging
import torch.distributed as dist
import torch.nn.functional as F

from arguments import (
    MHA2MLAModelArguments,
    MHA2MLADataArguments,
    MHA2MLATrainingArguments,
)
from helpers import (
    load_dataset,
    freeze_non_attn_weights,
    freeze_non_attn_weights_v2,
    load_checkpoint,
    save_checkpoint,
    get_trainable_state_dict,
    bundle_mla_tensor,
    restore_mla_tensor_path,
    enable_input_require_grads,
    lr_scheduler_builder,
)
from patching_mlafication import patch_mlafication
from patching_mla_fwd import patch_mla_fwd_instance

from transformers import TrainerCallback
import time

logger = logging.get_logger(__name__)


def register_deepspeed_safe_globals():
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address

    torch.serialization.add_safe_globals(
        [LossScaler, ZeroStageEnum, fragment_address]
    )


class MHA2MLATrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_name = self.args.optim

            trainable_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]

            if dist.get_rank() == 0:
                num_trainable = sum(p.numel() for p in trainable_params)
                print(f"[optimizer] num_trainable_params = {num_trainable}", flush=True)

            if "adam" in optimizer_name:
                self.optimizer = torch.optim.AdamW(
                    params=trainable_params,
                    lr=self.args.learning_rate,
                    betas=(
                        self.args.adam_beta1,
                        self.args.adam_beta2,
                    ),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                    fused=bool(self.args.optim == "adamw_torch_fused"),
                )
            else:
                raise ValueError(f"Unknown optimizer factory {optimizer_name}")
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            if self.args.use_constant_with_warmup_decay_scheduler:
                self.lr_scheduler = lr_scheduler_builder(
                    optimizer=optimizer,
                    lr_scheduler_args=SimpleNamespace(**self.args.lr_scheduler_kwargs),
                    total_training_steps=num_training_steps,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _save(self, output_dir=None, state_dict=None):
        if self.args.is_freeze_non_attn:
            state_dict = get_trainable_state_dict(self.model, state_dict=state_dict)
        output_dir = output_dir or self.args.output_dir
        bundle_state = bundle_mla_tensor(self.model, output_dir)
        try:
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        finally:
            restore_mla_tensor_path(bundle_state)


class DistillationTrainer(MHA2MLATrainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.model_accepts_loss_kwargs = False

        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        logits_student = outputs_student.get("logits").flatten(0, 1)

        with torch.inference_mode():
            outputs_teacher = self.teacher(**inputs)
            logits_teacher = outputs_teacher.get("logits").flatten(0, 1)

        log_p_student = F.log_softmax(logits_student / self.temperature, dim=-1)
        p_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)

        loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        loss = loss_kd * (self.temperature**2)

        return (loss, outputs_student) if return_outputs else loss

class GPUMemoryCallback(TrainerCallback):
    def __init__(self, log_every_steps=10, log_rank0_only=False):
        self.log_every_steps = log_every_steps
        self.log_rank0_only = log_rank0_only

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every_steps != 0:
            return

        if not torch.cuda.is_available():
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.log_rank0_only and rank != 0:
            return

        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

        print(
            f"[rank {rank}] step={state.global_step} "
            f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
            f"max_alloc={max_alloc:.2f}GB max_reserved={max_reserved:.2f}GB",
            flush=True,
        )

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def get_parent_module_and_attr(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def module_has_any_parameter(module: nn.Module) -> bool:
    return any(True for _ in module.parameters(recurse=False))


def module_has_trainable_parameter(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters(recurse=False))


def module_has_any_trainable_descendant(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())

def get_module_by_name(root: nn.Module, name: str):
    if name == "":
        return root
    module = root
    for part in name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def find_fully_frozen_leaf_modules(student_model: nn.Module):
    """
    返回 student 中适合被整体替换为 teacher 对应模块的模块名列表。

    条件：
    1. 这个模块自身直接持有参数（recurse=False）
    2. 这些直接参数全部是 requires_grad=False
    3. 子模块里不存在 trainable 参数
    这样可以避免替掉包含 trainable qkv/attn 的父模块。
    """
    to_share = []

    for name, module in student_model.named_modules():
        if name == "":
            continue

        # 这个模块自己没有直接参数，不替
        direct_params = list(module.parameters(recurse=False))
        if len(direct_params) == 0:
            continue

        # 这个模块直接参数里只要有 trainable，就不能替
        if any(p.requires_grad for p in direct_params):
            continue

        # 子树里如果还有 trainable 参数，也不能整体替
        has_trainable_child = False
        for child_name, child in module.named_children():
            if any(p.requires_grad for p in child.parameters()):
                has_trainable_child = True
                break

        if has_trainable_child:
            continue

        to_share.append(name)

    return to_share


def share_frozen_leaf_modules(student_model: nn.Module, teacher_model: nn.Module):
    """
    把 student 中所有完全冻结的叶子模块，替换成 teacher 中同名模块对象。
    """
    module_names_to_share = find_fully_frozen_leaf_modules(student_model)

    # print("module_names_to_share:")
    # print(module_names_to_share)

    if dist.get_rank() == 0:
        print(f"[share] num modules to share = {len(module_names_to_share)}", flush=True)

    replaced = 0
    replaced_params = 0
    old_modules = []

    for name in module_names_to_share:
        s_module = get_module_by_name(student_model, name)
        t_module = get_module_by_name(teacher_model, name)

        # 再做一次安全检查
        s_params = list(s_module.parameters())
        t_params = list(t_module.parameters())

        if len(s_params) != len(t_params):
            raise ValueError(f"Parameter count mismatch at module {name}")

        for sp, tp in zip(s_params, t_params):
            if sp.shape != tp.shape:
                raise ValueError(f"Shape mismatch at module {name}")
            if sp.dtype != tp.dtype:
                raise ValueError(f"Dtype mismatch at module {name}")

        parent, attr = get_parent_module_and_attr(student_model, name)
        old_modules.append(s_module)  # 暂存旧对象，替换完成后统一释放引用
        setattr(parent, attr, t_module)

        replaced += 1
        replaced_params += sum(p.numel() for p in t_module.parameters())

        if dist.get_rank() == 0 and replaced <= 10:
            print(f"[share] replaced module: {name}", flush=True)

    # 释放 Python 引用
    del old_modules[:]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        print(
            f"[share] replaced_modules={replaced}, replaced_params={replaced_params}",
            flush=True,
        )

    return student_model, teacher_model


def verify_shared_modules(student_model: nn.Module, teacher_model: nn.Module, max_print=20):
    names = find_fully_frozen_leaf_modules(student_model)
    checked = 0

    for name in names[:max_print]:
        try:
            s_mod = get_module_by_name(student_model, name)
            t_mod = get_module_by_name(teacher_model, name)
            same_obj = id(s_mod) == id(t_mod)
            if dist.get_rank() == 0:
                print(f"[verify shared] {name}: same_module={same_obj}", flush=True)
            checked += 1
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[verify shared] {name}: failed with {e}", flush=True)

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    # load arguments
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True)
    cfg = cfg_parser.parse_args()
    hf_parser = HfArgumentParser(
        (MHA2MLAModelArguments, MHA2MLADataArguments, MHA2MLATrainingArguments)
    )
    mha2mla_args, data_args, train_args = hf_parser.parse_yaml_file(cfg.cfg_file)
    if train_args.is_freeze_non_attn:
        train_args.save_only_model = True

    # load tokenizer and model
    name = mha2mla_args.model_name_or_path
    model_args = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. load teacher
    teacher_model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # 2. load student
    mha_model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    mla_model = patch_mlafication(mha_model, model_args, mha2mla_args)
    patch_mla_fwd_instance(mla_model)

    resume_from_checkpoint = train_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        register_deepspeed_safe_globals()
        load_checkpoint(mla_model, resume_from_checkpoint, strict=False)

    student_model = mla_model
    student_model.config.mha2mla = asdict(mha2mla_args)
    student_model.config.mha2mla["checkpoint_type"] = (
        "attn_only" if train_args.is_freeze_non_attn else "full"
    )

    # 3. freeze teacher
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # 4. freeze student, keep only qkv/attn trainable
    if train_args.is_freeze_non_attn:
        freeze_non_attn_weights_v2(student_model)

    # 5. share frozen leaf modules
    student_model, teacher_model = share_frozen_leaf_modules(student_model, teacher_model)
    verify_shared_modules(student_model, teacher_model, max_print=20)

    if dist.get_rank() == 0:
        s_total, s_train, s_frozen = count_params(student_model)
        t_total, t_train, t_frozen = count_params(teacher_model)
        print(
            f"[param count] student total={s_total}, trainable={s_train}, frozen={s_frozen}",
            flush=True,
        )
        print(
            f"[param count] teacher total={t_total}, trainable={t_train}, frozen={t_frozen}",
            flush=True,
        )

    # 6. move to gpu
    if train_args.gradient_checkpointing:
        enable_input_require_grads(student_model)
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    train_dataset = load_dataset(data_args, train_args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if train_args.max_steps == 0:
        save_checkpoint(
            student_model,
            train_args.output_dir,
            tokenizer,
            trainable_only=train_args.is_freeze_non_attn,
        )
    else:
        trainer = DistillationTrainer(
            model=student_model,
            teacher_model=teacher_model,
            processing_class=tokenizer,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            temperature=train_args.kl_temperature,
            callbacks=[GPUMemoryCallback(log_every_steps=20, log_rank0_only=False)],
        )
        # trainer.train(resume_from_checkpoint)
        trainer.train()
        trainer.log(asdict(mha2mla_args))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
