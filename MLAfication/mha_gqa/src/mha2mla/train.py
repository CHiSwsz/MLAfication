import os
import argparse
from dataclasses import asdict
from types import SimpleNamespace

import torch

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


logger = logging.get_logger(__name__)


def register_deepspeed_safe_globals():
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address

    torch.serialization.add_safe_globals(
        [LossScaler, ZeroStageEnum, fragment_address]
    )


class MHA2MLATrainer(Trainer):
    # Create optimizer/scheduler inside Trainer so DeepSpeed binds them to the real wrapped optimizer.
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_name = self.args.optim
            if "adam" in optimizer_name:
                self.optimizer = torch.optim.AdamW(
                    params=self.model.parameters(),
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

        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            logits_teacher = outputs_teacher.get("logits").detach().flatten(0, 1)

        log_p_student = F.log_softmax(logits_student / self.temperature, dim=-1)
        p_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)

        loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
        loss = loss_kd * (self.temperature**2)

        return (loss, outputs_student) if return_outputs else loss


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
        if train_args.resume_from_checkpoint is not None:
            raise ValueError(
                "resume_from_checkpoint is not supported when is_freeze_non_attn=true "
                "because checkpoints only save trainable attention parameters."
            )
        train_args.save_only_model = True

    # load tokenizer and model
    name = mha2mla_args.model_name_or_path
    model_args = AutoConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resume_from_checkpoint = train_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        register_deepspeed_safe_globals()
        mha_model = AutoModelForCausalLM.from_config(model_args)
    else:
        mha_model = AutoModelForCausalLM.from_pretrained(name)
    if dist.get_rank() == 0:
        print(mha_model)
    mla_model = patch_mlafication(mha_model, model_args, mha2mla_args)
    if dist.get_rank() == 0:
        print(mla_model)
    patch_mla_fwd_instance(mla_model)
    if resume_from_checkpoint is not None:
        load_checkpoint(mla_model, resume_from_checkpoint)
    student_model = mla_model
    teacher_model = AutoModelForCausalLM.from_pretrained(name)
    student_model.config.mha2mla = asdict(mha2mla_args)
    student_model.config.mha2mla["checkpoint_type"] = (
        "attn_only" if train_args.is_freeze_non_attn else "full"
    )
    student_model = student_model.to(device).to(torch.bfloat16)
    teacher_model = teacher_model.to(device).to(torch.bfloat16)
    if train_args.is_freeze_non_attn:
        freeze_non_attn_weights_v2(student_model)
        if train_args.gradient_checkpointing:
            enable_input_require_grads(student_model)

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
        )
        trainer.train(resume_from_checkpoint)
        trainer.log(asdict(mha2mla_args))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
