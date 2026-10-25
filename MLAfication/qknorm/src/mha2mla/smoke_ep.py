"""Eight-GPU numerical, AC, optimizer and checkpoint smoke test (no base weights)."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import deepspeed
import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from ep_distill_utils import engine_config, export_hf_checkpoint, expert_layout
from patching_mlafication import patch_mlafication
from patching_mla_fwd import patch_mla_fwd_instance


class NoTokenizer:
    def save_pretrained(self, destination):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError('Run smoke_ep.py with eight ranks')
    torch.manual_seed(123)
    config = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                           moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=32, num_experts=16, num_experts_per_tok=8,
                           norm_topk_prob=True, use_cache=False)
    config._attn_implementation = 'sdpa'
    tensors = {
        'q_proj': torch.randn(2, 128, 128) * .02,
        'k_r_proj': torch.randn(2, 16, 128) * .02,
        'k_up_weight': torch.randn(2, 48, 32) * .02,
        'v_up_weight': torch.randn(2, 64, 32) * .02,
        'kv_down_weight': torch.randn(2, 32, 128) * .02,
        'gamma_q': torch.ones(2, 128), 'rope_index': torch.arange(4).view(1, 4, 1).expand(2, 4, 2),
    }
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        torch.save(tensors, root / 'mla.pt')
    dist.barrier()

    def make_model():
        torch.manual_seed(19)
        model = Qwen3MoeForCausalLM(config)
        patch_mlafication(model, config, SimpleNamespace(mla_tensor_path=str(root / 'mla.pt')))
        patch_mla_fwd_instance(model)
        model.config.mha2mla = {'mla_tensor_path': str(root / 'mla.pt'), 'checkpoint_type': 'full'}
        return model.to(dtype=torch.bfloat16)

    reference = make_model().cuda().eval()
    tokens = (torch.arange(32, device='cuda').reshape(1, 32) + rank + 1) % 256
    with torch.no_grad():
        expected = reference(tokens).logits
    cfg = {'global_batch_size': 8, 'ep_size': 8, 'learning_rate': .01, 'max_grad_norm': 1.0}
    teacher_model = make_model().requires_grad_(False)
    teacher, _, _, _ = deepspeed.initialize(
        model=teacher_model, config=engine_config(cfg, 1, world, teacher=True))
    teacher.eval()
    with torch.no_grad():
        torch.testing.assert_close(teacher(tokens).logits, expected, rtol=.03, atol=.006)
    assert not any(p.requires_grad for p in teacher.module.parameters())
    model = make_model()
    layout = expert_layout(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    engine, _, _, _ = deepspeed.initialize(model=model, config=engine_config(cfg, 1, world))
    engine.eval()
    actual = engine(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.006)
    engine.train()
    router = engine.module.model.layers[0].mlp.router.gate.weight
    expert = engine.module.model.layers[0].mlp.experts.w1
    before_router, before_expert = router.detach().clone(), expert.detach().clone()
    reference(tokens, labels=tokens).loss.backward()
    reference_router_grad = reference.model.layers[0].mlp.gate.weight.grad.float()
    reference_expert_grad = reference.model.layers[0].mlp.experts.gate_up_proj.grad.float()
    for gradient in (reference_router_grad, reference_expert_grad):
        dist.all_reduce(gradient)
        gradient.div_(world)
    local_expert_grad = reference_expert_grad[rank * 2:(rank + 1) * 2, :128]
    for update in range(2):
        loss = engine(tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite smoke loss')
        engine.backward(loss)
        if update == 0:
            from deepspeed.utils import safe_get_full_grad
            # Catch incorrect EP/DP gradient scaling, which Adam's first update
            # can conceal because it is nearly invariant to a constant scale.
            for parameter, expected_grad in ((router, reference_router_grad), (expert, local_expert_grad)):
                actual_grad = safe_get_full_grad(parameter).float()
                relative_error = (actual_grad - expected_grad).norm() / expected_grad.norm().clamp_min(1e-12)
                assert relative_error < .08, f'EP gradient relative error: {relative_error.item()}'
        engine.step()
    with torch.no_grad():
        torch.testing.assert_close(teacher(tokens).logits, expected, rtol=.03, atol=.006)
    assert not torch.equal(router, before_router), 'Router did not update'
    assert not torch.equal(expert, before_expert), 'Experts did not update'
    engine.eval()
    with torch.no_grad():
        trained = engine(tokens).logits
    engine.save_checkpoint(str(root / 'native'), tag='smoke', client_state={'step': 2})
    loaded, state = engine.load_checkpoint(str(root / 'native'), tag='smoke')
    assert loaded and state['step'] == 2
    with torch.no_grad():
        torch.testing.assert_close(engine(tokens).logits, trained, rtol=0, atol=0)
    export_hf_checkpoint(engine, NoTokenizer(), root / 'portable', layout, root / 'mla.pt', {'step': 2})
    # Restore the portable export into an ordinary, non-EP patched Transformers model.
    from safetensors.torch import load_file
    restored = make_model().cuda().eval()
    index = json.loads((root / 'portable/model.safetensors.index.json').read_text())
    weights = {}
    for shard in sorted(set(index['weight_map'].values())):
        weights.update(load_file(str(root / 'portable' / shard)))
    restored.load_state_dict(weights, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens).logits, trained, rtol=.03, atol=.01)
    dist.barrier()
    if rank == 0:
        (root / 'SUCCESS').write_text('EP8/grouped GEMM/AC/router+expert updates/native+HF roundtrip passed\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
