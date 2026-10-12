import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from tqdm import tqdm
import torch.distributed as dist

from patch_func import(
    create_custom_apply_rotary_pos_emb,
    create_custom_apply_rotary_pos_emb_mha2mla,
    create_custom_sepearte_rope_nope_mha2mla,
    apply_U,
    seperate_nope_rope
)
from svdllm import (
    original_svd,
    svdllm,
    scaled_svdllm,
    scaled_UD_svdllm,
    scaled_KV_svdllm
)
from mha2mla_datasets import MHA2MLADataArguments, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MLA initialization tensors for MHA/GQA models."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--lora_rank", type=int, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument(
        "--svd_method",
        choices=(
            "original_svd",
            "svdllm",
            "scaled_svdllm",
            "scaled_UD_svdllm",
            "scaled_KV_svdllm",
        ),
        default="scaled_svdllm",
        help="Low-rank initializer. The paper uses scaled_svdllm (SVDLLM V2).",
    )
    return parser.parse_args()


cli_args = parse_args()
num_samples = cli_args.num_samples
topk = cli_args.topk
lora_rank = cli_args.lora_rank

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

tokenizer = AutoTokenizer.from_pretrained(cli_args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(cli_args.model_name_or_path).to(device).eval()
base_model = getattr(model, model.base_model_prefix)
model.config.use_cache = False
base_model.config.use_cache = False
ROTARY_MODELING = modeling_qwen2 if model.config.model_type == "qwen2" else modeling_llama
data_args = MHA2MLADataArguments()
data_args.sequence_length = cli_args.sequence_length
ds = load_dataset(tokenizer, data_args)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
sampler = torch.utils.data.distributed.DistributedSampler(ds)
dataloader = DataLoader(ds, batch_size=data_args.per_gpu_batchsize, sampler=sampler)

def hook_uniform_baseline(module, args, kwargs, output):
    origin_attn_out = output[0].flatten(0, 1).detach()
    origin_apply_rotary_pos_emb = ROTARY_MODELING.apply_rotary_pos_emb
    kv = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
    ROTARY_MODELING.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb(
        partial_rope_uniform_step=n_freqs // topk
    )

    with torch.no_grad():  # avoid save calculation graph(oom)
        attn_output, _ = module.forward(
            hidden_states=kwargs["hidden_states"],
            attention_mask=kwargs["attention_mask"],
            position_ids=kwargs["position_ids"],
            position_embeddings=kwargs["position_embeddings"],
        )
    attn_output = attn_output.flatten(0, 1).detach()
    attn_mse_loss = torch.nn.MSELoss(reduction="sum")
    if kwargs["attention_mask"] is not None:
        mask = kwargs["attention_mask"][:, 0, -1, :].flatten()
        error = attn_mse_loss(attn_output[mask], origin_attn_out[mask])
    else:
        # if all attention masks == 1, kwargs["attention_mask"] is None
        error = attn_mse_loss(attn_output, origin_attn_out)
    if dist.is_available() and dist.is_initialized():
        # sum error across all ranks(GPUs)
        dist.all_reduce(error, op=dist.ReduceOp.SUM)
    if int(os.environ.get("RANK", 0)) == 0:
        # only rank 0 prints and records
        ERROR_SUM_uniform[module.layer_idx] += error.item()
    ROTARY_MODELING.apply_rotary_pos_emb = origin_apply_rotary_pos_emb

def hook_permute_freqs(module, args, kwargs, output):
    hidden_states = kwargs["hidden_states"]
    q = module.q_proj(hidden_states)  # [B, L, nq*d]
    k = module.k_proj(hidden_states)  # [B, L, nkv*d]
    B, L = k.shape[:2]
    device = hidden_states.device
    q = q.view(B, L, -1, module.head_dim)       # [B, L, nq, d]
    k = k.view(B, L, -1, module.head_dim)       # [B, L, nkv, d]
    n_freqs = module.head_dim // 2
    nq = q.size(2)
    nkv = k.size(2)
    num_key_value_groups = nq // nkv
    cos, sin = kwargs["position_embeddings"]    # [L, 2*n_freqs]
    cos = cos.view(1, L, 1, 2, n_freqs).transpose(-1, -2).to(device=device, dtype=q.dtype)
    sin = sin.view(1, L, 1, 2, n_freqs).transpose(-1, -2).to(device=device, dtype=q.dtype)
    q = q.reshape(B, L, nq, 2, n_freqs).transpose(-1, -2).contiguous()   # [B, L, nq, n_freqs, 2]
    k = k.reshape(B, L, nkv, 2, n_freqs).transpose(-1, -2).contiguous()  # [B, L, nkv, n_freqs, 2]

    freq_chunk_size = min(100, n_freqs)
    mask = torch.tril(torch.ones(L, L, device=device, dtype=q.dtype), diagonal=0)
    diff = torch.zeros(n_freqs, nkv, device=device, dtype=torch.float32)
    for start in range(0, n_freqs, freq_chunk_size):
        end = min(start + freq_chunk_size, n_freqs)
        cos_chunk_q = cos[..., start:end, :]
        sin_chunk_q = sin[..., start:end, :]
        cos_chunk_k = cos_chunk_q.squeeze(2)
        sin_chunk_k = sin_chunk_q.squeeze(2)

        for kv_idx in range(nkv):
            head_start = kv_idx * num_key_value_groups
            head_end = head_start + num_key_value_groups

            q_chunk = q[:, :, head_start:head_end, start:end, :]
            k_chunk = k[:, :, kv_idx, start:end, :]

            q_pe_chunk = (q_chunk * cos_chunk_q) + (ROTARY_MODELING.rotate_half(q_chunk) * sin_chunk_q)
            k_pe_chunk = (k_chunk * cos_chunk_k) + (ROTARY_MODELING.rotate_half(k_chunk) * sin_chunk_k)

            q_chunk = q_chunk.permute(0, 3, 2, 1, 4).contiguous()        # [B, f, g, L, 2]
            q_pe_chunk = q_pe_chunk.permute(0, 3, 2, 1, 4).contiguous()  # [B, f, g, L, 2]
            k_chunk = k_chunk.permute(0, 2, 1, 3).contiguous()           # [B, f, L, 2]
            k_pe_chunk = k_pe_chunk.permute(0, 2, 1, 3).contiguous()     # [B, f, L, 2]

            attn_nope = torch.einsum("bfhqc,bfkc->bfhqk", q_chunk, k_chunk)
            attn_rope = torch.einsum("bfhqc,bfkc->bfhqk", q_pe_chunk, k_pe_chunk)
            diff[start:end, kv_idx] = ((attn_rope - attn_nope).abs() * mask).sum(
                dim=(0, 2, 3, 4), dtype=torch.float32
            )

            del q_chunk, k_chunk, q_pe_chunk, k_pe_chunk, attn_nope, attn_rope
    if dist.is_available() and dist.is_initialized():
        # sum diff across all ranks(GPUs)
        dist.all_reduce(diff, op=dist.ReduceOp.SUM)
    if int(os.environ.get("RANK", 0)) == 0:
        # only rank 0 prints and records
        DIFF[module.layer_idx] += diff.detach()

def hook_mha2mla_baseline(module, args, kwargs, output):
    origin_attn_out = output[0].flatten(0, 1).detach()
    origin_apply_rotary_pos_emb = ROTARY_MODELING.apply_rotary_pos_emb
    kv = getattr(module, "num_key_value_heads", module.config.num_key_value_heads)
    ROTARY_MODELING.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_mha2mla(
        idx_mat=RANK[module.layer_idx], topk=topk
    )

    with torch.no_grad():  # avoid save calculation graph(oom)
        attn_output, _ = module.forward(
            hidden_states=kwargs["hidden_states"],
            attention_mask=kwargs["attention_mask"],
            position_ids=kwargs["position_ids"],
            position_embeddings=kwargs["position_embeddings"],
        )
    attn_output = attn_output.flatten(0, 1).detach()
    attn_mse_loss = torch.nn.MSELoss(reduction="sum")
    if kwargs["attention_mask"] is not None:
        mask = kwargs["attention_mask"][:, 0, -1, :].flatten()
        error = attn_mse_loss(attn_output[mask], origin_attn_out[mask])
    else:
        # if all attention masks == 1, kwargs["attention_mask"] is None
        error = attn_mse_loss(attn_output, origin_attn_out)
    if dist.is_available() and dist.is_initialized():
        # sum error across all ranks(GPUs)
        dist.all_reduce(error, op=dist.ReduceOp.SUM)
    if int(os.environ.get("RANK", 0)) == 0:
        # only rank 0 prints and records
        ERROR_SUM_mha2mla[module.layer_idx] += error.item()
    ROTARY_MODELING.apply_rotary_pos_emb = origin_apply_rotary_pos_emb

def hook_svd_collect(module, args, kwargs, output):
    hidden_states = kwargs["hidden_states"]
    input_shape = hidden_states.shape[:-1]
    B, L = input_shape
    head_dim = module.head_dim
    kv = module.config.num_key_value_heads

    tmp = hidden_states.view(-1, hidden_states.size(-1))
    tmp = tmp.float() / L
    tmp = tmp.transpose(0, 1) @ tmp
    if E_XXT[module.layer_idx] is not None:
        E_XXT[module.layer_idx] += tmp.detach()
    else:
        E_XXT[module.layer_idx] = tmp.detach()

    query_states = module.q_proj(hidden_states).view(*input_shape, -1, head_dim)
    key_states = module.k_proj(hidden_states).view(*input_shape, -1, head_dim)
    value_states = module.v_proj(hidden_states).view(*input_shape, -1, head_dim)
    query_states_no_bias = torch.nn.functional.linear(hidden_states, module.q_proj.weight, bias=None).view(
        *input_shape, -1, head_dim
    )
    key_states_no_bias = torch.nn.functional.linear(hidden_states, module.k_proj.weight, bias=None).view(
        *input_shape, -1, head_dim
    )
    value_states_no_bias = torch.nn.functional.linear(hidden_states, module.v_proj.weight, bias=None).view(
        *input_shape, -1, head_dim
    )
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    query_states_no_bias = query_states_no_bias.transpose(1, 2)
    key_states_no_bias = key_states_no_bias.transpose(1, 2)
    value_states_no_bias = value_states_no_bias.transpose(1, 2)

    func = create_custom_sepearte_rope_nope_mha2mla(idx_mat=RANK[module.layer_idx], topk=topk)
    q_rope, k_rope, q_nope, k_nope = func(query_states_no_bias, key_states_no_bias)
    tmp = (k_nope.reshape(-1, head_dim) ** 2).sum(dim=0)
    if K_NORM[module.layer_idx] is not None:
        K_NORM[module.layer_idx] += tmp.detach()
    else:
        K_NORM[module.layer_idx] = tmp.detach()
    tmp = (value_states_no_bias.reshape(-1, head_dim) ** 2).sum(dim=0)
    if V_NORM[module.layer_idx] is not None:
        V_NORM[module.layer_idx] += tmp.detach()
    else:
        V_NORM[module.layer_idx] = tmp.detach()

    cos, sin = kwargs["position_embeddings"]
    rope_func = create_custom_apply_rotary_pos_emb_mha2mla(idx_mat=RANK[module.layer_idx], topk=topk)
    query_states, key_states = rope_func(
        query_states, key_states, cos, sin, position_ids=kwargs["position_ids"], unsqueeze_dim=1
    )

    # SDPA
    attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    attn_output, _ = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        kwargs["attention_mask"],
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=getattr(module, "scaling", head_dim ** (-0.5)),
        sliding_window=getattr(module, "sliding_window", None),
        softcap=getattr(module.config, "attn_logit_softcapping", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()   # [B,L,nh*dh]
    attn_output = module.o_proj(attn_output)

    if isinstance(output, tuple):
        return (attn_output,) + output[1:]
    else:
        return attn_output

def hook_mha2mla_partial_rope_svd(module, args, kwargs, output):
    hidden_states = kwargs["hidden_states"]
    input_shape = hidden_states.shape[:-1]
    B, L = input_shape
    head_dim = module.head_dim
    kv = module.config.num_key_value_heads
    nq = module.config.num_attention_heads
    layer_idx = module.layer_idx

    query_states = module.q_proj(hidden_states)

    key_states_full = module.k_proj(hidden_states)
    value_states_full = module.v_proj(hidden_states)

    W_down = SVD_weight["W_down"][layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
    W_up_k = SVD_weight["W_up_k"][layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
    W_up_v = SVD_weight["W_up_v"][layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)

    Wk_nope_approx = W_up_k @ W_down
    Wv_approx = W_up_v @ W_down

    k_nope_approx = torch.nn.functional.linear(hidden_states, Wk_nope_approx)
    v_approx = torch.nn.functional.linear(hidden_states, Wv_approx)
    if module.v_proj.bias is not None:
        v_bias = module.v_proj.bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        v_approx = v_approx + v_bias.view(1, 1, -1)

    idx_mat = RANK[layer_idx].to(hidden_states.device).long()
    n_freqs, num_kv_heads = idx_mat.shape
    freq_mask_k = torch.ones(n_freqs, num_kv_heads, dtype=torch.bool, device=hidden_states.device)
    freq_mask_k[idx_mat[:topk], torch.arange(num_kv_heads, device=hidden_states.device)] = False
    mask_k = freq_mask_k.repeat(2, 1)
    row_mask_flat = mask_k.t().contiguous().reshape(-1)

    key_states = key_states_full.clone()
    key_states[..., row_mask_flat] = k_nope_approx
    value_states = v_approx

    query_states = query_states.view(*input_shape, -1, head_dim)
    key_states = key_states.view(*input_shape, -1, head_dim)
    value_states = value_states.view(*input_shape, -1, head_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = kwargs["position_embeddings"]
    rope_func = create_custom_apply_rotary_pos_emb_mha2mla(
        idx_mat=RANK[layer_idx], topk=topk
    )
    query_states, key_states = rope_func(
        query_states, key_states, cos, sin,
        position_ids=kwargs["position_ids"], unsqueeze_dim=1
    )

    attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
    attn_output, _ = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        kwargs["attention_mask"],
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=getattr(module, "scaling", head_dim ** (-0.5)),
        sliding_window=getattr(module, "sliding_window", None),
        softcap=getattr(module.config, "attn_logit_softcapping", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.o_proj(attn_output).flatten(0, 1)
    origin_attn_out = output[0].flatten(0, 1).detach()

    attn_mse_loss = torch.nn.MSELoss(reduction="sum")
    if kwargs["attention_mask"] is not None:
        mask = kwargs["attention_mask"][:, 0, -1, :].flatten()
        error = attn_mse_loss(attn_output[mask], origin_attn_out[mask])
    else:
        error = attn_mse_loss(attn_output, origin_attn_out)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(error, op=dist.ReduceOp.SUM)

    if int(os.environ.get("RANK", 0)) == 0:
        ERROR_SUM_SVD[module.layer_idx] += error.item()

def register_hook(hook_func: callable):
    handles = [layer.self_attn.register_forward_hook(hook_func, with_kwargs=True) for layer in model.model.layers]
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()
    handles.clear()

def model_forward(need_count_tokens=False, offset_batches=0):
    rank = int(os.environ.get("RANK", 0))
    total_steps = num_samples // data_args.per_gpu_batchsize // world_size
    pbar = tqdm(total=total_steps, desc="fwd", disable=(rank != 0))
    for idx, batch in enumerate(dataloader):
        if idx < offset_batches:
            continue
        if idx - offset_batches >= total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            if need_count_tokens:
                batch_tokens = attention_mask.sum()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
                global NUM_TOKENS
                NUM_TOKENS += batch_tokens.item()
            _ = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
        if rank == 0:
            pbar.update(1)
    if rank == 0:
        pbar.close()

def do_SVD(svdnorm_balance_method):
    global SVD_weight
    SVD_weight = {
        "W_up_k": {},
        "W_up_v": {},
        "W_down": {},
    }
    METHOD2FUNC = {
        "original_svd": original_svd,
        "svdllm": svdllm,
        "scaled_svdllm": scaled_svdllm,
        "scaled_UD_svdllm": scaled_UD_svdllm,
        "scaled_KV_svdllm": scaled_KV_svdllm,
    }

    for layer_idx in tqdm(range(len(model.model.layers))):
        input = {
            "k_norm": K_NORM[layer_idx],
            "v_norm": V_NORM[layer_idx],
            "kvproj_input": E_XXT[layer_idx],
        }
        Wk = model.model.layers[layer_idx].self_attn.k_proj.weight.float()
        Wv = model.model.layers[layer_idx].self_attn.v_proj.weight.float()
        _, Wk = seperate_nope_rope(Wk, idx_mat=RANK[layer_idx], topk=topk)

        W_up_k, W_up_v, W_down = METHOD2FUNC[svdnorm_balance_method](
            Wk=Wk, Wv=Wv, input=input, r=lora_rank
        )

        device = model.device
        W_up_k = W_up_k.detach().to(device)
        W_up_v = W_up_v.detach().to(device)
        W_down = W_down.detach().to(device)

        SVD_weight["W_up_k"][layer_idx] = W_up_k.to(dtype=torch.bfloat16)
        SVD_weight["W_up_v"][layer_idx] = W_up_v.to(dtype=torch.bfloat16)
        SVD_weight["W_down"][layer_idx] = W_down.to(dtype=torch.bfloat16)

def save_mla_tensors():
    mla_tensor = [None for _ in range(n_layers)]
    for layer_idx in range(n_layers):
        q_proj = model.model.layers[layer_idx].self_attn.q_proj
        k_proj = model.model.layers[layer_idx].self_attn.k_proj
        v_proj = model.model.layers[layer_idx].self_attn.v_proj

        Wq = q_proj.weight
        Wqr, Wqc = seperate_nope_rope(Wq, idx_mat=RANK[layer_idx], topk=topk)
        Wk = k_proj.weight
        Wkr, Wkc = seperate_nope_rope(Wk, idx_mat=RANK[layer_idx], topk=topk)
        layer_tensor = {
            "q_r_proj": Wqr.to(dtype=torch.bfloat16),
            "q_c_proj": Wqc.to(dtype=torch.bfloat16),
            "k_r_proj": Wkr.to(dtype=torch.bfloat16),
            "kv_down_weight": SVD_weight["W_down"][layer_idx].to(dtype=torch.bfloat16),
            "v_up_weight": SVD_weight["W_up_v"][layer_idx].to(dtype=torch.bfloat16),
            "k_up_weight": SVD_weight["W_up_k"][layer_idx].to(dtype=torch.bfloat16),
            "rope_index": torch.sort(RANK[layer_idx][:topk], dim=0).values.to(dtype=torch.long),
        }

        if q_proj.bias is not None:
            q_r_bias, q_c_bias = seperate_nope_rope(q_proj.bias, idx_mat=RANK[layer_idx], topk=topk)
            layer_tensor["q_r_proj_bias"] = q_r_bias.to(dtype=torch.bfloat16)
            layer_tensor["q_c_proj_bias"] = q_c_bias.to(dtype=torch.bfloat16)
        if k_proj.bias is not None:
            k_r_bias, _ = seperate_nope_rope(k_proj.bias, idx_mat=RANK[layer_idx], topk=topk)
            layer_tensor["k_r_proj_bias"] = k_r_bias.to(dtype=torch.bfloat16)
        if v_proj.bias is not None:
            layer_tensor["v_proj_bias"] = v_proj.bias.to(dtype=torch.bfloat16)

        mla_tensor[layer_idx] = layer_tensor

    keys = list(mla_tensor[0].keys())
    concated_mla_tensor = {
        key: torch.stack([mla_tensor[i][key] for i in range(n_layers)], dim=0)
        for key in keys
    }

    for k, v in concated_mla_tensor.items():
        print(f"{k}: {tuple(v.shape)}  dtype={v.dtype}  device={v.device}")

    concated_mla_tensor = {
        key: value.detach().cpu() for key, value in concated_mla_tensor.items()
    }
    output_dir = os.path.dirname(os.path.abspath(cli_args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(concated_mla_tensor, cli_args.output_path)
    print(f"MLA tensors saved to {cli_args.output_path}")


num_batches = num_samples // data_args.per_gpu_batchsize // world_size
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
head_dim = getattr(
    model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
)
n_freqs = head_dim // 2
if not 0 < topk < n_freqs:
    raise ValueError(f"topk must be between 1 and {n_freqs - 1}, got {topk}")
if num_samples < data_args.per_gpu_batchsize * world_size:
    raise ValueError("num_samples must provide at least one batch per distributed rank")
ERROR_SUM_uniform = [0 for _ in range(n_layers)]
NUM_TOKENS = 0
handles = register_hook(hook_uniform_baseline)
model_forward(need_count_tokens=True, offset_batches=0)
remove_hooks(handles)
if int(os.environ.get("RANK", 0)) == 0:
    ERROR_SUM_uniform = [error_sum / NUM_TOKENS for error_sum in ERROR_SUM_uniform]
    print("ERROR_SUM_uniform:", [f"{x:.2f}" for x in ERROR_SUM_uniform])
    print("ERROR_SUM_uniform AVG:", f"{sum(ERROR_SUM_uniform)/len(ERROR_SUM_uniform):.2f}")
    print("Total tokens: ", NUM_TOKENS)

DIFF = [0 for _ in range(n_layers)]
RANK = [0 for _ in range(n_layers)]
handles = register_hook(hook_permute_freqs)
model_forward(need_count_tokens=False, offset_batches=0)
remove_hooks(handles)
if int(os.environ.get("RANK", 0)) == 0:
    for layer_idx in range(n_layers):
        rank_up_to_down = torch.argsort(DIFF[layer_idx], dim=0, descending=True)  # [n_freqs, nkv]
        RANK[layer_idx] = rank_up_to_down.detach()

        # print("RANK (top 10 frequencies per kv head):")
        # for kv_idx in range(model.config.num_key_value_heads):
        #     topk_freqs = rank_up_to_down[:, kv_idx][:10].cpu().numpy().tolist()
        #     print(f"Layer {layer_idx} KV head {kv_idx}: {topk_freqs}")

for layer_idx in range(n_layers):
    if rank != 0:
        RANK[layer_idx] = torch.empty(
            n_freqs,
            model.config.num_key_value_heads,
            dtype=torch.long,
            device=device,
        )
    dist.broadcast(RANK[layer_idx], src=0)


ERROR_SUM_mha2mla = [0 for _ in range(n_layers)]
handles = register_hook(hook_mha2mla_baseline)
model_forward(need_count_tokens=False, offset_batches=0)
remove_hooks(handles)
if int(os.environ.get("RANK", 0)) == 0:
    ERROR_SUM_mha2mla = [error_sum / NUM_TOKENS for error_sum in ERROR_SUM_mha2mla]
    print("ERROR_SUM_mha2mla:", [f"{x:.2f}" for x in ERROR_SUM_mha2mla])
    print("ERROR_SUM_mha2mla AVG:", f"{sum(ERROR_SUM_mha2mla)/len(ERROR_SUM_mha2mla):.2f}")

E_XXT = [None for _ in range(n_layers)]
K_NORM = [None for _ in range(n_layers)]
V_NORM = [None for _ in range(n_layers)]
handles = register_hook(hook_svd_collect)
model_forward(need_count_tokens=False, offset_batches=0)
remove_hooks(handles)
for values in (E_XXT, K_NORM, V_NORM):
    for value in values:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
K_NORM = [k_norm.sqrt().mean() for k_norm in K_NORM]
V_NORM = [v_norm.sqrt().mean() for v_norm in V_NORM]
print(K_NORM, V_NORM)

do_SVD(cli_args.svd_method)

ERROR_SUM_SVD = [0 for _ in range(n_layers)]
handles = register_hook(hook_mha2mla_partial_rope_svd)
model_forward(need_count_tokens=False, offset_batches=0)
remove_hooks(handles)

if int(os.environ.get("RANK", 0)) == 0:
    ERROR_SUM_SVD = [x / NUM_TOKENS for x in ERROR_SUM_SVD]
    print("ERROR_SUM_SVD:", [f"{x:.2f}" for x in ERROR_SUM_SVD])
    print("ERROR_SUM_SVD AVG:", f"{sum(ERROR_SUM_SVD)/len(ERROR_SUM_SVD):.2f}")

if rank == 0:
    save_mla_tensors()

dist.destroy_process_group()
