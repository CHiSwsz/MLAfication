import sys
import torch
import inspect
from transformers.models.qwen2.modeling_qwen2 import rotate_half


# Monkey patching to use custom partial-rope function
def create_custom_apply_rotary_pos_emb(partial_rope_uniform_step):
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        step = partial_rope_uniform_step
        for i in range(0, q.size(-1), step):
            q_embed[..., i + 1 : i + step] = q[..., i + 1 : i + step]
            k_embed[..., i + 1 : i + step] = k[..., i + 1 : i + step]
        return q_embed, k_embed

    return custom_apply_rotary_pos_emb

def create_custom_apply_rotary_pos_emb_mha2mla(idx_mat, topk):
    def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Get the calling frame
        group_size = q.size(1) // k.size(1)  # num_heads / num_kv_heads

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # ---- 构造 head_dim 级 mask ----
        n_freqs, device = idx_mat.size(0), q.device
        head_dim = q.size(-1)
        freq_mask_k = torch.ones(n_freqs, k.size(1), dtype=torch.bool, device=device)
        freq_mask_k[idx_mat[:topk], torch.arange(k.size(1), device=device)] = False
        freq_mask_q = freq_mask_k.repeat_interleave(group_size, dim=1)
        mask_q = freq_mask_q.repeat(2, 1)   # [head_dim, num_heads]
        mask_k = freq_mask_k.repeat(2, 1)   # [head_dim, num_kv_heads]
        q_embed = torch.where(mask_q.t().view(1, q.size(1), 1, head_dim), q, q_embed)
        k_embed = torch.where(mask_k.t().view(1, k.size(1), 1, head_dim), k, k_embed)

        return q_embed, k_embed

    return custom_apply_rotary_pos_emb

def create_custom_sepearte_rope_nope_mha2mla(idx_mat, topk):
    def custom_sepearte_rope_nope(q, k):
        group_size = q.size(1) // k.size(1)

        n_freqs, device = idx_mat.size(0), q.device
        head_dim = q.size(-1)

        freq_mask_k = torch.ones(n_freqs, k.size(1), dtype=torch.bool, device=device)
        freq_mask_k[idx_mat[:topk], torch.arange(k.size(1), device=device)] = False
        freq_mask_q = freq_mask_k.repeat_interleave(group_size, dim=1)

        mask_q = freq_mask_q.repeat(2, 1)
        mask_k = freq_mask_k.repeat(2, 1)

        q_mask_view = mask_q.t().view(1, q.size(1), 1, head_dim)  # True = nope
        k_mask_view = mask_k.t().view(1, k.size(1), 1, head_dim)  # True = nope

        q_rope = torch.where(~q_mask_view, q, torch.zeros_like(q))
        k_rope = torch.where(~k_mask_view, k, torch.zeros_like(k))
        q_nope = torch.where(q_mask_view, q, torch.zeros_like(q))
        k_nope = torch.where(k_mask_view, k, torch.zeros_like(k))

        return q_rope, k_rope, q_nope, k_nope

    return custom_sepearte_rope_nope

def seperate_nope_rope(W, idx_mat, topk):
    idx_mat = idx_mat.long()
    n_freqs, num_kv_heads = idx_mat.shape
    head_dim = n_freqs * 2
    num_heads = W.shape[0] // head_dim

    freq_mask = torch.ones(n_freqs, num_kv_heads, dtype=torch.bool, device=idx_mat.device)
    freq_mask[idx_mat[:topk], torch.arange(num_kv_heads, device=idx_mat.device)] = False

    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        freq_mask = freq_mask.repeat_interleave(repeat, dim=1)

    mask = freq_mask.repeat(2, 1)
    row_mask_flat = mask.t().contiguous().reshape(-1).to(W.device)  # True = nope

    W_rope = W[~row_mask_flat]
    W_nope = W[row_mask_flat]
    return W_rope, W_nope

def apply_U(x, repeat, U, parts):
    nh = x.size(-2)
    n_freqs = x.size(-1) // 2
    for h_idx in range(nh):
        for p_idx, (l, r) in enumerate(parts):
            u = U[h_idx // repeat][p_idx]
            x[..., h_idx, l: r + 1] = x[..., h_idx, l: r + 1] @ u # real
            x[..., h_idx, n_freqs + l: n_freqs + r + 1] = x[..., h_idx, n_freqs + l: n_freqs + r + 1] @ u # imag
    return x
