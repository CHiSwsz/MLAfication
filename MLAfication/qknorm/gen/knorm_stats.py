"""Token-wise key RMS statistics, pooled over batches and distributed ranks."""

import json
from pathlib import Path

import torch
import torch.distributed as dist


def key_rms(key_states, eps):
    """The pre-KNorm denominator, with shape [batch, sequence, KV heads]."""
    return (key_states.detach().float().square().mean(dim=-1) + eps).sqrt()


class TokenNormStats:
    """Merge central moments in FP64; keep heads separate and count valid tokens.

    ``std`` is the population standard deviation (correction=0) of token-wise
    norms, not a standard error or a standard deviation of batch means.
    """

    def __init__(self, num_heads, device="cpu"):
        self.count = 0
        self.mean = torch.zeros(num_heads, dtype=torch.float64, device=device)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, norms, attention_mask=None):
        if norms.ndim != 3 or norms.shape[-1] != self.mean.numel():
            raise ValueError("Expected norms with shape [batch, sequence, KV heads]")
        values = norms.detach().reshape(-1, norms.shape[-1]).to(self.mean)
        if attention_mask is not None:
            if attention_mask.shape != norms.shape[:2]:
                raise ValueError("Expected a 2D token attention mask matching the norms")
            values = values[attention_mask.to(device=values.device, dtype=torch.bool).reshape(-1)]
        count = values.shape[0]
        if not count:
            return
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite key norm encountered at a valid token")
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        total = self.count + count
        delta = mean - self.mean
        self.m2 += variance * count + delta.square() * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def synchronize(self):
        """Pool moments, including unequal or empty ranks; call once after updates."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return
        count = self.mean.new_tensor(self.count)
        dist.all_reduce(count)
        total = int(count.item())
        if not total:
            return
        mean = self.mean * self.count
        dist.all_reduce(mean)
        mean /= total
        m2 = self.m2 + self.count * (self.mean - mean).square()
        dist.all_reduce(m2)
        self.count, self.mean, self.m2 = total, mean, m2

    @property
    def std(self):
        if not self.count:
            raise ValueError("No valid tokens collected")
        return (self.m2 / self.count).clamp_min(0).sqrt()

    def records(self):
        means, stds = self.mean.tolist(), self.std.tolist()
        return [
            {"head": head, "count": self.count, "mean": mean, "std": std,
             "cv": std / mean if mean else None}
            for head, (mean, std) in enumerate(zip(means, stds))
        ]


def context_record(stats, sequence_length, layer=None, head=None):
    layers = [{"layer": i, "heads": value.records()} for i, value in enumerate(stats)]
    layer = len(layers) // 2 if layer is None else layer
    if not 0 <= layer < len(layers):
        raise ValueError(f"Invalid layer index: {layer}")
    heads = layers[layer]["heads"]
    head = len(heads) // 2 if head is None else head
    if not 0 <= head < len(heads):
        raise ValueError(f"Invalid KV head index: {head}")
    return {"context_length": sequence_length, "layers": layers,
            "selected": {"layer": layer, **heads[head]}}


def write_report(path, contexts, **metadata):
    report = {
        "schema_version": 1,
        "norm_definition": "sqrt(mean(key_before_knorm.float() ** 2, dim=-1) + rms_norm_eps)",
        "std_definition": "sqrt(mean((token_norm - mean_token_norm) ** 2))",
        "std_correction": 0,
        "padding_excluded": True,
        **metadata,
        "contexts": contexts,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
