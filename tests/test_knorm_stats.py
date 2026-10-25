"""CPU checks for token-wise KNorm statistics and native Qwen3 observation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MLAfication/qknorm/eval"))
from knorm_stats import TokenNormStats, context_record, key_rms, write_report

spec = importlib.util.spec_from_file_location(
    "collect_knorm_stats", ROOT / "MLAfication/qknorm/eval/collect_knorm_stats.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class KNormStatsTests(unittest.TestCase):
    def test_token_statistics_mask_padding_and_weight_unequal_batches(self):
        keys = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                              [[1000., 1000.], [1000., 1000.]]],
                             [[[9., 10.], [11., 12.]], [[1000., 1000.], [1000., 1000.]],
                              [[1000., 1000.], [1000., 1000.]]]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        norms = key_rms(keys, 1e-6)
        stats = TokenNormStats(2)
        stats.update(norms[:1], mask[:1])
        stats.update(norms[1:], mask[1:])
        expected = norms[mask.bool()].double()
        self.assertEqual(stats.count, 3)
        torch.testing.assert_close(stats.mean, expected.mean(0))
        torch.testing.assert_close(stats.std, expected.std(0, correction=0))

    def test_small_variance_at_large_mean_and_constant_values(self):
        values = (1e9 + torch.arange(10, dtype=torch.float64) * .01).reshape(1, 10, 1)
        stats = TokenNormStats(1)
        for batch in values.split(3, dim=1):
            stats.update(batch)
        torch.testing.assert_close(stats.std, values.std(dim=(0, 1), correction=0), rtol=1e-5, atol=1e-8)
        constant = TokenNormStats(1)
        constant.update(torch.ones(1, 1, 1))
        self.assertEqual(constant.std.item(), 0)

    def test_empty_and_nonfinite_valid_tokens(self):
        stats = TokenNormStats(1)
        stats.update(torch.full((1, 2, 1), float("nan")), torch.zeros(1, 2))
        self.assertEqual(stats.count, 0)
        with self.assertRaisesRegex(ValueError, "No valid tokens"):
            _ = stats.std
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            stats.update(torch.full((1, 1, 1), float("nan")))

    def test_native_qwen3_hooks_match_direct_norms_and_leave_model_unchanged(self):
        torch.manual_seed(13)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, pad_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
        mask = ids.ne(0).long()
        rows = [{"input_ids": a, "attention_mask": b} for a, b in zip(ids, mask)]
        captured = []
        norm = model.model.layers[1].self_attn.k_norm
        handle = norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        with torch.inference_mode():
            before = model(ids, attention_mask=mask).logits
        handle.remove()
        expected = key_rms(captured[0], config.rms_norm_eps)[mask.bool()].double()
        stats, count, full = collector.collect_context(model, DataLoader(rows, batch_size=2), 4, "cpu")
        self.assertEqual((count, full, stats[1].count), (3, 1, 9))
        torch.testing.assert_close(stats[1].mean, expected.mean(0), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(stats[1].std, expected.std(0, correction=0), atol=1e-7, rtol=1e-6)
        with torch.inference_mode():
            after = model(ids, attention_mask=mask).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertFalse(norm._forward_pre_hooks)
        shorter, _, _ = collector.collect_context(model, DataLoader(rows), 2, "cpu")
        self.assertEqual(shorter[1].count, 6)
        # Without a mask, packed rows count every position, including EOS/pad IDs.
        packed = [{"input_ids": row} for row in ids]
        all_tokens, _, _ = collector.collect_context(model, DataLoader(packed), 4, "cpu")
        self.assertEqual(all_tokens[0].count, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            write_report(path, [context_record(stats, 4)], key_source="original_pre_knorm")
            report = json.loads(path.read_text())
        self.assertEqual(report["std_correction"], 0)
        selected = report["contexts"][0]["selected"]
        self.assertEqual((selected["layer"], selected["head"], selected["count"]), (1, 1, 9))


if __name__ == "__main__":
    unittest.main()
