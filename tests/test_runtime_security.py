"""CPU-only checks for the patched tensor stack, using no pretrained weights."""
import importlib.util
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorRuntimeTests(unittest.TestCase):
    def test_weights_only_accepts_tensors_and_rejects_arbitrary_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            expected = torch.arange(6).reshape(2, 3)
            torch.save({'weight': expected}, path)
            restored = torch.load(path, map_location='cpu', weights_only=True)
            torch.testing.assert_close(restored['weight'], expected)
            torch.save(SimpleNamespace(value='not a tensor checkpoint'), path)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(path, map_location='cpu', weights_only=True)

    def test_current_mla_patch_forward_and_backward(self):
        patch = load_module('MLAfication/mha_gqa/src/mha2mla/patching_mlafication.py', 'runtime_mla_patch')
        forward = load_module('MLAfication/mha_gqa/src/mha2mla/patching_mla_fwd.py', 'runtime_mla_forward')
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config)
        weights = {
            'q_r_proj': torch.randn(1, 16, 32) * .01,
            'q_c_proj': torch.randn(1, 16, 32) * .01,
            'k_r_proj': torch.randn(1, 8, 32) * .01,
            'k_up_weight': torch.randn(1, 8, 4) * .01,
            'v_up_weight': torch.randn(1, 16, 4) * .01,
            'kv_down_weight': torch.randn(1, 4, 32) * .01,
            'rope_index': torch.tensor([[[0, 0], [1, 1]]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mla.pt'
            torch.save(weights, path)
            model = patch.patch_mlafication(model, config, SimpleNamespace(mla_tensor_path=str(path)))
            model = forward.patch_mla_fwd_instance(model)
            tokens = torch.tensor([[1, 2, 3, 4]])
            output = model(tokens, labels=tokens, use_cache=False)
            self.assertEqual(output.logits.shape, (1, 4, 64))
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(model.model.layers[0].self_attn.kv_proj.down_kv.weight.grad)


if __name__ == '__main__':
    unittest.main()
