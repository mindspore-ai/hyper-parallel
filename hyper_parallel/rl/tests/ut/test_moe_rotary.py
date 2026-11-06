# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CPU regressions for native MoE RoPE recovery after meta materialization."""
import unittest

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from rl.workers.adapters.moe_rotary import rebuild_qwen3_moe_rotary_state


def tiny_config(rope_type: str = "default") -> Qwen3MoeConfig:
    """Return a small native model with explicit rotary configuration."""
    parameters = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type != "default":
        parameters["factor"] = 2.0
    return Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        rope_parameters=parameters,
    )


class TestMoERotaryRecovery(unittest.TestCase):
    """Verify actual native-model probabilities and derived buffer recovery."""

    def test_meta_reload_recovers_native_logits(self):
        """Recover native logits after weights load without non-persistent RoPE state."""
        torch.manual_seed(7)
        reference = Qwen3MoeForCausalLM(tiny_config()).eval()
        with torch.device("meta"):
            restored = Qwen3MoeForCausalLM(tiny_config()).eval()
        restored.to_empty(device="cpu")
        restored.load_state_dict(reference.state_dict())
        rotary = restored.model.rotary_emb
        rotary.inv_freq.zero_()
        rotary.original_inv_freq.zero_()
        tokens = torch.arange(32).remainder(31).unsqueeze(0)
        with torch.no_grad():
            expected = reference(tokens).logits
            broken = restored(tokens).logits
        self.assertGreater(float((broken - expected).abs().max()), 0.0)
        parameter_ids = [id(parameter) for parameter in restored.parameters()]
        rebuild_qwen3_moe_rotary_state(restored)
        with torch.no_grad():
            actual = restored(tokens).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(parameter_ids, [id(parameter) for parameter in restored.parameters()])
        self.assertFalse(any("inv_freq" in name for name in restored.state_dict()))

    def test_scaled_rope_and_original_frequency_recovery(self):
        """Restore FP32 frequencies, scaling, and dynamic cache from native configuration."""
        for rope_type in ("default", "linear", "dynamic"):
            with self.subTest(rope_type=rope_type):
                model = Qwen3MoeForCausalLM(tiny_config(rope_type))
                reference = Qwen3MoeForCausalLM(tiny_config(rope_type)).model.rotary_emb
                rotary = model.model.rotary_emb
                rotary.inv_freq.zero_()
                rotary.original_inv_freq.zero_()
                rotary.attention_scaling = 0.0
                rotary.max_seq_len_cached = 1000
                rebuild_qwen3_moe_rotary_state(model)
                self.assertEqual(rotary.inv_freq.dtype, torch.float32)
                torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
                torch.testing.assert_close(rotary.original_inv_freq, reference.original_inv_freq, rtol=0, atol=0)
                self.assertEqual(rotary.attention_scaling, reference.attention_scaling)
                self.assertEqual(rotary.max_seq_len_cached, reference.max_seq_len_cached)

    def test_unmaterialized_model_is_rejected(self):
        """Prevent accidental recovery into meta storage."""
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(tiny_config())
        with self.assertRaisesRegex(ValueError, "materialized"):
            rebuild_qwen3_moe_rotary_state(model)


if __name__ == "__main__":
    unittest.main()
