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
"""Qwen3-MoE rotary state must survive empty pretrained materialization."""

import unittest

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.qwen3_moe.adapter.registration import QWEN3_MOE_ADAPTER_SPEC


class TestRotaryMaterialization(unittest.TestCase):
    """Exercise actual HF rotary buffers through Hyper's materialization lifecycle."""

    def test_rotary_buffers_rebuilt_without_changing_parameters(self) -> None:
        """Restore both frequency buffers and native RoPE output after to_empty."""
        config = Qwen3MoeConfig(hidden_size=128, num_attention_heads=4, head_dim=32)
        reference = Qwen3MoeRotaryEmbedding(config, device="cpu")
        hook = QWEN3_MOE_ADAPTER_SPEC.materialization
        self.assertTrue(callable(hook), "Qwen3-MoE must register its rotary materialization hook")
        for dtype in (torch.float32, torch.bfloat16):
            for reason in ("checkpoint_load", "random_init"):
                with self.subTest(dtype=dtype, reason=reason):
                    model = nn.Module()
                    model.rotary_emb = Qwen3MoeRotaryEmbedding(config, device="meta").to(dtype=dtype)
                    model.weight = nn.Parameter(torch.empty(2, device="meta", dtype=dtype))
                    hook(model)
                    model.to_empty(device="cpu")
                    with torch.no_grad():
                        model.weight.fill_(7)
                        for buffer in model.buffers():
                            buffer.fill_(float("nan"))
                    parameter = model.weight
                    rebuild_materialized_state(
                        model, MaterializationContext(reason=reason, device=torch.device("cpu")),
                    )
                    self.assertIs(model.weight, parameter)
                    torch.testing.assert_close(model.weight, torch.full((2,), 7, dtype=dtype), rtol=0, atol=0)
                    for name, expected in reference.named_buffers():
                        actual = model.rotary_emb.get_buffer(name)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
                    positions = torch.tensor([[0, 1, 7, 31]])
                    hidden = torch.zeros(1, 4, 128, dtype=dtype)
                    expected_module = Qwen3MoeRotaryEmbedding(config, device="cpu").to(dtype=dtype)
                    actual_cos, actual_sin = model.rotary_emb(hidden, positions)
                    expected_cos, expected_sin = expected_module(hidden, positions)
                    torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=0)
                    torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=0)
