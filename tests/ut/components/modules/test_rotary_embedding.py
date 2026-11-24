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
"""HF rotary adoption and final meta-state reconstruction."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from hyper_parallel.components.modules.common.rotary_embedding import RotaryEmbedding
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state


class TestRotarySource(unittest.TestCase):
    """Compare real source buffers, outputs and HF initialization hooks."""

    def test_meta_source_rebuilds_both_native_buffers_exactly(self) -> None:
        """Feature: Rotary source materialization.
        Description: Adopt an HF meta source and materialize it through the public lifecycle.
        Expectation: Both buffers and cos/sin match the independent CPU source exactly.
        """
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=32,
        )
        reference = Qwen3RotaryEmbedding(config)
        with torch.device("meta"):
            source = Qwen3RotaryEmbedding(config)
            target = RotaryEmbedding(module=source)
        self.assertTrue(target.inv_freq.is_meta)
        target.to_empty(device="cpu")
        rebuild_materialized_state(target, MaterializationContext(reason="random_init", device=torch.device("cpu")))
        self.assertEqual(dict(target.named_buffers()).keys(), dict(reference.named_buffers()).keys())
        for name, value in reference.named_buffers():
            torch.testing.assert_close(getattr(target, name), value, rtol=0, atol=0)
        value, positions = torch.randn(2, 7, 16), torch.arange(7).expand(2, -1)
        for actual, expected in zip(target(value, positions), reference(value, positions)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model = Qwen3ForCausalLM(config)
        model._init_weights(target)
        torch.testing.assert_close(target.inv_freq, reference.inv_freq, rtol=0, atol=0)
        torch.testing.assert_close(target.original_inv_freq, reference.original_inv_freq, rtol=0, atol=0)


    def test_explicit_output_and_axis_contracts_without_model_identity(self) -> None:
        """Generic source names must support declared complex and interleaved layouts."""
        source = torch.nn.Module()
        source.config = SimpleNamespace(rope_parameters={"mrope_section": (1, 1, 2)})
        source.register_buffer("inv_freq", torch.tensor([1.0, 0.1, 0.01, 0.001]), persistent=False)
        positions = torch.arange(4).expand(1, -1)
        axes = torch.stack((positions, positions + 1, positions + 2))
        x = torch.empty(1, 4, 8)
        with self.assertRaisesRegex(ValueError, "explicit mrope_interleaved"):
            RotaryEmbedding(module=source)
        member = RotaryEmbedding(module=source, output_format="complex", mrope_interleaved=True)
        angles = axes.float().unsqueeze(-1) * source.inv_freq
        expected = angles[0].clone()
        expected[..., 1:3:3] = angles[1, ..., 1:3:3]
        expected[..., 2:6:3] = angles[2, ..., 2:6:3]
        torch.testing.assert_close(member(x, axes), torch.polar(torch.ones_like(expected), expected), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "output_format"):
            RotaryEmbedding(module=source, output_format="unknown")

    def test_meta_rebuild_uses_source_initialization_protocol(self) -> None:
        """HF sources must use their declared initializer rather than a module-path heuristic."""
        config = Qwen3Config(hidden_size=16, num_attention_heads=2, head_dim=8)
        frequency = torch.tensor([1.0, 0.125, 0.015625, 0.001953125])
        with torch.device("meta"):
            source = Qwen3RotaryEmbedding(config)
        with patch.object(source, "compute_default_rope_parameters", return_value=(frequency, 1.0)) as builder:
            member = RotaryEmbedding(module=source)
        builder.assert_called_once_with(config, device="cpu")
        member.to_empty(device="cpu")
        rebuild_materialized_state(member, MaterializationContext(reason="random_init", device=torch.device("cpu")))
        torch.testing.assert_close(member.inv_freq, frequency, rtol=0, atol=0)
        torch.testing.assert_close(member.original_inv_freq, frequency, rtol=0, atol=0)
