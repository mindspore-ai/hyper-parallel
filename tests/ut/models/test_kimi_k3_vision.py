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
"""Temporal vision's host geometry and complete meta-state restoration contracts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.models.kimi_k3.components.vision import TemporalVisionTransformer
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state


class TestKimiK3Vision(unittest.TestCase):
    """Exercise real vision blocks without optional remote-source or accelerator dependencies."""

    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            patch_size=2, init_pos_emb_height=4, init_pos_emb_width=4, init_pos_emb_time=4,
            pos_emb_type="divided_fixed", num_attention_heads=2, num_hidden_layers=2, hidden_size=16,
            intermediate_size=32, merge_kernel_size=(2, 2), merge_type="sd2_tpool", qkv_hidden_size=24,
            initializer_range=0.02,
        )

    def test_meta_state_and_cached_frequency_rebuild(self) -> None:
        """Feature: Complete temporal vision materialization.
        Description: Rebuild both cold meta state and a previously warmed rotary cache.
        Expectation: Learned scales/positions initialize, derived buffers restore, and real gradients remain finite.
        """
        torch.set_num_threads(2)
        with torch.device("meta"):
            model = TemporalVisionTransformer(config=self._config())
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        context = MaterializationContext(reason="random_init", device=torch.device("cpu"))
        rebuild_materialized_state(model, context)
        grid = torch.tensor([[1, 4, 4], [2, 4, 4]])
        inputs = torch.randn(48, 3, 2, 2, requires_grad=True)
        outputs = model(inputs, grid)
        sum(item.square().mean() for item in outputs).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        temporal = model.patch_embed.pos_emb.time_weight.clone()
        rotary = model.encoder.rope_2d.freqs_cis.clone()
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        rebuild_materialized_state(model, context)
        torch.testing.assert_close(model.patch_embed.pos_emb.time_weight, temporal, atol=0, rtol=0)
        torch.testing.assert_close(model.encoder.rope_2d.freqs_cis, rotary, atol=0, rtol=0)
        torch.testing.assert_close(model.encoder.final_layernorm.weight, torch.ones(16), atol=0, rtol=0)

    def test_sdpa_and_invalid_geometry(self) -> None:
        """Feature: Host media geometry.
        Description: Run SDPA with two clip lengths and reject device-only or mismatched patch metadata.
        Expectation: SDPA forward/backward completes; invalid geometry fails before visual execution.
        """
        model = TemporalVisionTransformer(config=self._config(), attention_backend="sdpa")
        inputs = torch.randn(48, 3, 2, 2, requires_grad=True)
        outputs = model(inputs, torch.tensor([[1, 4, 4], [2, 4, 4]]))
        self.assertEqual([tuple(item.shape) for item in outputs], [(4, 4, 16), (4, 4, 16)])
        sum(item.square().mean() for item in outputs).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        with self.assertRaisesRegex(ValueError, "CPU"):
            model(inputs, torch.empty(2, 3, device="meta", dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "patch count"):
            model(inputs, torch.tensor([[1, 4, 4]]))
