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
"""Registered model construction supports native HF subclasses without redundant from_config wrappers."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg


import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3MoeConfig

from hyper_parallel.models._transformers.model_builder import _init_model
from hyper_parallel.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


class TestRegisteredConstruction(unittest.TestCase):
    """Exercise real HF dtype and attention initialization through the shared custom dispatch."""

    def test_hf_subclass_uses_native_configuration_constructor(self) -> None:
        """Feature: Registered HF model construction.
        Description: Build a real HF subclass without adding a model-specific from_config forwarding method.
        Expectation: Native initialization consumes dtype/attention options and strips AutoModel-only keywords.
        """
        config = Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
        )
        config.architectures = ["Qwen3MoeForCausalLM"]
        with patch(
            "hyper_parallel.models._transformers.model_builder._resolve_custom_model_cls",
            return_value=Qwen3MoeForCausalLM,
        ):
            custom, model = _init_model(object, None, config, "eager", torch.float32, False, trust_remote_code=True)
        self.assertTrue(custom)
        self.assertIsInstance(model, Qwen3MoeForCausalLM)
        self.assertTrue(model.lm_head.supports_chunked_cross_entropy)
        self.assertEqual(model.config._attn_implementation, "eager")
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        self.assertEqual(tuple(output.logits.shape), (1, 3, 32))
