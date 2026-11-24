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
"""Qwen3 default family declarations and explicit replacement precedence."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg


import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.qwen3.qwen3_spec import MODEL_SPEC


class TestQwen3Spec(unittest.TestCase):
    """Exercise actual HF structure and default discovery without kernels or weights."""


    def test_cpu_forward_and_gradients_match_hf(self) -> None:
        """Feature: Portable Qwen3 default families.
        Description: Run real HF and public-family models on CPU with a padding mask.
        Expectation: Logits and every parameter gradient agree with the source implementation.
        """
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
        )
        config._attn_implementation = "eager"
        config.use_cache = False
        source = Qwen3ForCausalLM(config).eval()
        target = copy.deepcopy(source)
        target, _ = prepare_model_components(target, build_module_context(model_spec=MODEL_SPEC, config=config))
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        expected = source(input_ids=ids, attention_mask=mask).logits
        actual = target(input_ids=ids, attention_mask=mask).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for left, right in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
