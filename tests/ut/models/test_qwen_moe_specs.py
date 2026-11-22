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
"""Default HF MoE model migrations execute ordinary CPU forward and backward."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg


import copy
import unittest

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.qwen2_moe.qwen2_moe_spec import MODEL_SPEC as QWEN2_SPEC
from hyper_parallel.models.qwen3_moe.qwen3_moe_spec import MODEL_SPEC as QWEN3_SPEC


class TestQwenMoeSpecs(unittest.TestCase):
    """Use actual HF decoders and natural routes without replacing device checks."""

    def test_cpu_models_match_source(self) -> None:
        """Feature: Portable model defaults.
        Description: Install model specifications into real two-layer HF MoE models with padded input.
        Expectation: Logits, embedding/router/expert gradients match and no NPU kernel is called.
        """
        for config_class, model_class, spec in (
            (Qwen2MoeConfig, Qwen2MoeForCausalLM, QWEN2_SPEC),
            (Qwen3MoeConfig, Qwen3MoeForCausalLM, QWEN3_SPEC),
        ):
            with self.subTest(model=spec.model_type):
                config = config_class(
                    hidden_size=16,
                    intermediate_size=32,
                    moe_intermediate_size=8,
                    shared_expert_intermediate_size=12,
                    num_experts=4,
                    num_experts_per_tok=2,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    vocab_size=32,
                )
                config._attn_implementation = "eager"
                config.use_cache = False
                source = model_class(config).eval()
                target = copy.deepcopy(source)
                target, _ = prepare_model_components(target, build_module_context(model_spec=spec, config=config))
                ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
                mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
                expected = source(input_ids=ids, attention_mask=mask, output_router_logits=True)
                actual = target(input_ids=ids, attention_mask=mask, output_router_logits=True)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
                expected.logits.square().mean().backward()
                actual.logits.square().mean().backward()
                left = dict(source.named_parameters())
                for name, parameter in target.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if name in left:
                        torch.testing.assert_close(parameter.grad, left[name].grad, rtol=1e-4, atol=1e-6)
