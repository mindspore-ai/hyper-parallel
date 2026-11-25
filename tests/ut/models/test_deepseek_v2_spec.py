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
"""Independent V2 identity, complex rotary source protocol and separate Muon matrix boundaries."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from __future__ import annotations

import copy
import unittest

import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM

from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.deepseek_v3.deepseek_v3_spec import MODEL_SPEC, V2_SPEC
from hyper_parallel.models.module_build import build_module_context, prepare_model_components


class TestDeepseekV2Spec(unittest.TestCase):
    """Exercise actual HF V2 layers without borrowing a V3 config or gate protocol."""

    def test_source_values_and_gradients(self) -> None:
        """Feature: V2 compatibility identity.
        Description: Adopt a full dense-to-MoE decoder with native complex-valued rotary outputs.
        Expectation: Every parameter identity and separate latent projection is preserved; outputs/gradients match.
        """
        torch.set_num_threads(2)
        config = DeepseekV2Config(
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            first_k_dense_replace=1,
            n_routed_experts=4,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            vocab_size=64,
        )
        config._attn_implementation = "eager"
        config.use_cache = False
        source = DeepseekV2ForCausalLM(config).eval()
        original = copy.deepcopy(source)
        parameters = dict(original.named_parameters())
        target, _ = prepare_model_components(original, build_module_context(model_spec=V2_SPEC, config=config))
        self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
        for name, parameter in parameters.items():
            self.assertIs(parameter, target.get_parameter(name))
        self.assertFalse(target.model.layers[0].self_attn.fuse_latent_qkv)
        self.assertEqual(target.model.layers[0].self_attn.key_expansion_layout, "bnsd")
        self.assertEqual(target.model.rotary_emb.output_format, "complex")
        ids = torch.tensor([[1, 2, 3, 4]])
        expected, actual = source(input_ids=ids).logits, target(input_ids=ids).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=1e-4, atol=1e-6)


class TestDeepseekV3Spec(unittest.TestCase):
    """Verify source values survive the declared latent down-projection fusion."""


    def test_packed_latent_source_values_and_gradients(self) -> None:
        """Feature: V3 public MLA and routed experts.
        Description: Preserve dense-to-MoE roles while copying physical latent weights through the declared transform.
        Expectation: Logits, packed logical gradients and every unchanged source gradient agree.
        """
        torch.set_num_threads(2)
        config = DeepseekV3Config(
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            first_k_dense_replace=1,
            n_routed_experts=4,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            vocab_size=64,
        )
        config._attn_implementation = "eager"
        config.use_cache = False
        source = DeepseekV3ForCausalLM(config).eval()
        target, _ = prepare_model_components(
            copy.deepcopy(source), build_module_context(model_spec=MODEL_SPEC, config=config)
        )
        self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
        self.assertEqual(target.model.layers[0].self_attn.key_expansion_layout, "bsnd")
        ids = torch.tensor([[1, 2, 3, 4]])
        expected, actual = source(input_ids=ids).logits, target(input_ids=ids).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            if ".linear_qkv." in name:
                gate_name = name.replace(".linear_qkv.", ".q_a_proj.")
                key_name = name.replace(".linear_qkv.", ".kv_a_proj_with_mqa.")
                expected_value = torch.cat((reference[gate_name], reference[key_name]), dim=0)
                expected_gradient = torch.cat((reference[gate_name].grad, reference[key_name].grad), dim=0)
                torch.testing.assert_close(parameter, expected_value, rtol=0, atol=0)
            else:
                expected_gradient = reference[name].grad
            torch.testing.assert_close(parameter.grad, expected_gradient, rtol=1e-4, atol=1e-6)
