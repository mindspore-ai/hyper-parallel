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
"""Native Kimi Linear hybrid-role and complete source-gradient parity."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from __future__ import annotations

import copy
import unittest

import torch
from transformers import KimiLinearConfig, KimiLinearForCausalLM

from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.kimi_linear.kimi_linear_spec import MODEL_SPEC
from hyper_parallel.models.module_build import build_module_context, prepare_model_components


class TestKimiLinearSpec(unittest.TestCase):
    """Exercise unbounded nested forget gates and ungated NoPE MLA in the same native model."""

    def test_hybrid_values_and_gradients(self) -> None:
        """Feature: Kimi Linear default public families.
        Description: Preserve full linear/full-attention and dense/sparse roles with source FP32/BF16 state.
        Expectation: Source keys and parameter identities remain intact; logits and all gradients match.
        """
        torch.set_num_threads(2)
        config = KimiLinearConfig(
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            vocab_size=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_local_experts=4,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            layer_types=["linear_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse"],
            linear_head_dim=8,
            linear_num_heads=2,
        )
        config._attn_implementation = "eager"
        config.use_cache = False
        for dtype, q_rank in ((torch.float32, 8), (torch.bfloat16, 8),
                              (torch.float32, None), (torch.bfloat16, None)):
            config.q_lora_rank = q_rank
            with self.subTest(dtype=dtype):
                source = KimiLinearForCausalLM(config).to(dtype).eval()
                adopted = copy.deepcopy(source)
                identities = dict(adopted.named_parameters())
                target, _ = prepare_model_components(
                    adopted, build_module_context(model_spec=MODEL_SPEC, config=config)
                )
                self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
                self.assertEqual(tuple(target.state_dict()), tuple(source.state_dict()))
                for name, parameter in identities.items():
                    self.assertIs(parameter, target.get_parameter(name))
                self.assertTrue(target.model.layers[1].self_attn.use_nope)
                self.assertFalse(target.model.layers[1].self_attn.fuse_latent_qkv)
                ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
                mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
                expected = source(input_ids=ids, attention_mask=mask, use_cache=False).logits
                actual = target(input_ids=ids, attention_mask=mask, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=1e-5 if dtype == torch.float32 else 2e-2,
                                           atol=1e-6 if dtype == torch.float32 else 2e-3)
                expected.float().square().mean().backward()
                actual.float().square().mean().backward()
                reference = dict(source.named_parameters())
                for name, parameter in target.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        reference[name].grad,
                        rtol=1e-4 if dtype == torch.float32 else 3e-2,
                        atol=1e-6 if dtype == torch.float32 else 3e-3,
                        msg=lambda message, key=name: key + ": " + message,
                    )
