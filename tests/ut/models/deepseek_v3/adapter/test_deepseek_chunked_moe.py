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
"""DeepSeek routing, complete gradients and source checkpoint round trip."""
import copy
import unittest

import torch
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.models.deepseek_v3.adapter.conversion.chunked_moe import DeepseekV3ChunkMoE
from hyper_parallel.models.replacement import (
    ModuleReplacementRequest, apply_module_replacements, compile_module_replacements,
)


class TestDeepseekChunkMoE(unittest.TestCase):
    """Use the independent HF source for forward, routing and all gradient checks."""

    @staticmethod
    def _source(normalize=True):
        config = DeepseekV3Config.from_dict({
            "hidden_size": 16, "intermediate_size": 32, "moe_intermediate_size": 24, "n_routed_experts": 8,
            "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
            "norm_topk_prob": normalize, "routed_scaling_factor": 1.7,
        })
        module = DeepseekV3MoE(config)
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        module.gate.e_score_correction_bias.copy_(torch.linspace(-0.1, 0.2, 8))
        return module

    @staticmethod
    def _replace(source, chunk_size):
        model = torch.nn.Module()
        model.config = source.config
        model.mlp = source
        request = ModuleReplacementRequest(match=("mlp",), target=DeepseekV3ChunkMoE,
                                           kwargs={"chunk_size": chunk_size}, module_type=DeepseekV3MoE)
        plan = compile_module_replacements(model, overrides=(request,))
        model, mappings = apply_module_replacements(model, plan, weights_mapping=[])
        model._weight_conversions = mappings
        return model

    def test_routing_and_complete_gradients(self):
        """
        Feature: routing and complete gradients
        Description: Nonzero bias, group top2 and optional normalization match the source.
        Expectation: The documented output, gradient and state assertions pass.
        """
        torch.manual_seed(913)
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                source = self._source(normalize)
                candidate = self._replace(copy.deepcopy(source), 5)
                x = torch.randn(2, 7, 16, requires_grad=True)
                other = x.detach().clone().requires_grad_()
                cotangent = torch.randn_like(x)
                ids, scores = source.route_tokens_to_experts(source.gate(x))
                actual_scores, actual_ids, _ = candidate.mlp.router(x.reshape(-1, 16), candidate.mlp.expert_bias)
                torch.testing.assert_close(actual_ids, ids)
                torch.testing.assert_close(actual_scores, scores)
                expected = source(x)
                actual = candidate.mlp(other)
                expected.backward(cotangent)
                actual.backward(cotangent)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(other.grad, x.grad, rtol=2e-5, atol=2e-6)
                gradients = revert_weight_conversion(
                    candidate, {name: p.grad for name, p in candidate.named_parameters()},
                )
                for name, parameter in source.named_parameters():
                    torch.testing.assert_close(gradients["mlp." + name], parameter.grad,
                                               rtol=2e-5, atol=2e-6, msg=name)

    def test_replacement_checkpoint_and_baseline(self):
        """
        Feature: replacement checkpoint and baseline
        Description: Production replacement preserves source checkpoint tensors and a selectable baseline.
        Expectation: The documented output, gradient and state assertions pass.
        """
        source = self._source()
        expected = {"mlp." + name: value.clone() for name, value in source.state_dict().items()}
        candidate = self._replace(source, None)
        restored = revert_weight_conversion(candidate, dict(candidate.state_dict()))
        self.assertEqual(set(restored), set(expected))
        for name, value in expected.items():
            torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
        self.assertIs(candidate.mlp.router.gate.weight, source.gate.weight)
        self.assertIs(candidate.mlp.experts.w2, source.experts.down_proj)
        self.assertIs(candidate.mlp.shared_expert.w1.weight, source.shared_experts.gate_proj.weight)
        x = torch.randn(1, 11, 16)
        whole = candidate.mlp(x)
        candidate.mlp.chunk_size = 4
        torch.testing.assert_close(candidate.mlp(x), whole, rtol=2e-5, atol=2e-6)
