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
"""Portable expert source parity, including sparse routes and routing-weight gradients."""

# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg


import copy
import unittest

import torch
from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.torch_grouped_experts import TorchGroupedExperts
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestTorchGroupedExperts(unittest.TestCase):
    """Verify actual family selection and independent HF numerical equivalence."""

    def test_source_parity(self) -> None:
        """
        Feature: Portable routed experts.
        Description: Adopt HF weights and compare outputs and all gradients for sparse and empty routes.
        Expectation: FP32 and BF16 mathematics retain HF expert and probability accumulation order.
        """
        for dtype in (torch.float32, torch.bfloat16):
            for tokens in (0, 7):
                with self.subTest(dtype=dtype, tokens=tokens):
                    torch.manual_seed(12)
                    config = Qwen2MoeConfig(
                        hidden_size=16,
                        moe_intermediate_size=12,
                        num_experts=4,
                        num_experts_per_tok=2,
                        num_hidden_layers=2,
                    )
                    source = Qwen2MoeExperts(config).to(dtype=dtype)
                    for parameter in source.parameters():
                        torch.nn.init.normal_(parameter, std=0.1)
                    target = GroupedExperts(
                        module=copy.deepcopy(source), use_grouped_gemm=False, context={"num_layers": 2}
                    )
                    self.assertIsInstance(target, TorchGroupedExperts)
                    inputs = torch.randn(tokens, 16, dtype=dtype, requires_grad=True)
                    inputs_copy = inputs.detach().clone().requires_grad_()
                    ids = torch.tensor([[0, 2], [2, 0], [0, 0], [2, 2], [2, 0], [0, 2], [0, 0]])[:tokens]
                    weights = torch.rand(tokens, 2, requires_grad=True)
                    weights_copy = weights.detach().clone().requires_grad_()
                    expected = source(inputs, ids, weights)
                    actual = target(inputs_copy, ids, weights_copy)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if tokens:
                        expected.float().square().sum().backward()
                        actual.float().square().sum().backward()
                        torch.testing.assert_close(inputs_copy.grad, inputs.grad, rtol=0, atol=0)
                        torch.testing.assert_close(weights_copy.grad, weights.grad, rtol=0, atol=0)
                        for original, adopted in zip(source.parameters(), target.parameters()):
                            torch.testing.assert_close(adopted.grad, original.grad, rtol=0, atol=0)
