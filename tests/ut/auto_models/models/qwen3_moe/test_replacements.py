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
"""Public family adoption of actual HF packed Qwen3 experts."""
# Transformers @strict supplies the supported keyword constructor at runtime.
# pylint: disable=unexpected-keyword-arg

import copy

import torch
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from hyper_parallel.components.modules import GroupedExperts


def test_actual_packed_source_retains_parameters_and_full_gradients() -> None:
    """Preserve HF's expert boundary and independently compare its full differentiation."""
    torch.set_num_threads(2)
    config = Qwen3MoeConfig(hidden_size=8, moe_intermediate_size=16, num_experts=4,
                           num_experts_per_tok=2, num_hidden_layers=2)
    source = Qwen3MoeExperts(config)
    # HF initializes these bare tensors only through the enclosing pretrained model.
    for parameter in source.parameters():
        nn.init.normal_(parameter, std=0.02)
    reference = copy.deepcopy(source)
    target = GroupedExperts(module=source, context={"num_layers": 2}, use_grouped_gemm=False)
    assert target.gate_up_proj is source.gate_up_proj
    assert target.down_proj is source.down_proj
    hidden = torch.randn(6, 8, requires_grad=True)
    ids = torch.tensor([[0, 2], [1, 3], [2, 1], [0, 3], [1, 2], [3, 0]])
    scores = torch.rand(6, 2, requires_grad=True)
    actual = target(hidden, ids, scores)
    expected = reference(hidden, ids, scores)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (hidden, scores, *target.parameters()))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (hidden, scores, *reference.parameters()))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7)
