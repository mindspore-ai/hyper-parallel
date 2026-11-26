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
"""Independent terminal-family, router objective and HF orchestration parity."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.common.causal_lm_head import CausalLMHead


class TestCausalLMHead(unittest.TestCase):
    """Verify full logits, chunked gradients and source ownership."""

    def test_head_preserves_weight_and_matches_dense_ce(self) -> None:
        """Feature: Terminal projection family.
        Description: Adopt a source weight and compute masked, pre-shifted CE in small chunks.
        Expectation: Loss and all input/weight gradients match the independent dense expression.
        """
        source = nn.Linear(8, 19, bias=False)
        target = CausalLMHead(module=source)
        self.assertIs(source.weight, target.weight)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        labels = torch.randint(19, (2, 5))
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[0, 0] = False
        dense = F.cross_entropy(
            # Torch exposes this runtime callable through a native or generated descriptor.
            F.linear(hidden, source.weight).flatten(0, 1),  # pylint: disable=not-callable
            labels.masked_fill(~mask, -100).flatten(),
            reduction="sum",
        )
        expected = torch.autograd.grad(dense, (hidden, source.weight))
        result = target(hidden, targets=labels, loss_mask=mask, chunk_size=2)
        actual = torch.autograd.grad(result.loss_sum, (hidden, target.weight))
        torch.testing.assert_close(result.loss_sum, dense)
        self.assertEqual(result.valid_token_count.item(), 9)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(target(input=hidden), source(hidden), rtol=0, atol=0)
