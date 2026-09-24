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
"""Masked loss gradients and complete-sequence coverage without accelerator setup."""

import math
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as loss
from tests.common.mark_utils import arg_mark


class TestVocabularyCrossEntropy(unittest.TestCase):
    """Check an analytic uniform distribution and ignored-token semantics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_masked_uniform_loss_and_gradient(self) -> None:
        """Feature: Aligned CE.

        Description: Exercise uniform logits with masked targets.
        Expectation: The analytic weighted gradient matches the result.
        """
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[2, -100, 0]])
        mask = torch.tensor([[1.0, 0.0, 0.5]])
        with patch.object(loss, "dist") as communication:
            communication.get_rank.return_value = 0
            communication.get_world_size.return_value = 1
            result = loss.masked_vocab_parallel_loss(logits, labels, mask, vocab_size=logits.shape[-1])
            result.backward()
        expected = torch.tensor([[[1 / 6, 1 / 6, -1 / 2, 1 / 6],
                                  [0, 0, 0, 0], [-1 / 4, 1 / 12, 1 / 12, 1 / 12]]])
        self.assertAlmostEqual(result.item(), math.log(4), places=6)
        torch.testing.assert_close(logits.grad, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_all_ignored_tokens_are_finite(self) -> None:
        """Feature: Aligned CE.

        Description: Mask every target.
        Expectation: Loss and gradients are finite zeros.
        """
        logits = torch.arange(12.0).reshape(1, 3, 4).requires_grad_()
        with patch.object(loss, "dist") as communication:
            communication.get_rank.return_value = 0
            communication.get_world_size.return_value = 1
            result = loss.masked_vocab_parallel_loss(
                logits, torch.full((1, 3), -100), torch.zeros(1, 3), vocab_size=4)
            result.backward()
        self.assertEqual(result.item(), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_full_length_sum_keeps_last_token_and_its_gradient(self) -> None:
        """Feature: Sequence reduction.

        Description: Reduce a full 256K sequence including the partial final tile.
        Expectation: Every token contributes one to the sum and gradient.
        """
        values = torch.ones(262144, requires_grad=True)
        result = loss.reference_sequence_sum(values)
        result.backward()
        self.assertEqual(result.item(), 262144)
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))
