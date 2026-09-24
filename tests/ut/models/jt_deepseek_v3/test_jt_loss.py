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
"""Model-owned weighting around public cross entropy, without custom backward."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import masked_vocab_parallel_loss
from tests.common.mark_utils import arg_mark


class TestJTLoss(unittest.TestCase):
    """Check supervision, local ownership and public parallel dispatch."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_loss_and_gradient(self):
        """Feature: Masked objective.

        Description: Include ignored targets, fractional weights and a masked valid target.
        Expectation: Public CE plus JT weighting matches independent dense autograd.
        """
        torch.manual_seed(12)
        values = torch.randn(1, 5, 7, requires_grad=True)
        oracle = values.detach().clone().requires_grad_()
        labels = torch.tensor([[1, 2, -100, 4, 6]])
        mask = torch.tensor([[0.25, 0., 0., 1.5, 1.]])
        actual = masked_vocab_parallel_loss(values, labels, mask, vocab_size=7)
        actual.backward()
        expected = (F.cross_entropy(oracle.flatten(0, 1), labels.flatten(), reduction="none") * mask.flatten()).sum()
        expected = expected / (mask.sum() + 1e-8)
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(values.grad, oracle.grad)
        self.assertEqual(torch.count_nonzero(values.grad[:, 1:3]).item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_all_masked_is_zero(self):
        """Feature: Empty supervision.

        Description: Mask every target.
        Expectation: Loss and logit gradients are finite zeros.
        """
        values = torch.randn(1, 2, 4, requires_grad=True)
        loss = masked_vocab_parallel_loss(values, torch.full((1, 2), -100), torch.zeros(1, 2), vocab_size=4)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(values.grad.abs().sum().item(), 0.)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_public_parallel_dispatch_and_missing_mesh(self):
        """Feature: Framework CE ownership.

        Description: Supply a public mesh and then omit it for incomplete logits.
        Expectation: Parallel CE receives the logical vocabulary and missing mesh fails clearly.
        """
        prefix = "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3."
        values, labels, mask = torch.randn(1, 2, 4), torch.tensor([[1, 5]]), torch.ones(1, 2)
        mesh = object()
        with patch(prefix + "_get_loss_parallel_mesh", return_value=mesh), \
                patch(prefix + "vocab_parallel_cross_entropy_local", return_value=torch.tensor([2., 4.])) as ce:
            result = masked_vocab_parallel_loss(values, labels, mask, vocab_size=8)
        self.assertEqual(result.item(), 3.)
        self.assertIs(ce.call_args.kwargs["mesh"], mesh)
        self.assertEqual(ce.call_args.kwargs["reduction"], "none")
        with self.assertRaisesRegex(ValueError, "loss_parallel context"):
            masked_vocab_parallel_loss(values, labels, mask, vocab_size=8)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unbound_loss_never_uses_the_default_process_group(self):
        """Feature: Local standalone model.

        Description: Run full-vocabulary CE while an unrelated default group exists.
        Expectation: Neither training nor evaluation queries that group.
        """
        labels, mask = torch.tensor([[1, 2]]), torch.ones(1, 2)
        with patch("torch.distributed.all_reduce", side_effect=AssertionError), \
                patch("torch.distributed.get_rank", side_effect=AssertionError), \
                patch("torch.distributed.get_world_size", side_effect=AssertionError):
            values = torch.randn(1, 2, 4, requires_grad=True)
            loss = masked_vocab_parallel_loss(values, labels, mask, vocab_size=4)
            loss.backward()
            with torch.no_grad():
                evaluated = masked_vocab_parallel_loss(values, labels, mask, vocab_size=4)
        torch.testing.assert_close(loss, evaluated)
        self.assertTrue(torch.isfinite(values.grad).all())
