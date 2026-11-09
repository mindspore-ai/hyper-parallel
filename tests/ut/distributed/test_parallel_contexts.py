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
"""Local protocol math and resource binding without distributed initialization."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from tests.common.mark_utils import arg_mark


class TestParallelContexts(unittest.TestCase):
    """Exercise per-call ownership, gradients and zero-slot semantics on CPU."""

    def _mesh(self):
        return SimpleNamespace(ndim=1, size=lambda: 1, get_local_rank=lambda: 0,
                               get_group=Mock(side_effect=AssertionError("local execution must not communicate")))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_cp_local_is_identity_and_pending_has_one_consumer(self):
        """Feature: Local CP transport contract.
        Description: Gather a local sequence and consume the returned pending result.
        Expectation: Tensor identity and gradients are preserved without communication; a second wait is rejected.
        """
        context = CPContext(self._mesh())
        tensor = torch.randn(1, 3, 2, requires_grad=True)
        pending = context.launch_gather_sequence(tensor)
        result, partition = pending.wait()
        self.assertIs(result, tensor)
        self.assertEqual((partition.offset, partition.global_length), (0, 3))
        with self.assertRaisesRegex(RuntimeError, "twice"):
            pending.wait()
        result.sum().backward()
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ep_probabilities_have_gradients_and_zero_slots_are_executed(self):
        """Feature: Expert-slot probability semantics.
        Description: Dispatch selected slots, compute weighted outputs and combine them through the receipt.
        Expectation: Zero-probability slots execute, input/probability gradients match and receipts cannot be reused.
        """
        context = EPContext(self._mesh(), num_experts=2)
        tokens = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        ids = torch.tensor([[1, 0], [1, 1]])
        weights = torch.tensor([[0., 0.7], [0.2, 0.4]], requires_grad=True)
        batch = context.dispatch(tokens, ids, weights)
        self.assertEqual(batch.hidden_states.shape[0], 4)
        output = context.combine(batch.hidden_states * batch.routing_weights[:, None], batch.receipt)
        expected = tokens * weights.sum(dim=1, keepdim=True)
        torch.testing.assert_close(output, expected)
        actual_grad = torch.autograd.grad(output.sum(), (tokens, weights), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.sum(), (tokens, weights))
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference)
        with self.assertRaisesRegex(ValueError, "consumed"):
            context.combine(batch.hidden_states, batch.receipt)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ep_empty_tokens_and_invalid_receipts(self):
        """Feature: Empty expert batches and receipt ownership.
        Description: Dispatch an empty batch and attempt cross-context receipt consumption and invalid expert IDs.
        Expectation: Empty gradients retain their shapes while invalid ownership and IDs fail explicitly.
        """
        context = EPContext(self._mesh(), num_experts=2)
        tokens = torch.empty(0, 3, requires_grad=True)
        batch = context.dispatch(tokens, torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2, requires_grad=True))
        other = EPContext(self._mesh(), num_experts=2)
        with self.assertRaisesRegex(ValueError, "another context"):
            other.combine(batch.hidden_states, batch.receipt)
        output = context.combine(batch.hidden_states, batch.receipt)
        self.assertEqual(output.shape, (0, 3))
        output.sum().backward()
        self.assertEqual(tokens.grad.shape, tokens.shape)
        with self.assertRaisesRegex(ValueError, "expert ID"):
            context.dispatch(torch.ones(1, 3), torch.tensor([[-1]]), torch.ones(1, 1))


    def test_ep_interleaved_dispatches_keep_independent_receipts(self):
        """Two outstanding calls may combine in reverse order without sharing routing state."""
        context = EPContext(self._mesh(), num_experts=2)
        first = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        second = torch.tensor([[5., 6.]], requires_grad=True)
        first_weights = torch.tensor([[0.2, 0.7], [0.4, 0.3]], requires_grad=True)
        second_weights = torch.tensor([[0.8, 0.1]], requires_grad=True)
        one = context.dispatch(first, torch.tensor([[1, 0], [0, 1]]), first_weights)
        two = context.dispatch(second, torch.tensor([[1, 0]]), second_weights)
        result_two = context.combine(two.hidden_states * two.routing_weights[:, None], two.receipt)
        result_one = context.combine(one.hidden_states * one.routing_weights[:, None], one.receipt)
        expected_one = first * first_weights.sum(dim=1, keepdim=True)
        expected_two = second * second_weights.sum(dim=1, keepdim=True)
        torch.testing.assert_close(result_one, expected_one)
        torch.testing.assert_close(result_two, expected_two)
        actual = torch.autograd.grad(result_one.sum() + result_two.sum(),
                                     (first, second, first_weights, second_weights), retain_graph=True)
        expected = torch.autograd.grad(expected_one.sum() + expected_two.sum(),
                                       (first, second, first_weights, second_weights))
        for gradient, reference in zip(actual, expected):
            torch.testing.assert_close(gradient, reference)
        with self.assertRaisesRegex(ValueError, "consumed"):
            context.combine(one.hidden_states, one.receipt)
