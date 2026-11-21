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


def _permute_routed_slots(hidden_states, expert_indices, routing_weights):
    """Prepare selected routes in expert-major order for context-only tests."""
    top_k = expert_indices.shape[1]
    ids, order = expert_indices.reshape(-1).sort(stable=True)
    rows = torch.arange(order.numel(), device=order.device)
    source_inverse = torch.empty_like(order).scatter_(0, order, rows)
    hidden = hidden_states.index_select(0, order // top_k)
    weights = routing_weights.reshape(-1).index_select(0, order)
    return hidden, ids, weights, source_inverse


def _sum_routed_slots(values, source_inverse, top_k):
    """Restore source-route order and sum top-k outputs for context-only tests."""
    values = values.index_select(0, source_inverse)
    return values.reshape(-1, top_k, values.shape[-1]).sum(dim=1)


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
        Description: Dispatch selected slots, compute weighted outputs and restore the source tokens.
        Expectation: Zero-probability slots execute and token/probability gradients match the reference.
        """
        context = EPContext(self._mesh(), num_experts=2)
        tokens = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        ids = torch.tensor([[1, 0], [1, 1]])
        weights = torch.tensor([[0., 0.7], [0.2, 0.4]], requires_grad=True)
        slots, expert_ids, slot_weights, source_inverse = _permute_routed_slots(tokens, ids, weights)
        hidden, counts, local_weights, *transport = context.dispatch(slots, expert_ids, slot_weights)
        self.assertEqual(hidden.shape[0], 4)
        torch.testing.assert_close(counts, torch.tensor([1, 3]))
        restored = context.combine(hidden * local_weights[:, None], *transport)
        output = _sum_routed_slots(restored, source_inverse, ids.shape[1])
        expected = tokens * weights.sum(dim=1, keepdim=True)
        torch.testing.assert_close(output, expected)
        actual_grad = torch.autograd.grad(output.sum(), (tokens, weights), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.sum(), (tokens, weights))
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ep_empty_tokens_and_invalid_transport(self):
        """Feature: Empty expert batches and combine argument validation.
        Description: Dispatch an empty batch and combine it with its per-call split sizes and inverse mapping.
        Expectation: Empty gradients retain their shapes while an invalid mapping or expert ID is rejected.
        """
        context = EPContext(self._mesh(), num_experts=2)
        tokens = torch.empty(0, 3, requires_grad=True)
        hidden, _, _, send, recv, recv_inverse = context.dispatch(tokens, torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "recv_inverse"):
            context.combine(hidden, send, recv, recv_inverse.long())
        output = context.combine(hidden, send, recv, recv_inverse)
        self.assertEqual(output.shape, (0, 3))
        output.sum().backward()
        self.assertEqual(tokens.grad.shape, tokens.shape)
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            context.dispatch(torch.ones(1, 3), torch.tensor([-1]))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_ep_interleaved_dispatches_keep_independent_mappings(self):
        """Feature: Independent EP call metadata.
        Description: Dispatch two calls and combine them in reverse order.
        Expectation: Each call restores its own tokens and gradients without shared routing state.
        """
        context = EPContext(self._mesh(), num_experts=2)
        first = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        second = torch.tensor([[5., 6.]], requires_grad=True)
        first_weights = torch.tensor([[0.2, 0.7], [0.4, 0.3]], requires_grad=True)
        second_weights = torch.tensor([[0.8, 0.1]], requires_grad=True)
        first_ids, second_ids = torch.tensor([[1, 0], [0, 1]]), torch.tensor([[1, 0]])
        first_slots, first_experts, first_probs, first_inverse = _permute_routed_slots(
            first, first_ids, first_weights,
        )
        second_slots, second_experts, second_probs, second_inverse = _permute_routed_slots(
            second, second_ids, second_weights,
        )
        first_hidden, _, first_probs, *first_transport = context.dispatch(first_slots, first_experts, first_probs)
        second_hidden, _, second_probs, *second_transport = context.dispatch(second_slots, second_experts, second_probs)
        result_two = context.combine(second_hidden * second_probs[:, None], *second_transport)
        result_one = context.combine(first_hidden * first_probs[:, None], *first_transport)
        result_two = _sum_routed_slots(result_two, second_inverse, second_ids.shape[1])
        result_one = _sum_routed_slots(result_one, first_inverse, first_ids.shape[1])
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
