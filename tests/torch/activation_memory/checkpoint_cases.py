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
"""NPU system-test cases for Hyper's Torch non-reentrant checkpoint implementation."""
# pylint: disable=missing-public-docstring,missing-public-type-hints,wrong-import-position,protected-access
import unittest

import torch

from hyper_parallel.core.activation_memory.checkpoint import (
    checkpoint,
    clear_recompute_session,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
)
from hyper_parallel.core.activation_memory import CheckpointPolicy, checkpoint as core_checkpoint
from tests.torch.activation_memory.utils import set_seed

_NPU_DEVICE = "npu"


def _randn(*size: int, **kwargs) -> torch.Tensor:
    """Create a random tensor explicitly on NPU."""
    return torch.randn(*size, device=_NPU_DEVICE, **kwargs)


def _ones(*size: int, **kwargs) -> torch.Tensor:
    """Create an all-ones tensor explicitly on NPU."""
    return torch.ones(*size, device=_NPU_DEVICE, **kwargs)


class TestCheckpoint(unittest.TestCase):
    """Validate checkpoint gradient parity and RNG restoration on Ascend."""

    def test_backward_matches_non_checkpointed_function(self):
        """Checkpoint gradients should match regular autograd for inputs and weights."""
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()
        baseline_weight = weight.detach().clone().requires_grad_()

        baseline = torch.sin(baseline_input * baseline_weight)
        baseline.sum().backward()

        output = checkpoint(lambda x, w: torch.sin(x * w), input_tensor, weight)
        output.sum().backward()

        self.assertTrue(torch.allclose(input_tensor.grad, baseline_input.grad))
        self.assertTrue(torch.allclose(weight.grad, baseline_weight.grad))

    def test_preserve_rng_state_matches_regular_autograd(self):
        """Random masks should be restored when preserve_rng_state is enabled."""
        baseline_input = _ones(32, requires_grad=True)
        checkpoint_input = baseline_input.detach().clone().requires_grad_()

        set_seed(7)
        baseline_output = torch.nn.functional.dropout(baseline_input, p=0.5, training=True)
        baseline_output.sum().backward()

        set_seed(7)
        output = checkpoint(
            lambda x: torch.nn.functional.dropout(x, p=0.5, training=True),
            checkpoint_input,
            preserve_rng_state=True,
        )
        output.sum().backward()

        self.assertTrue(torch.equal(output, baseline_output))
        self.assertTrue(torch.equal(checkpoint_input.grad, baseline_input.grad))

    def test_grad_inside_forward_matches_regular_autograd(self):
        """Checkpoint should support an unpack triggered by grad inside forward."""
        input_tensor = _randn(4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()

        def function(value):
            intermediate = value.sin()
            inner_grad = torch.autograd.grad(intermediate.sum(), value, create_graph=True)[0]
            return intermediate * inner_grad

        baseline_output = function(baseline_input)
        baseline_output.sum().backward()
        output = checkpoint(function, input_tensor)
        output.sum().backward()

        self.assertTrue(torch.allclose(output, baseline_output))
        self.assertTrue(torch.allclose(input_tensor.grad, baseline_input.grad))


class TestScheduledRecomputation(unittest.TestCase):
    """Validate early scheduling and dx/dw separated backward behavior."""

    def test_separate_dx_dw_without_session_recomputes_per_graph_task(self):
        """Native GraphTask keys should avoid unpack loss across dx and dw calls."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        output = checkpoint(function, input_tensor, weight)
        torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
        torch.autograd.grad(output.sum(), weight)

        self.assertEqual(len(calls), 3)

    def test_prefired_session_is_shared_by_dx_and_dw(self):
        """A retained session should reuse one prefired recomputation for dx and dw."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()
        baseline_weight = weight.detach().clone().requires_grad_()

        baseline = torch.nn.functional.gelu(baseline_input @ baseline_weight)
        expected_dx = torch.autograd.grad(baseline.sum(), baseline_input, retain_graph=True)[0]
        expected_dw = torch.autograd.grad(baseline.sum(), baseline_weight)[0]

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        with recompute_handle_collector_ctx() as handles:
            output = checkpoint(function, input_tensor, weight)

        session_id = ("micro-batch", 0)
        try:
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                recompute_handle(handles[0], session_id)
                actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
                actual_dw = torch.autograd.grad(output.sum(), weight)[0]
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(handles), 1)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.allclose(actual_dx, expected_dx))
        self.assertTrue(torch.allclose(actual_dw, expected_dw))

    def test_one_session_serves_multiple_checkpoint_frames(self):
        """One session should prefire and serve multiple sequential checkpoint frames."""
        input_tensor = _randn(4, 4, requires_grad=True)
        first_weight = _randn(4, 6, requires_grad=True)
        second_weight = _randn(6, 3, requires_grad=True)
        reference_input = input_tensor.detach().clone().requires_grad_()
        reference_first_weight = first_weight.detach().clone().requires_grad_()
        reference_second_weight = second_weight.detach().clone().requires_grad_()
        reference_hidden = torch.nn.functional.gelu(reference_input @ reference_first_weight)
        reference_output = torch.nn.functional.silu(reference_hidden @ reference_second_weight)
        expected_grads = torch.autograd.grad(
            reference_output.sum(),
            (reference_input, reference_first_weight, reference_second_weight),
        )
        calls = [0, 0]

        def first_block(current_input, current_weight):
            calls[0] += 1
            return torch.nn.functional.gelu(current_input @ current_weight)

        def second_block(current_input, current_weight):
            calls[1] += 1
            return torch.nn.functional.silu(current_input @ current_weight)

        with recompute_handle_collector_ctx() as handles:
            hidden = checkpoint(first_block, input_tensor, first_weight)
            output = checkpoint(second_block, hidden, second_weight)

        session_id = ("multiple-frames", id(output))
        try:
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                for handle in handles:
                    recompute_handle(handle, session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                actual_dx = torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)[0]
            with recompute_session_ctx(session_id, retain_on_unpack=False):
                actual_dws = torch.autograd.grad(output.sum(), (first_weight, second_weight))
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(handles), 2)
        self.assertEqual(calls, [2, 2])
        self.assertTrue(torch.allclose(actual_dx, expected_grads[0]))
        self.assertTrue(torch.allclose(actual_dws[0], expected_grads[1]))
        self.assertTrue(torch.allclose(actual_dws[1], expected_grads[2]))

    def test_nested_checkpoint_without_session_still_works(self):
        """Ordinary GraphTask-based recomputation should preserve native nested behavior."""
        input_tensor = _randn(4, requires_grad=True)
        baseline_input = input_tensor.detach().clone().requires_grad_()

        def inner(value):
            return value.sin()

        def outer(value):
            return checkpoint(inner, value).cos()

        expected = baseline_input.sin().cos()
        expected_grad = torch.autograd.grad(expected.sum(), baseline_input)[0]
        output = checkpoint(outer, input_tensor)
        actual_grad = torch.autograd.grad(output.sum(), input_tensor)[0]

        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(torch.allclose(actual_grad, expected_grad))

    def test_prefired_session_consumes_sac_cache_only_once(self):
        """SAC replay should happen during prefire, not again for dx and dw consumers."""
        calls = []
        input_tensor = _randn(4, 4, requires_grad=True)
        weight = _randn(4, 4, requires_grad=True)

        def function(x, w):
            calls.append(None)
            return torch.nn.functional.gelu(x @ w)

        def policy_fn(ctx, op, *args, **kwargs):
            del ctx, op, args, kwargs
            return CheckpointPolicy.MUST_SAVE

        with recompute_handle_collector_ctx() as handles:
            output = core_checkpoint(function, input_tensor, weight, policy_fn=policy_fn)

        session_id = "sac-split-backward"
        try:
            recompute_handle(handles[0], session_id)
            with recompute_session_ctx(session_id, retain_on_unpack=True):
                torch.autograd.grad(output.sum(), input_tensor, retain_graph=True)
                torch.autograd.grad(output.sum(), weight)
        finally:
            clear_recompute_session(session_id)

        self.assertEqual(len(calls), 2)
