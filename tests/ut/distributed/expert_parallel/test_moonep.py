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

"""CPU numerical and lifecycle tests for collective MoonEP."""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import torch
from torch.nn import functional

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.activation_memory.api import checkpoint
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.distributed.expert_parallel import moonep, moonep_dedup
from hyper_parallel.distributed.expert_parallel.moonep_planner import compile_plan, select_allocation
from hyper_parallel.distributed.expert_parallel.moonep_runtime import (
    MoonEPExecution, MoonEPTransport, checkpoint_execution,
)


def _cpu_gmm(inputs, weights, boundaries, *, weight_grad=False):
    """Use independent dense matrix products in place of the optional NPU kernel."""
    start = 0
    outputs = []
    for expert, end in enumerate(boundaries.tolist()):
        outputs.append(inputs[:, start:end] @ weights[start:end] if weight_grad
                       else inputs[start:end] @ weights[expert])
        start = end
    return torch.stack(outputs) if weight_grad else torch.cat(outputs)


def _gate(value):
    gate, up = value.chunk(2, dim=-1)
    return functional.silu(gate) * up


def _copy_exchange(output, inputs, *args, **kwargs):
    del args, kwargs
    output.copy_(inputs)


def _identity_exchange(inputs, *args):
    del args
    return inputs.clone()


class TestMoonEPPlanner(unittest.TestCase):
    """Validate conservation, balancing and hard receiver replica capacity."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_default_capacity_and_fallback(self) -> None:
        """
        Feature: Replica capacity planning.
        Description: Run the whole-deficit fallback on a partial-fill counterexample.
        Expectation: Allocation conserves counts, balances ranks and respects receiver slots.
        """
        totals = np.array([0, 0, 4, 4, 4, 4, 4, 4])
        counts = np.zeros((4, 8), dtype=np.int64)
        counts[0] = totals
        allocation = select_allocation(counts)
        np.testing.assert_array_equal(allocation.sum(1), totals)
        np.testing.assert_array_equal(allocation.sum(0), np.full(4, 6))
        remote = np.arange(8)[:, None] // 2 != np.arange(4)
        self.assertLessEqual(int(((allocation > 0) & remote).sum(0).max()), 2)
        self.assertEqual(int(allocation[remote].sum()), 12)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_zero_capacity_and_invalid_values(self) -> None:
        """
        Feature: Replica capacity validation.
        Description: Exercise balanced and skewed routes with zero or invalid capacities.
        Expectation: Balanced home execution succeeds and infeasible or invalid inputs fail.
        """
        counts = np.ones((2, 4), dtype=np.int64)
        allocation = select_allocation(counts, 0)
        self.assertEqual(int(allocation[:2, 1].sum() + allocation[2:, 0].sum()), 0)
        counts[:, 2:] = 0
        with self.assertRaisesRegex(RuntimeError, "capacity 0"):
            select_allocation(counts, 0)
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_allocation(counts, invalid)


class TestMoonEPExecution(unittest.TestCase):
    """Compare deduplicated expert outputs and all gradients with a dense oracle."""

    def _parity(self, input_major, recompute):
        torch.manual_seed(17)
        ids = torch.tensor([[0, 1], [1, 2], [2, 0]])
        counts = np.bincount(ids.numpy().reshape(-1), minlength=4).reshape(1, 4)
        plan = compile_plan(counts, select_allocation(counts), ids.flatten(), 0)
        transport = SimpleNamespace(group=None, gradient_group=None, weight_group=None, trace=[],
                                    prefetch_previous=Mock(), stream_context=lambda **kwargs: nullcontext())
        hidden = torch.randn(3, 4, requires_grad=True)
        scores = torch.randn(3, 2).softmax(-1).requires_grad_()
        gate = torch.randn(4, 4, 6)
        down = torch.randn(4, 3, 4)
        if not input_major:
            gate, down = gate.transpose(1, 2).contiguous(), down.transpose(1, 2).contiguous()
        gate.requires_grad_()
        down.requires_grad_()
        expected_inputs = [value.detach().clone().requires_grad_() for value in (hidden, scores, gate, down)]
        source, probabilities, ref_gate, ref_down = expected_inputs
        if not input_major:
            ref_gate, ref_down = ref_gate.transpose(1, 2), ref_down.transpose(1, 2)
        expected = torch.stack([
            sum((_gate(source[token] @ ref_gate[expert]) @ ref_down[expert]) * probabilities[token, choice]
                for choice, expert in enumerate(ids[token]))
            for token in range(3)
        ])
        owner, created = object(), []

        def create() -> MoonEPExecution:
            """Compile the compressed route for one original invocation."""
            dedup = moonep_dedup.compile_token_dedup(plan, 2, None)
            execution = MoonEPExecution(plan, transport, dedup)
            created.append(execution)
            return execution

        def forward(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """Evaluate the retained expert invocation.

            Args:
                value: Source hidden vectors.
                weights: Differentiable router probabilities.
            """
            execution = checkpoint_execution(owner, create)
            return moonep.run_moonep_experts(value, execution, weights, lambda: (gate, down), _gate, input_major)

        with ExitStack() as stack:
            stack.enter_context(patch.object(moonep, "_gmm", side_effect=_cpu_gmm))
            stack.enter_context(patch.object(moonep, "ep_all_to_all", side_effect=_identity_exchange))
            stack.enter_context(patch.object(moonep_dedup, "ep_all_to_all", side_effect=_identity_exchange))
            stack.enter_context(patch.object(moonep_dedup.dist, "all_to_all_single", side_effect=_copy_exchange))
            actual = checkpoint(forward, hidden, scores, use_reentrant=False) if recompute else forward(hidden, scores)
            upstream = torch.randn_like(actual)
            (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(actual, expected)
        for actual_input, expected_input in zip((hidden, scores, gate, down), expected_inputs):
            torch.testing.assert_close(actual_input.grad, expected_input.grad)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].dedup.send, [3])
        self.assertTrue(created[0].complete)
        self.assertIsNone(created[0].incoming)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_dedup_gradients_and_checkpoint_replay(self) -> None:
        """
        Feature: Deduplicated expert numerical parity.
        Description: Compare both weight orientations with and without checkpoint replay.
        Expectation: Outputs and router, input and parameter gradients match dense computation.
        """
        for input_major in (False, True):
            for recompute in (False, True):
                with self.subTest(input_major=input_major, recompute=recompute):
                    self._parity(input_major, recompute)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_checkpoint_invocations_do_not_share_plans(self) -> None:
        """
        Feature: Checkpoint invocation isolation.
        Description: Replay repeated expert calls from two outstanding checkpoint invocations.
        Expectation: Each replay consumes the matching original plans in call order.
        """
        owner = object()
        transport = SimpleNamespace(trace=[])
        factory = Mock(side_effect=lambda: SimpleNamespace(transport=transport))
        contexts = [create_recompute_contexts() for _ in range(2)]
        originals = []
        for forward, _ in contexts:
            with forward, torch.no_grad():
                originals.append([checkpoint_execution(owner, factory), checkpoint_execution(owner, factory)])
        for index in (1, 0):
            with contexts[index][1]:
                for expected in originals[index]:
                    self.assertIs(checkpoint_execution(owner, factory), expected)
        self.assertEqual(factory.call_count, 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_prefetch_depth_and_fsdp_boundary(self) -> None:
        """
        Feature: Standalone backward prefetch.
        Description: Vary depth over earlier live calls and enter an FSDP-managed boundary.
        Expectation: Prefetch respects depth and does not read parameters from other FSDP units.
        """
        executions = [SimpleNamespace(complete=False, fsdp=False, prefetch=Mock()) for _ in range(4)]
        # Mock supports weak references, unlike SimpleNamespace.
        calls = [Mock(wraps=item, complete=False, fsdp=False, prefetch=item.prefetch) for item in executions]
        transport = object.__new__(MoonEPTransport)
        transport.trace = [weakref.ref(item) for item in calls]
        for depth in (0, 1, 2, 4):
            for item in calls:
                item.prefetch.reset_mock()
            calls[-1].prefetch_depth = depth
            transport.prefetch_previous(calls[-1])
            self.assertEqual(sum(item.prefetch.call_count for item in calls), min(depth, 3))
        calls[-1].fsdp = True
        for item in calls:
            item.prefetch.reset_mock()
        transport.prefetch_previous(calls[-1])
        self.assertEqual(sum(item.prefetch.call_count for item in calls), 0)
