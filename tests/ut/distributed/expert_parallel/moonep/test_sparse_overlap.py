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
"""CPU event-order coverage for weight/token overlap in sparse autograd."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from threading import RLock
from types import MethodType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts import readiness, transport
from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from hyper_parallel.core.multicore.modules.routed_experts.frame import ExpertState
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingSpec, RingWorkspace
from hyper_parallel.distributed.expert_parallel.moonep.fused import autograd as sparse_autograd
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import ExecutionKey, SparseMoonEPRuntime
from tests.common.mark_utils import arg_mark


class _Stream:
    """Log stream dependencies without entering a device runtime."""

    def __init__(self, name: str, operations: list[tuple[Any, ...]]) -> None:
        """Bind a named stream to the shared operation log."""
        self.name, self.operations = name, operations

    def wait_event(self, event: _Event) -> None:
        """Record the enqueue order of a wait."""
        self.operations.append(("wait", self.name, event))


class _Event:
    """Record which fake stream initializes a fence."""

    def record(self, stream: _Stream) -> None:
        """Record one event in the shared enqueue log."""
        stream.operations.append(("record", stream.name, self))


class _Context:
    """Expose custom autograd saves while recording checkpoint-unpack position."""

    def __init__(self, operations: list[tuple[Any, ...]]) -> None:
        """Bind autograd saves to the shared operation log."""
        self.operations = operations
        self.saved = ()
        self.target_tokens = None
        self.group_metadata = None

    def save_for_backward(self, *values: torch.Tensor) -> None:
        """Retain CPU tensors exactly as the custom forward requests."""
        self.saved = values

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]:
        """Log the potential replay boundary before returning the saved state."""
        self.operations.append(("unpack",))
        return self.saved


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestSparseOverlap(unittest.TestCase):
    """Run real phase/transport preparation with only device compute mocked."""

    def setUp(self) -> None:
        """Install CPU tensors and an ordered fake C/W/T/G stream environment."""
        self.operations = []
        self.compute_stream = _Stream("C", self.operations)
        self.current = self.compute_stream
        npu = SimpleNamespace(Event=_Event, current_stream=lambda _device: self.current,
                              stream=self._stream, device=lambda _device: nullcontext())
        self.enterContext(patch.object(torch, "npu", npu, create=True))
        self.records = self.enterContext(patch.object(torch.Tensor, "record_stream"))
        self.enterContext(patch.object(transport, "_load_native"))
        self.enterContext(patch.object(transport, "vector_core_count", return_value=48))
        self.enterContext(patch.object(transport, "upload_metadata", side_effect=self._upload))
        self.enterContext(patch.object(readiness, "upload_metadata", side_effect=self._upload))
        self.enterContext(patch.object(torch.ops.hyper_parallel, "routed_experts_transport", create=True,
                                       side_effect=self._native_transport))
        self.enterContext(patch.object(sparse_autograd, "combine", side_effect=self._combine))
        self.enterContext(patch.object(sparse_autograd, "combine_gradient", side_effect=self._combine_gradient))
        self.replaying = self.enterContext(patch.object(sparse_autograd, "is_replaying", return_value=False))
        self.workspace = self._workspace()
        self.route = RouteDescriptor(
            0, 2, 3, 4, (0, 2), (2, 4),
            (RouteEdge(0, 0, 0, 0, 0, 1), RouteEdge(0, 1, 1, 1, 0, 2)),
            (RouteEdge(0, 0, 0, 0, 0, 1), RouteEdge(1, 0, 0, 0, 1, 1), RouteEdge(1, 0, 2, 1, 2, 2)))
        self.key = ExecutionKey(0, 1, 0)
        self.home = (torch.empty((2, 16, 32), dtype=torch.bfloat16),
                     torch.empty((2, 16, 16), dtype=torch.bfloat16))
        self.weight_ready = _Event()
        self.weight_ready.record(_Stream("W", self.operations))
        self.ticket = SimpleNamespace(ready=self.weight_ready, remote={
            (2, "gate"): torch.empty((16, 32), dtype=torch.bfloat16),
            (2, "down"): torch.empty((16, 16), dtype=torch.bfloat16)})
        self.expert = SimpleNamespace(top_k=1, home_count=2, rank=0, owner=SimpleNamespace(), token_dedup=False,
                                      forward_ratio=0.75, backward_ratio=1.0,
                                      read_weights=Mock(side_effect=self._read_weights))
        self.compute = SimpleNamespace(prepare_forward=self._prepare_forward, run_forward=self._run_forward,
                                       prepare_backward=self._prepare_backward, run_backward=self._run_backward,
                                       forward_pipeline=SimpleNamespace(enabled=False))
        self.plan = SimpleNamespace(route=self.route, order=torch.arange(3), inverse_order=torch.arange(3),
                                    compute=self.compute)
        self.runtime = SimpleNamespace(workspace=self.workspace, executions={self.key: (self.expert, self.plan)},
                                       prepare=Mock(return_value=self.ticket), arena=PhaseArena("cpu"),
                                       statistics=Counter(), poisoned=False, gradient_outputs={},
                                       gradient_accumulators={}, gradient_staging={},
                                       gradient_domain=SimpleNamespace(quota_elements=16), backward_layers=set(),
                                       release=Mock(side_effect=lambda _key: self.operations.append(("release",))),
                                       complete=Mock(), validate_backward=Mock(), start_gradient=self._start_gradient)
        self.runtime.consume = MethodType(SparseMoonEPRuntime.consume, self.runtime)
        self.hidden = torch.empty((3, 16), dtype=torch.bfloat16)
        self.scores = torch.empty((3, 1), dtype=torch.bfloat16)
        self.prepared = []
        self.enterContext(patch.object(sparse_autograd, "prepare_exchange", side_effect=self._prepare_exchange))

    @staticmethod
    def _upload(value, device):
        return value.to(device=device)

    @contextmanager
    def _stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    def _workspace(self):
        workspace = RingWorkspace.__new__(RingWorkspace)
        spec = RingSpec(2, 16, block_rows=2)
        workspace.__dict__.update(
            spec=spec, rank=0, device=torch.device("cpu"), closed=False, poisoned=False, closing=False,
            metadata={}, send_sequence=[[0, 0], [0, 0]], receive_sequence=[[0, 0], [0, 0]],
            data=torch.empty(spec.data_shape, dtype=torch.bfloat16),
            signals=torch.zeros(spec.signal_shape, dtype=torch.int32),
            headers=torch.empty(spec.header_shape, dtype=torch.int64),
            transport_stream=_Stream("T", self.operations), last_progress=None, _lock=RLock(), _completion=None)

        def _preflight(error):
            self.operations.append(("preflight",))
            if error is not None:
                raise RuntimeError("phase preflight failed") from error

        workspace.check_preflight = Mock(side_effect=_preflight)
        return workspace

    def _prepare_exchange(self, *args, **kwargs):
        prepared = transport.prepare_exchange(*args, **kwargs)
        self.prepared.append(prepared)
        return prepared

    def _native_transport(self, *_args):
        self.operations.append(("transport", self.current.name))

    def _read_weights(self):
        self.operations.append(("read_weights",))
        return self.home

    def _ready(self):
        return readiness.ExpertReadiness.create(torch.ones((1, 2), dtype=torch.int32), torch.device("cpu"))

    def _prepare_forward(self, x, _gate, _down, intermediate, *, allocator):
        self.operations.append(("prepare_forward",))
        return SimpleNamespace(output=allocator("project", tuple(x.shape), x.dtype), ready=self._ready(),
                               state=ExpertState(x, x.new_empty((x.shape[0], 2 * intermediate))))

    def _run_forward(self, work, _dependency):
        self.operations.append(("forward",))
        return work.output, work.state, work.ready

    def _prepare_backward(self, dv, state, _gate, _down, *, allocator):
        self.operations.append(("prepare_backward",))
        return SimpleNamespace(output=allocator("project", tuple(state.x.shape), dv.dtype), ready=self._ready())

    def _run_backward(self, work, publish, _dependency):
        self.operations.append(("backward",))
        for role, shape in (("down", (2, 16, 16)), ("gate", (2, 16, 32))):
            publish(role, torch.empty(shape, dtype=torch.float32))
        return work.output, work.ready

    def _start_gradient(self, key, role, _gradient):
        incoming = self.prepared[0].value
        accumulator = self.runtime.gradient_accumulators[key]
        self.assertEqual(incoming.untyped_storage().data_ptr(), accumulator.untyped_storage().data_ptr())
        self.assertEqual(accumulator.numel(), max(value.numel() for value in self.home))
        ready = _Event()
        ready.record(_Stream("G", self.operations))
        return SimpleNamespace(value=self.runtime.gradient_outputs[key][role], ready=ready)

    def _combine(self, *_args, output, **_kwargs):
        self.operations.append(("combine",))
        return output

    def _combine_gradient(self, dy, *_args, outputs):
        self.assertTrue(dy.is_contiguous())
        self.operations.append(("combine_gradient",))
        return outputs

    def _assert_overlap_order(self, compute_name):
        incoming = self.operations.index(("transport", "T"))
        weight_wait = self.operations.index(("wait", "C", self.weight_ready))
        compute = self.operations.index((compute_name,))
        self.assertLess(self.operations.index(("preflight",)), incoming)
        self.assertLess(incoming, weight_wait)
        self.assertLess(weight_wait, compute)
        for prepared in self.prepared:
            self.assertLess(self.operations.index(("record", "C", prepared.initialized)), incoming)
        self.assertLess(self.operations.index(("record", "C", self.prepared[0].source_admitted)), weight_wait)
        self.assertNotIn(("record", "C", self.prepared[1].source_admitted), self.operations)

    def test_consume_returns_fence_without_wait_and_binds_fresh_views(self):
        """Pending W contents do not prevent safe host pointer/view preparation."""
        first_gate, first_down, first_ready = self.runtime.consume(self.key)
        previous = self.home
        self.home = tuple(torch.empty_like(value) for value in previous)
        second_gate, second_down, second_ready = self.runtime.consume(self.key)

        self.assertIs(first_ready, self.weight_ready)
        self.assertIs(second_ready, self.weight_ready)
        self.assertIs(first_gate[1], self.ticket.remote[2, "gate"])
        self.assertIs(first_down[1], self.ticket.remote[2, "down"])
        self.assertEqual(second_gate[0].data_ptr(), self.home[0][0].data_ptr())
        self.assertEqual(second_down[0].data_ptr(), self.home[1][0].data_ptr())
        self.assertNotEqual(first_gate[0].data_ptr(), second_gate[0].data_ptr())
        self.assertNotIn(("wait", "C", self.weight_ready), self.operations)
        self.assertEqual(self.expert.read_weights.call_count, 2)
        self.assertEqual(self.records.call_count, 8)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_forward_submits_tokens_before_waiting_for_weights(self):
        """Feature: Block-ready forward preparation and weight/token overlap.

        Description: Compile route metadata after consuming weights and before preflight.
        Expectation: Tokens start before the weight wait and preparation errors still converge.
        """
        context = _Context(self.operations)
        metadata = Mock(side_effect=lambda *_args: self.operations.append(("metadata",)))
        self.compute.forward_pipeline = SimpleNamespace(
            enabled=True, supports=Mock(return_value=True), prepare_route_metadata=metadata)

        result = sparse_autograd.SparseExpertFunction.forward(
            context, self.hidden, self.scores, *self.home, self.key, self.runtime)

        self._assert_overlap_order("forward")
        metadata.assert_called_once()
        self.assertIs(metadata.call_args.args[0], self.route)
        self.assertIs(metadata.call_args.args[1], self.workspace)
        self.assertEqual(metadata.call_args.args[2].data_ptr(), self.hidden.data_ptr())
        self.assertIs(metadata.call_args.args[3], False)
        self.assertLess(self.operations.index(("read_weights",)), self.operations.index(("metadata",)))
        self.assertLess(self.operations.index(("metadata",)), self.operations.index(("preflight",)))
        self.assertTrue(self.prepared[0].publish_each_block)
        self.assertIsNotNone(self.prepared[0].ready.task_expected)
        self.assertEqual(result.shape, self.hidden.shape)
        self.workspace.check_preflight.assert_called_once_with(None)
        self.runtime.release.assert_called_once_with(self.key)
        self.assertLess(self.operations.index(("combine",)), self.operations.index(("release",)))

    def test_native_keepalives_do_not_retain_graphs_and_manual_gradients_reach_inputs(self) -> None:
        """Feature: Async sparse-expert autograd ownership.

        Description: Retain transport, router-score, and output aliases like native handlers.
        Expectation: Aliases share storage without history; gradients reach both original leaves.
        """
        hidden_leaf = torch.ones_like(self.hidden, requires_grad=True)
        score_leaf = torch.ones(self.scores.shape, dtype=torch.float32, requires_grad=True)
        hidden, scores = hidden_leaf * 2, score_leaf * 3
        retained = {}

        def _combine(values, inverse, probabilities, tokens, top_k, *, output):
            retained.update(scores=probabilities, output=output)
            return self._combine(values, inverse, probabilities, tokens, top_k, output=output)

        def _backward(_context, _derivative):
            return torch.ones_like(hidden), torch.ones_like(scores), None, None, None, None

        with patch.object(sparse_autograd, "combine", side_effect=_combine), \
                patch.object(sparse_autograd.SparseExpertFunction, "backward", new=staticmethod(_backward)):
            result = sparse_autograd.SparseExpertFunction.apply(
                hidden, scores, *self.home, self.key, self.runtime)

            source = self.prepared[0].source
            for value in (source, retained["scores"], retained["output"]):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
            self.assertEqual(source.data_ptr(), hidden.data_ptr())
            self.assertEqual(retained["scores"].data_ptr(), scores.data_ptr())
            self.assertEqual(retained["output"].data_ptr(), result.data_ptr())
            self.assertTrue(result.requires_grad)
            self.assertIsNotNone(result.grad_fn)

            result.backward(torch.ones_like(result))

        torch.testing.assert_close(hidden_leaf.grad, torch.full_like(hidden_leaf, 2))
        torch.testing.assert_close(score_leaf.grad, torch.full_like(score_leaf, 3))

    def test_replay_submits_tokens_then_waits_without_releasing_the_weight_ticket(self):
        """Replay uses the same safe event order and retains weights for its backward."""
        self.replaying.return_value = True
        context = _Context(self.operations)

        sparse_autograd.SparseExpertFunction.forward(
            context, self.hidden, self.scores, *self.home, self.key, self.runtime)

        self._assert_overlap_order("forward")
        self.assertFalse(self.prepared[0].publish_each_block)
        self.assertIsNone(self.prepared[0].ready.task_expected)
        self.expert.read_weights.assert_called_once()
        self.runtime.release.assert_not_called()

    def test_backward_computes_source_derivatives_and_submits_tokens_before_weight_wait(self):
        """dscore/dV remain independent of W, while the first expert GEMM joins it."""
        context = _Context(self.operations)
        context.key, context.runtime = self.key, self.runtime
        context.tokens, context.shapes = 3, tuple(value.shape for value in self.home)
        context.saved = (torch.empty((4, 16), dtype=torch.bfloat16),
                         torch.empty((4, 32), dtype=torch.bfloat16),
                         torch.empty_like(self.hidden), self.scores, self.plan.order)
        dy = torch.ones((1, 16), dtype=torch.bfloat16).expand(3, 16)

        gradients = sparse_autograd.SparseExpertFunction.backward(context, dy)

        self._assert_overlap_order("backward")
        self.assertLess(self.operations.index(("unpack",)), self.operations.index(("read_weights",)))
        self.assertLess(self.operations.index(("unpack",)), self.operations.index(("prepare_backward",)))
        self.assertLess(self.operations.index(("combine_gradient",)), self.operations.index(("transport", "T")))
        self.assertIs(self.prepared[0].source, self.prepared[1].value)
        self.assertEqual(gradients[0].shape, dy.shape)
        self.runtime.release.assert_called_once_with(self.key)
        self.runtime.complete.assert_called_once_with(self.key)
