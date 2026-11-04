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
"""CPU replica pool ownership checks with mocked NPU streams and collectives."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused import weights as weights_module
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool, overlaps, prepare_gradient_storage
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import compile_transfer_rounds
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import TransferDomain
from tests.common.mark_utils import arg_mark


class _Streams:
    """Represent stream/event submission order without any device initialization."""

    def __init__(self) -> None:
        """Create a compute stream and collect communication events for inspection."""
        self.compute = Mock(name="compute_stream")
        self.current = self.compute
        self.events = []

    @staticmethod
    def Stream(**_kwargs: Any) -> Mock:  # pylint: disable=invalid-name
        """Create a communication stream handle with observable wait calls."""
        return Mock(name="communication_stream")

    def Event(self) -> Mock:  # pylint: disable=invalid-name
        """Create a fence whose record stream can be inspected by the test."""
        event = Mock(name="event")
        self.events.append(event)
        return event

    def current_stream(self, _device: torch.device) -> Mock:
        """Return the current mocked stream."""
        return self.current

    @contextmanager
    def stream(self, stream: Mock) -> Iterator[None]:
        """Run a callback inside its intended communication stream."""
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestReplicaPool(unittest.TestCase):
    """Check parameterized byte accounting and physical matrix layouts."""

    def test_parameterized_budget_uses_replica_capacity(self):
        """The main cap=8 manifest reserves 496 MiB including both staging domains."""
        shapes = ((2048, 1536), (768, 2048))
        for capacity in (0, 1, 8, 16):
            with self.subTest(capacity=capacity):
                weight = ReplicaPool(torch.device("meta"), torch.bfloat16)
                gradient = ReplicaPool(torch.device("meta"), torch.float32)
                weight.allocate(8, capacity, shapes, (True, True))
                gradient.allocate(8, capacity, shapes, (True, True))
                self.assertEqual(weight.reserved_bytes, 2 * capacity * 9 * 1024**2)
                self.assertEqual(gradient.reserved_bytes, 96 * 1024**2)
                total = weight.reserved_bytes + gradient.reserved_bytes + 256 * 1024**2
                self.assertEqual(total, (352 + 18 * capacity) * 1024**2)
                if capacity == 8:
                    self.assertEqual(total, 496 * 1024**2)


    def test_borrowed_accumulator_removes_only_owned_gradient_bytes(self):
        """Runtime staging plus two cap=8 W slots reserves 400 MiB before arena scratch."""
        gradient = ReplicaPool(torch.device("meta"), torch.float32)
        gradient.allocate(8, 8, ((2048, 1536), (768, 2048)), (True, True), caller_accumulator=True)
        self.assertEqual(gradient.reserved_bytes, 0)
        self.assertIsNone(gradient.accumulator)
        self.assertEqual(gradient.home, {})
        weight = ReplicaPool(torch.device("meta"), torch.bfloat16)
        weight.allocate(8, 8, ((2048, 1536), (768, 2048)), (True, True))
        self.assertEqual(weight.reserved_bytes + gradient.reserved_bytes + 256 * 1024**2, 400 * 1024**2)

    def test_incoming_capacity_covers_empty_low_and_large_token_counts(self):
        """Both views retain exact logical extents while the arena reserves their maximum."""
        for rows in (0, 1, 16):
            with self.subTest(rows=rows):
                arena = PhaseArena("cpu")
                token = arena.begin()
                allocate = Mock(wraps=arena.allocate)
                incoming, accumulator, staging = prepare_gradient_storage(
                    allocate, (rows, 4), ((1, 2, 4), (1, 2, 2)), staging_elements=16)
                allocate.assert_called_once_with("incoming", (max(rows * 4, 80),), torch.bfloat16)
                self.assertEqual(incoming.shape, (rows, 4))
                self.assertEqual(accumulator.shape, (8,))
                self.assertEqual(accumulator.dtype, torch.float32)
                for value in staging:
                    self.assertEqual(value.shape, (16,))
                    self.assertEqual(value.data_ptr() % 32, 0)
                    self.assertFalse(overlaps(accumulator, value))
                self.assertFalse(overlaps(*staging))
                self.assertEqual(accumulator.data_ptr() % 32, 0)
                self.assertEqual(incoming.untyped_storage().data_ptr(), accumulator.untyped_storage().data_ptr())
                pointer = accumulator.data_ptr()
                arena.end(token)
                token = arena.begin()
                _, again, _ = prepare_gradient_storage(
                    arena.allocate, (rows, 4), ((1, 2, 4), (1, 2, 2)), staging_elements=16)
                self.assertEqual(again.data_ptr(), pointer)
                arena.end(token)
                arena.close()


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestTransferDomainPool(unittest.TestCase):
    """Check leases and complete gradient outputs without a process group or NPU."""

    def setUp(self) -> None:
        """Replace all stream, storage-recording, and communication boundaries."""
        self.streams = _Streams()
        self.distributed = Mock()
        self.distributed.get_world_size.return_value = 2
        self.distributed.all_to_all_single.side_effect = self._exchange
        self.work = Mock()
        patches = (patch.object(torch.Tensor, "record_stream"),
                   patch.object(weights_module.torch, "npu", self.streams, create=True),
                   patch.object(weights_module, "dist", self.distributed))
        started = []
        for patcher in patches:
            started.append(patcher.start())
            self.addCleanup(patcher.stop)
        self.record = started[0]
        self.shapes, self.layouts = ((2, 4), (2, 2)), (False, False)
        self.home = (torch.ones((1, 2, 4), dtype=torch.bfloat16), torch.ones((1, 2, 2), dtype=torch.bfloat16))
        allocation = torch.tensor([[1, 0], [1, 1]], dtype=torch.int64)
        self.rounds = compile_transfer_rounds(allocation, (("gate", 8), ("down", 4)), quota_elements=32)

    def _exchange(self, output, _input, _receive, _send, **_kwargs):
        output.fill_(7)
        return self.work

    def _domain(self, dtype=torch.bfloat16, *, admit=True, caller_accumulator=False, caller_staging=False):
        domain = TransferDomain(object(), 0, torch.device("cpu"), dtype=dtype, quota_bytes=64)
        if admit:
            domain.admit(1, 1, self.shapes, self.layouts, caller_accumulator=caller_accumulator,
                         caller_staging=caller_staging)
        return domain

    def _materialize(self, domain, *, remote=True):
        def _restore():
            self.assertIs(self.streams.current, domain.stream)
            return self.home

        return domain.materialize((0, 1) if remote else (0,), 1, _restore,
                                  self.rounds if remote else (), self.shapes, self.layouts)

    def test_slot_reuse_waits_for_last_compute_reader(self):
        """A released slot reuses storage only after W waits for its C completion fence."""
        domain = self._domain()
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("hot allocation")):
            first, second = self._materialize(domain), self._materialize(domain)
            self.assertTrue(all(slot.in_use for slot in domain._pool.slots))
            self.assertNotEqual(first.remote[1, "gate"].data_ptr(), second.remote[1, "gate"].data_ptr())
            domain.release_lease(first)
            last_reader = self.streams.events[-1]
            self.streams.compute.wait_event.assert_called_with(first.ready)
            last_reader.record.assert_called_once_with(self.streams.compute)
            third = self._materialize(domain)
            domain.stream.wait_event.assert_called_with(last_reader)
            self.assertEqual(first.remote[1, "gate"].data_ptr(), third.remote[1, "gate"].data_ptr())
            self.assertEqual(third.generation, first.generation + 1)
            domain.release_lease(second)
            domain.release_lease(third)
        self.assertTrue(all(not slot.in_use for slot in domain._pool.slots))
        self.assertTrue(self.record.called)
        self.distributed.all_reduce.assert_not_called()
        self.assertEqual(self.work.wait.call_count, 3)

    def test_zero_remote_still_consumes_a_logical_slot(self):
        """Empty receivers cannot advance their collective schedule past two leases."""
        domain = self._domain()
        first, second = self._materialize(domain, remote=False), self._materialize(domain, remote=False)
        self.assertEqual(first.remote, {})
        self.assertEqual(second.remote, {})
        self.assertTrue(all(slot.in_use for slot in domain._pool.slots))
        domain.release_lease(first)
        self.assertTrue(any(not slot.in_use for slot in domain._pool.slots))
        domain.release_lease(second)
        self.distributed.all_to_all_single.assert_not_called()


    def test_gradient_outputs_survive_reusing_the_fp32_accumulator(self):
        """Autograd owns each BF16 result while G reuses its FP32 region in FIFO order."""
        domain = self._domain(torch.float32)
        gate = torch.full((1, 2, 4), 1.25, dtype=torch.float32)
        down = torch.full((1, 2, 2), 2.5, dtype=torch.float32)
        gate_output = torch.empty_like(gate, dtype=torch.bfloat16)
        down_output = torch.empty_like(down, dtype=torch.bfloat16)
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("hot allocation")):
            first = domain.return_gradient("gate", (0,), 1, gate, (), output=gate_output)
            second = domain.return_gradient("down", (0,), 1, down, (), output=down_output)
        self.assertIs(first.value, gate_output)
        self.assertIs(second.value, down_output)
        torch.testing.assert_close(gate_output, gate.to(torch.bfloat16))
        torch.testing.assert_close(down_output, down.to(torch.bfloat16))
        first.ready.record.assert_called_once_with(domain.stream)
        second.ready.record.assert_called_once_with(domain.stream)
        self.distributed.all_reduce.assert_not_called()

    def test_gradient_output_cannot_alias_payload_or_domain_storage(self):
        """Different tensor dtypes cannot hide an overlapping physical allocation."""
        for owner in ("gradient", "send", "receive"):
            with self.subTest(owner=owner):
                domain = self._domain(torch.float32)
                gradient = torch.ones((1, 2, 4), dtype=torch.float32)
                storage = gradient if owner == "gradient" else domain.buffers[owner]
                output = storage.view(torch.bfloat16).reshape(-1)[:8].view(1, 2, 4)
                with self.assertRaisesRegex(ValueError, "must not alias"):
                    domain.return_gradient("gate", (0,), 1, gradient, (), output=output)
                self.assertTrue(domain.poisoned)


    def test_borrowed_accumulator_overwrite_waits_for_last_compute_reader(self):
        """The existing C-to-G wait precedes zero/copy/cast, with no hot storage allocation."""
        domain = self._domain(torch.float32, caller_accumulator=True)
        arena = PhaseArena("cpu")
        token = arena.begin()
        incoming, accumulator, _ = prepare_gradient_storage(
            arena.allocate, (4, 4), ((1, 2, 4), (1, 2, 2)), staging_elements=16)
        incoming.fill_(3)
        gradient = torch.full((1, 2, 2), 2.5, dtype=torch.float32)
        output = torch.empty_like(gradient, dtype=torch.bfloat16)
        order = []

        def _wait_compute(current):
            self.assertIs(current, self.streams.compute)
            torch.testing.assert_close(incoming, torch.full_like(incoming, 3))
            order.append("wait_last_dv_reader")

        domain.stream.wait_stream.side_effect = _wait_compute
        domain.stream.wait_stream.reset_mock()
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("hot allocation")):
            packet = domain.return_gradient("down", (0,), 1, gradient, (), output=output, accumulator=accumulator)
        order.append("cast_submitted")
        self.assertEqual(order, ["wait_last_dv_reader", "cast_submitted"])
        domain.stream.wait_stream.assert_called_once_with(self.streams.compute)
        packet.ready.record.assert_called_once_with(domain.stream)
        torch.testing.assert_close(accumulator[:4], gradient.flatten())
        torch.testing.assert_close(output, gradient.to(torch.bfloat16))
        self.assertEqual(domain.reserved_bytes, 128)
        self.assertTrue(self.record.called)
        self.distributed.all_reduce.assert_not_called()
        self.streams.compute.wait_event(packet.ready)
        arena.end(token)
        arena.close()


    def test_borrowed_staging_is_not_retained_after_the_frame(self):
        """Dropping phase views leaves no domain reference that could survive into CE."""
        domain = self._domain(torch.float32, caller_accumulator=True, caller_staging=True)
        arena = PhaseArena("cpu")
        token = arena.begin()
        incoming, accumulator, staging = prepare_gradient_storage(
            arena.allocate, (0, 4), ((1, 2, 4), (1, 2, 2)), staging_elements=16)
        reference = weakref.ref(staging[0]._base)
        packet = domain.return_gradient("gate", (), 1, torch.empty((0, 2, 4)), (),
                                        output=torch.empty((1, 2, 4), dtype=torch.bfloat16),
                                        accumulator=accumulator, staging=staging)
        self.streams.compute.wait_event(packet.ready)
        arena.end(token)
        arena.release_regions(("incoming",))
        del incoming, accumulator, staging
        self.assertIsNone(reference())
        self.assertEqual(domain.buffers, {})
        arena.close()
