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

    def test_weight_layout_and_gradient_accumulator_views(self):
        """W preserves physical transposes; G roles share one maximum-sized accumulator."""
        weight = ReplicaPool(torch.device("cpu"), torch.bfloat16)
        weight.allocate(1, 2, ((2, 4), (2, 2)), (True, False))
        self.assertTrue(weight.slots[0].matrices["gate"].transpose(-2, -1).is_contiguous())
        self.assertTrue(weight.slots[0].matrices["down"].is_contiguous())
        self.assertNotEqual(weight.slots[0].matrices["gate"].data_ptr(),
                            weight.slots[1].matrices["gate"].data_ptr())
        gradient = ReplicaPool(torch.device("cpu"), torch.float32)
        gradient.allocate(1, 2, ((2, 4), (2, 2)), (True, False))
        self.assertEqual(gradient.home["gate"].data_ptr(), gradient.home["down"].data_ptr())
        self.assertEqual(gradient.reserved_bytes, 32)

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

    def test_incoming_reinterpretation_rejects_misaligned_storage(self):
        """An odd BF16 offset cannot be reinterpreted as an aligned FP32 accumulator."""
        storage = torch.empty(81, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "aligned BF16/FP32"):
            prepare_gradient_storage(Mock(return_value=storage[1:]), (1, 4), ((1, 2, 4), (1, 2, 2)),
                                     staging_elements=16)

    def test_main_manifest_capacity_and_empty_rank(self):
        """Qwen needs 224 MiB at zero rows and retains the existing 256 MiB at 65536 rows."""
        shapes = ((8, 2048, 1536), (8, 768, 2048))
        for rows, capacity in ((0, 224), (1, 224), (65536, 256)):
            with self.subTest(rows=rows):
                def _allocate(_name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
                    return torch.empty(shape, dtype=dtype, device="meta")

                allocate = Mock(side_effect=_allocate)
                incoming, accumulator, staging = prepare_gradient_storage(
                    allocate, (rows, 2048), shapes, staging_elements=16 * 1024**2)
                allocate.assert_called_once_with("incoming", (capacity * 1024**2 // 2,), torch.bfloat16)
                self.assertEqual(incoming.shape, (rows, 2048))
                self.assertEqual(accumulator.numel() * 4, 96 * 1024**2)
                self.assertEqual(tuple(value.numel() * 4 for value in staging), (64 * 1024**2,) * 2)

    def test_staging_alignment_includes_short_roles_and_odd_quotas(self):
        """Alignment padding belongs to the admitted region, not a later allocation."""
        arena = PhaseArena("cpu")
        token = arena.begin()
        _, accumulator, staging = prepare_gradient_storage(
            arena.allocate, (0, 4), ((1, 1, 3), (1, 1, 1)), staging_elements=3)
        self.assertEqual(accumulator.numel(), 3)
        for value in staging:
            self.assertEqual(value.numel(), 3)
            self.assertEqual(value.data_ptr() % 32, 0)
            self.assertFalse(overlaps(accumulator, value))
        self.assertFalse(overlaps(*staging))
        arena.end(token)
        arena.close()

    def test_invalid_staging_quota_is_rejected_before_allocation(self):
        """A malformed quota cannot enter the collective admission with partial storage."""
        for quota in (0, -1, True, 1.5):
            with self.subTest(quota=quota):
                allocate = Mock()
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    prepare_gradient_storage(allocate, (0, 4), ((1, 2, 4), (1, 2, 2)),
                                             staging_elements=quota)
                allocate.assert_not_called()


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
            self.assertFalse(domain.has_free_slot)
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
        self.assertTrue(domain.has_free_slot)
        self.assertTrue(self.record.called)
        self.distributed.all_reduce.assert_not_called()
        self.assertEqual(self.work.wait.call_count, 3)

    def test_zero_remote_still_consumes_a_logical_slot(self):
        """Empty receivers cannot advance their collective schedule past two leases."""
        domain = self._domain()
        first, second = self._materialize(domain, remote=False), self._materialize(domain, remote=False)
        self.assertEqual(first.remote, {})
        self.assertEqual(second.remote, {})
        self.assertFalse(domain.has_free_slot)
        domain.release_lease(first)
        self.assertTrue(domain.has_free_slot)
        domain.release_lease(second)
        self.distributed.all_to_all_single.assert_not_called()

    def test_third_live_lease_fails_and_poisons_domain(self):
        """Exhausted capacity cannot silently overwrite either active weight borrow."""
        domain = self._domain()
        self._materialize(domain, remote=False)
        self._materialize(domain, remote=False)
        with self.assertRaisesRegex(RuntimeError, "both replica slots"):
            self._materialize(domain, remote=False)
        self.assertTrue(domain.poisoned)

    def test_foreign_and_repeated_releases_poison_domain(self):
        """A release cannot transfer ownership or reuse an already-expired token."""
        first, other = self._domain(), self._domain()
        lease = self._materialize(first, remote=False)
        with self.assertRaisesRegex(ValueError, "another domain"):
            other.release_lease(lease)
        self.assertTrue(other.poisoned)
        self.assertFalse(first.poisoned)
        first.release_lease(lease)
        with self.assertRaisesRegex(ValueError, "already been released"):
            first.release_lease(lease)
        self.assertTrue(first.poisoned)

    def test_admission_failure_clears_partial_storage(self):
        """Every partial W/G allocation failure clears locally published storage."""
        original_empty = torch.empty
        for dtype, allocations in ((torch.bfloat16, 6), (torch.float32, 3)):
            for fail_at in range(1, allocations + 1):
                with self.subTest(dtype=dtype, fail_at=fail_at):
                    domain, retained, calls = self._domain(dtype, admit=False), [], []

                    def _allocate(*args, **kwargs):
                        calls.append(1)
                        if len(calls) == fail_at:
                            raise MemoryError("injected allocation failure")
                        tensor = original_empty(*args, **kwargs)
                        retained.append(weakref.ref(tensor))
                        return tensor

                    with patch.object(weights_module.torch, "empty", side_effect=_allocate):
                        with self.assertRaisesRegex(MemoryError, "injected allocation failure"):
                            domain.admit(1, 1, self.shapes, self.layouts)
                    self.assertTrue(domain.poisoned)
                    self.assertFalse(domain.admitted)
                    self.assertEqual(domain.buffers, {})
                    self.assertEqual(domain.reserved_bytes, 0)
                    self.assertTrue(all(reference() is None for reference in retained))

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

    def test_empty_active_gradient_zeros_complete_home_output(self):
        """A later empty rank does not return values retained from its previous phase."""
        domain = self._domain(torch.float32)
        full = torch.ones((1, 2, 4), dtype=torch.float32)
        output = torch.empty((1, 2, 4), dtype=torch.bfloat16)
        domain.return_gradient("gate", (0,), 1, full, (), output=output)
        empty = torch.empty((0, 2, 4), dtype=torch.float32)
        domain.return_gradient("gate", (), 1, empty, (), output=output)
        torch.testing.assert_close(output, torch.zeros_like(output))

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

    def test_borrowed_accumulator_survives_role_reuse_and_empty_rank(self):
        """Down output stays independent when gate reuses scratch, including an empty receiver."""
        domain = self._domain(torch.float32, caller_accumulator=True)
        accumulator = torch.empty(8, dtype=torch.float32)
        down = torch.full((1, 2, 2), 2.5, dtype=torch.float32)
        down_output = torch.empty_like(down, dtype=torch.bfloat16)
        first = domain.return_gradient("down", (0,), 1, down, (), output=down_output, accumulator=accumulator)
        gate_output = torch.empty((1, 2, 4), dtype=torch.bfloat16)
        domain.return_gradient("gate", (), 1, torch.empty((0, 2, 4)), (),
                               output=gate_output, accumulator=accumulator)
        self.assertIs(first.value, down_output)
        torch.testing.assert_close(down_output, down.to(torch.bfloat16))
        torch.testing.assert_close(gate_output, torch.zeros_like(gate_output))
        self.assertEqual(domain.reserved_bytes, 128)

    def test_accumulator_ownership_must_match_admission(self):
        """Borrowed mode requires phase storage; owned mode rejects an ambiguous override."""
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                domain = self._domain(torch.float32, caller_accumulator=borrowed)
                accumulator = None if borrowed else torch.empty(8, dtype=torch.float32)
                with self.assertRaisesRegex(ValueError, "ownership differs"):
                    domain.return_gradient("gate", (), 1, torch.empty((0, 2, 4)), (),
                                           output=torch.empty((1, 2, 4), dtype=torch.bfloat16), accumulator=accumulator)
                self.assertTrue(domain.poisoned)

    def test_borrowed_accumulator_rejects_invalid_extent_dtype_layout_and_alignment(self):
        """Even down requires the complete max-role region admitted for later gate reuse."""
        cases = (torch.empty(4), torch.empty(8, dtype=torch.bfloat16), torch.empty(16)[::2],
                 torch.empty(9)[1:], torch.empty((2, 4)), torch.empty(8, device="meta"))
        for accumulator in cases:
            with self.subTest(shape=accumulator.shape, dtype=accumulator.dtype, device=accumulator.device):
                domain = self._domain(torch.float32, caller_accumulator=True)
                with self.assertRaises(ValueError):
                    domain.return_gradient("down", (), 1, torch.empty((0, 2, 2)), (),
                                           output=torch.empty((1, 2, 2), dtype=torch.bfloat16), accumulator=accumulator)
                self.assertTrue(domain.poisoned)

    def test_borrowed_accumulator_cannot_overlap_payload_output_or_staging(self):
        """Checks include the unused max-role tail and overlaps hidden by a BF16 view."""
        for owner in ("gradient", "output", "send", "receive"):
            with self.subTest(owner=owner):
                domain = self._domain(torch.float32, caller_accumulator=True)
                accumulator = torch.empty(8)
                gradient = torch.empty((1, 2, 2))
                output = torch.empty((1, 2, 2), dtype=torch.bfloat16)
                if owner == "gradient":
                    gradient = accumulator[4:].view(1, 2, 2)
                elif owner == "output":
                    output = accumulator.view(torch.bfloat16)[12:].view(1, 2, 2)
                else:
                    accumulator = domain.buffers[owner][:8]
                with self.assertRaisesRegex(ValueError, "must not alias"):
                    domain.return_gradient("down", (0,), 1, gradient, (), output=output, accumulator=accumulator)
                self.assertTrue(domain.poisoned)

    def test_borrowed_accumulator_allows_disjoint_views(self):
        """Disjoint byte ranges in one storage do not falsely alias scratch and output."""
        domain = self._domain(torch.float32, caller_accumulator=True)
        storage = torch.empty(16)
        accumulator, gradient = storage[:8], storage[8:12].view(1, 2, 2)
        output = storage[12:14].view(torch.bfloat16).view(1, 2, 2)
        gradient.fill_(2.5)
        domain.return_gradient("down", (0,), 1, gradient, (), output=output, accumulator=accumulator)
        torch.testing.assert_close(output, torch.full_like(output, 2.5))

    def test_borrowed_admission_failure_clears_staging(self):
        """External G admission has only its two staging allocations to roll back."""
        domain = self._domain(torch.float32, admit=False)
        storage = torch.empty(16)
        with patch.object(weights_module.torch, "empty", side_effect=[storage, MemoryError("receive OOM")]):
            with self.assertRaisesRegex(MemoryError, "receive OOM"):
                domain.admit(1, 1, self.shapes, self.layouts, caller_accumulator=True)
        self.assertEqual(domain.reserved_bytes, 0)
        self.assertEqual(domain.buffers, {})
        self.assertTrue(domain.poisoned)

    def test_borrowed_staging_admission_reserves_no_persistent_gradient_storage(self):
        """Borrowing both G regions avoids all persistent allocator calls and collectives."""
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("persistent allocation")):
            domain = self._domain(torch.float32, caller_accumulator=True, caller_staging=True)
        self.assertEqual(domain.buffers, {})
        self.assertEqual(domain.reserved_bytes, 0)
        self.distributed.all_reduce.assert_not_called()
        self.distributed.all_to_all_single.assert_not_called()

    def test_borrowed_staging_waits_for_dv_and_preserves_complete_output(self):
        """G starts writing the shared dV region after C, then serially reuses it for gate."""
        domain = self._domain(torch.float32, caller_accumulator=True, caller_staging=True)
        arena = PhaseArena("cpu")
        token = arena.begin()
        incoming, accumulator, staging = prepare_gradient_storage(
            arena.allocate, (64, 4), ((1, 2, 4), (1, 2, 2)), staging_elements=domain.quota_elements)
        incoming.fill_(3)
        allocation = torch.ones((2, 2), dtype=torch.int64)
        rounds = compile_transfer_rounds(allocation, (("down", 4),), reverse=True, quota_elements=16)
        gradient = torch.full((2, 2, 2), 2.5)
        output = torch.empty((1, 2, 2), dtype=torch.bfloat16)

        def _wait_compute(current: Mock) -> None:
            self.assertIs(current, self.streams.compute)
            torch.testing.assert_close(incoming, torch.full_like(incoming, 3))

        domain.stream.wait_stream.side_effect = _wait_compute
        domain.stream.wait_stream.reset_mock()
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("hot allocation")):
            first = domain.return_gradient("down", (0, 1), 1, gradient, rounds, output=output,
                                           accumulator=accumulator, staging=staging)
        domain.stream.wait_stream.assert_called_once_with(self.streams.compute)
        self.work.wait.assert_called_once_with()
        torch.testing.assert_close(output, torch.full_like(output, 9.5))
        torch.testing.assert_close(staging[0][:4], torch.full((4,), 2.5))
        torch.testing.assert_close(staging[1][:4], torch.full((4,), 7.0))
        first.ready.record.assert_called_once_with(domain.stream)
        domain.stream.wait_stream.side_effect = None
        gate_output = torch.empty((1, 2, 4), dtype=torch.bfloat16)
        second = domain.return_gradient("gate", (), 1, torch.empty((0, 2, 4)), (), output=gate_output,
                                        accumulator=accumulator, staging=staging)
        torch.testing.assert_close(gate_output, torch.zeros_like(gate_output))
        torch.testing.assert_close(output, torch.full_like(output, 9.5))
        self.assertEqual(domain.buffers, {})
        self.assertEqual(domain.reserved_bytes, 0)
        self.distributed.all_reduce.assert_not_called()
        self.streams.compute.wait_event(first.ready)
        self.streams.compute.wait_event(second.ready)
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

    def test_staging_ownership_must_match_admission(self):
        """Owned and borrowed domains reject overrides or missing staging before any write."""
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                domain = self._domain(torch.float32, caller_accumulator=True, caller_staging=borrowed)
                staging = None if borrowed else (torch.empty(16), torch.empty(16))
                with self.assertRaisesRegex(ValueError, "staging ownership differs"):
                    domain.return_gradient("gate", (), 1, torch.empty((0, 2, 4)), (),
                                           output=torch.empty((1, 2, 4), dtype=torch.bfloat16),
                                           accumulator=torch.empty(8), staging=staging)
                self.assertTrue(domain.poisoned)
                self.distributed.all_to_all_single.assert_not_called()

    def test_borrowed_staging_rejects_invalid_views_and_all_writable_aliases(self):
        """The full quotas cannot overlap each other, any accumulator byte, dW, or final output."""
        cases = ("pair", "size", "dtype", "layout", "alignment", "device",
                 "send_receive", "accumulator", "gradient", "output")
        for case in cases:
            with self.subTest(case=case):
                domain = self._domain(torch.float32, caller_accumulator=True, caller_staging=True)
                accumulator, gradient = torch.empty(8), torch.empty((1, 2, 4))
                output = torch.empty((1, 2, 4), dtype=torch.bfloat16)
                send, receive = torch.empty(16), torch.empty(16)
                if case == "size":
                    send = send[:15]
                elif case == "dtype":
                    send = torch.empty(16, dtype=torch.bfloat16)
                elif case == "layout":
                    send = torch.empty(32)[::2]
                elif case == "alignment":
                    send = torch.empty(17)[1:]
                elif case == "device":
                    send = torch.empty(16, device="meta")
                elif case == "send_receive":
                    receive = send
                elif case == "accumulator":
                    accumulator = send[8:]
                elif case == "gradient":
                    gradient = send[8:].view(1, 2, 4)
                elif case == "output":
                    output = receive[8:12].view(torch.bfloat16).view(1, 2, 4)
                staging = (send,) if case == "pair" else (send, receive)
                with self.assertRaises(ValueError):
                    domain.return_gradient("gate", (0,), 1, gradient, (), output=output,
                                           accumulator=accumulator, staging=staging)
                self.assertTrue(domain.poisoned)
                self.distributed.all_to_all_single.assert_not_called()
