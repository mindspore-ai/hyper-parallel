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
"""CPU and mocked stream coverage for the root temporary arena."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from hyper_parallel.core.multicore.modules.routed_experts import arena as arena_module
from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from tests.common.mark_utils import arg_mark


_QUANTUM = 2 * 1024 * 1024


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestPhaseArena(unittest.TestCase):
    """Check fixed storage ownership, capacity growth, and phase exclusivity."""

    def test_regions_have_independent_storage_and_rounded_capacity(self):
        """Different names never alias even when the requested shapes match."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        first = arena.allocate("first", (3, 7), torch.bfloat16)
        second = arena.allocate("second", (3, 7), torch.bfloat16)
        self.assertEqual(first.shape, (3, 7))
        self.assertTrue(first.is_contiguous())
        self.assertNotEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())
        self.assertEqual(first.untyped_storage().nbytes(), _QUANTUM)
        self.assertEqual(arena.reserved_bytes, 2 * _QUANTUM)
        arena.end(token)
        arena.close()
        self.assertEqual(arena.reserved_bytes, 0)

    def test_next_phase_reuses_high_water_storage(self):
        """Shape changes below the retained capacity keep the same allocation."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        original = arena.allocate("activation", (7, 9), torch.float32)
        original.fill_(13)
        arena.end(token)
        token = arena.begin()
        reused = arena.allocate("activation", (3, 5), torch.float32)
        self.assertEqual(original.data_ptr(), reused.data_ptr())
        self.assertEqual(arena.reserved_bytes, _QUANTUM)
        torch.testing.assert_close(reused, torch.full_like(reused, 13))
        arena.end(token)

    def test_growth_rounds_up_and_shrinking_keeps_capacity(self):
        """A name grows independently and its later smaller request does not free it."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        small = arena.allocate("region", (1,), torch.float32)
        arena.end(token)
        token = arena.begin()
        large = arena.allocate("region", (_QUANTUM // 4 + 1,), torch.float32)
        self.assertNotEqual(small.data_ptr(), large.data_ptr())
        self.assertEqual(arena.reserved_bytes, 2 * _QUANTUM)
        arena.end(token)
        token = arena.begin()
        shrunk = arena.allocate("region", (1,), torch.float32)
        self.assertEqual(shrunk.data_ptr(), large.data_ptr())
        self.assertEqual(arena.reserved_bytes, 2 * _QUANTUM)
        arena.end(token)

    def test_duplicate_name_and_dtype_change_are_rejected(self):
        """One phase cannot overwrite a previous borrow and a name keeps its dtype."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        arena.allocate("region", (2,), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "once per phase"):
            arena.allocate("region", (4,), torch.bfloat16)
        arena.end(token)
        token = arena.begin()
        with self.assertRaisesRegex(ValueError, "change dtype"):
            arena.allocate("region", (2,), torch.float32)
        arena.allocate("region", (2,), torch.bfloat16)
        arena.end(token)

    def test_active_phase_and_token_ownership_are_required(self):
        """Nested begin, foreign tokens, and repeated end leave the live lease intact."""
        arena, other = PhaseArena(torch.device("cpu")), PhaseArena(torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "active phase"):
            arena.allocate("region", (1,), torch.float32)
        token, foreign = arena.begin(), other.begin()
        with self.assertRaisesRegex(RuntimeError, "active phase"):
            arena.begin()
        with self.assertRaisesRegex(ValueError, "foreign"):
            arena.end(foreign)
        with self.assertRaisesRegex(RuntimeError, "active phase"):
            arena.close()
        arena.end(token)
        with self.assertRaisesRegex(ValueError, "already released"):
            arena.end(token)
        other.end(foreign)

    def test_close_is_idempotent_and_blocks_new_work(self):
        """Closed arenas cannot be reopened or allocated from."""
        arena = PhaseArena(torch.device("cpu"))
        arena.close()
        arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.begin()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.allocate("region", (1,), torch.float32)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.end(object())

    def test_growth_oom_preserves_previous_region(self):
        """A failed replacement neither drops old capacity nor consumes the name."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        original = arena.allocate("region", (2,), torch.float32)
        original.fill_(17)
        arena.end(token)
        token = arena.begin()
        with patch.object(arena_module.torch, "empty", side_effect=MemoryError("mock allocation failure")):
            with self.assertRaisesRegex(MemoryError, "mock allocation failure"):
                arena.allocate("region", (_QUANTUM // 4 + 1,), torch.float32)
        self.assertEqual(arena.reserved_bytes, _QUANTUM)
        retained = arena.allocate("region", (2,), torch.float32)
        self.assertEqual(retained.data_ptr(), original.data_ptr())
        torch.testing.assert_close(retained, torch.full_like(retained, 17))
        arena.end(token)

    def test_first_allocation_oom_does_not_publish_a_region(self):
        """A new name can be retried after its preparation allocation failed."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        with patch.object(arena_module.torch, "empty", side_effect=MemoryError("mock allocation failure")):
            with self.assertRaises(MemoryError):
                arena.allocate("region", (2,), torch.float32)
        self.assertEqual(arena.reserved_bytes, 0)
        self.assertEqual(arena.allocate("region", (2,), torch.float32).shape, (2,))
        arena.end(token)

    def test_empty_and_scalar_shapes(self):
        """Empty ranks reserve no payload while scalar tensors occupy one quantum."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        empty = arena.allocate("empty", (0, 7), torch.bfloat16)
        self.assertEqual(empty.shape, (0, 7))
        self.assertEqual(arena.reserved_bytes, 0)
        scalar = arena.allocate("scalar", (), torch.float32)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(arena.reserved_bytes, _QUANTUM)
        arena.end(token)

    def test_invalid_requests_do_not_consume_name(self):
        """Malformed names, shapes, and dtypes fail before storage is published."""
        arena = PhaseArena(torch.device("cpu"))
        token = arena.begin()
        requests = (("", (1,), torch.float32), (2, (1,), torch.float32), ("region", -1, torch.float32),
                    ("region", (-1,), torch.float32), ("region", (True,), torch.float32),
                    ("region", (1.5,), torch.float32), ("region", (1,), "float32"))
        for name, shape, dtype in requests:
            with self.subTest(request=(name, shape, dtype)), self.assertRaises(ValueError):
                arena.allocate(name, shape, dtype)
        self.assertEqual(arena.reserved_bytes, 0)
        arena.allocate("region", (1,), torch.float32)
        arena.end(token)

    def test_released_backward_regions_rebuild_high_water_only_four_times(self):
        """First B reconstructs the previous capacities; later smaller phases only borrow views."""
        arena = PhaseArena("cpu")
        regions = (("selections", torch.bfloat16), ("incoming", torch.bfloat16),
                   ("down_gradient", torch.float32), ("gate_gradient", torch.float32))
        token = arena.begin()
        for index, (name, dtype) in enumerate(regions, 1):
            arena.allocate(name, (index * _QUANTUM // dtype.itemsize,), dtype)
        shared = arena.allocate("activation", (1,), torch.bfloat16)
        arena.end(token)
        self.assertEqual(arena.release_regions(tuple(name for name, _ in regions)), 10 * _QUANTUM)
        self.assertEqual(arena.reserved_bytes, _QUANTUM)
        with patch.object(arena_module.torch, "empty", wraps=torch.empty) as allocate:
            for phase in range(3):
                token = arena.begin()
                for index, (name, dtype) in enumerate(regions, 1):
                    elements = 1 if phase == 0 else index * _QUANTUM // dtype.itemsize
                    value = arena.allocate(name, (elements,), dtype)
                    self.assertEqual(value.untyped_storage().nbytes(), index * _QUANTUM)
                reused = arena.allocate("activation", (1,), torch.bfloat16)
                self.assertEqual(reused.data_ptr(), shared.data_ptr())
                arena.end(token)
        self.assertEqual(allocate.call_count, 4)
        self.assertEqual(arena.reserved_bytes, 11 * _QUANTUM)
        arena.close()

    def test_rebuild_oom_keeps_capacity_and_dtype(self):
        """Failed regrowth cannot erase the last successful capacity or admit another dtype."""
        arena = PhaseArena("cpu")
        token = arena.begin()
        arena.allocate("region", (_QUANTUM // 4 + 1,), torch.float32)
        arena.end(token)
        arena.release_regions(("region",))
        token = arena.begin()
        with patch.object(arena_module.torch, "empty", side_effect=MemoryError("rebuild OOM")):
            with self.assertRaisesRegex(MemoryError, "rebuild OOM"):
                arena.allocate("region", (_QUANTUM,), torch.float32)
        self.assertEqual(arena.reserved_bytes, 0)
        with self.assertRaisesRegex(ValueError, "change dtype"):
            arena.allocate("region", (1,), torch.bfloat16)
        rebuilt = arena.allocate("region", (1,), torch.float32)
        self.assertEqual(rebuilt.untyped_storage().nbytes(), 2 * _QUANTUM)
        arena.end(token)
        arena.close()

    def test_release_is_idempotent_and_does_not_invalidate_other_owners(self):
        """Dropping the root reference preserves storage still captured by a queued reader."""
        arena = PhaseArena("cpu")
        token = arena.begin()
        captured = arena.allocate("region", (4,), torch.float32)
        captured.fill_(17)
        owner = weakref.ref(captured._base)
        arena.end(token)
        self.assertEqual(arena.release_regions(("region", "region", "missing")), _QUANTUM)
        self.assertEqual(arena.release_regions(("region",)), 0)
        self.assertEqual(arena.reserved_bytes, 0)
        self.assertIsNotNone(owner())
        torch.testing.assert_close(captured, torch.full_like(captured, 17))
        del captured
        self.assertIsNone(owner())
        arena.close()

    def test_release_rejects_active_closed_and_invalid_names(self):
        """Validation precedes every removal so a rejected call cannot partly retire storage."""
        arena = PhaseArena("cpu")
        token = arena.begin()
        arena.allocate("region", (1,), torch.float32)
        with self.assertRaisesRegex(RuntimeError, "active phase"):
            arena.release_regions(("region",))
        arena.end(token)
        for names in ("region", ("region", ""), ("region", 1), None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                arena.release_regions(names)
            self.assertEqual(arena.reserved_bytes, _QUANTUM)
        arena.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            arena.release_regions(("region",))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestPhaseArenaStreams(unittest.TestCase):
    """Use mocked NPU stream handles without loading or initializing a device."""

    def test_begin_allocate_end_and_close_require_original_stream(self):
        """Reject another C stream before it can access or release pooled storage."""
        first, second = object(), object()
        backend = SimpleNamespace(current_stream=Mock(return_value=first), Event=Mock(side_effect=Mock))
        device = SimpleNamespace(type="npu")
        with patch.object(arena_module.torch, "device", return_value=device), \
                patch.object(arena_module.torch, "npu", backend, create=True):
            arena = PhaseArena("npu:0")
            token = arena.begin()
            backend.current_stream.return_value = second
            with self.assertRaisesRegex(RuntimeError, "original compute stream"):
                arena.allocate("region", (1,), torch.float32)
            with self.assertRaisesRegex(RuntimeError, "original compute stream"):
                arena.end(token)
            backend.current_stream.return_value = first
            arena.end(token)
            backend.current_stream.return_value = second
            with self.assertRaisesRegex(RuntimeError, "original compute stream"):
                arena.begin()
            with self.assertRaisesRegex(RuntimeError, "original compute stream"):
                arena.close()
            backend.current_stream.return_value = first
            token = arena.begin()
            arena.end(token)
            arena.close()
        backend.current_stream.assert_called_with(device)

    def test_third_begin_waits_only_for_oldest_incomplete_phase(self):
        """Two phases submit freely; the third waits for one oldest C completion."""
        stream = object()
        events = [Mock(name=f"phase_{index}") for index in range(3)]
        for event in events:
            event.query.return_value = False
        backend = SimpleNamespace(current_stream=Mock(return_value=stream), Event=Mock(side_effect=events))
        with patch.object(arena_module.torch, "device", return_value=SimpleNamespace(type="npu")), \
                patch.object(arena_module.torch, "npu", backend, create=True):
            arena = PhaseArena("npu:0")
            for index in range(2):
                token = arena.begin()
                arena.end(token)
                events[index].record.assert_called_once_with(stream)
            for event in events:
                event.query.assert_not_called()
                event.synchronize.assert_not_called()
            token = arena.begin()
            events[0].query.assert_called_once_with()
            events[0].synchronize.assert_called_once_with()
            events[1].query.assert_not_called()
            arena.end(token)
            events[2].record.assert_called_once_with(stream)
            events[2].synchronize.assert_not_called()
            arena.close()
            for event in events:
                event.query.assert_called_once_with()
                event.synchronize.assert_called_once_with()

    def test_completed_oldest_does_not_block_and_close_drains_in_order(self):
        """Query avoids needless waits, and close handles every remaining completion."""
        stream, calls = object(), []
        events = [Mock(name=f"phase_{index}") for index in range(3)]
        for index, event in enumerate(events):
            event.record.side_effect = lambda _stream, index=index: calls.append(("record", index))
            event.query.side_effect = lambda index=index: calls.append(("query", index)) or index != 2
            event.synchronize.side_effect = lambda index=index: calls.append(("wait", index))
        backend = SimpleNamespace(current_stream=Mock(return_value=stream), Event=Mock(side_effect=events))
        with patch.object(arena_module.torch, "device", return_value=SimpleNamespace(type="npu")), \
                patch.object(arena_module.torch, "npu", backend, create=True):
            arena = PhaseArena("npu:0")
            for index in range(3):
                token = arena.begin()
                calls.append(("joined_TG", index))
                arena.end(token)
            self.assertEqual(calls, [("joined_TG", 0), ("record", 0), ("joined_TG", 1), ("record", 1),
                                     ("query", 0), ("joined_TG", 2), ("record", 2)])
            arena.close()
            arena.close()
        events[0].synchronize.assert_not_called()
        events[1].synchronize.assert_not_called()
        events[2].synchronize.assert_called_once_with()
        self.assertEqual(calls[-3:], [("query", 1), ("query", 2), ("wait", 2)])

    def test_release_keeps_pending_events_and_rejects_another_stream(self):
        """Retirement releases references without weakening the two-phase host bound."""
        current, other = object(), object()
        events = [Mock(), Mock()]
        for event in events:
            event.query.return_value = False
        backend = SimpleNamespace(current_stream=Mock(return_value=current), Event=Mock(side_effect=events))
        device = SimpleNamespace(type="npu")
        original_empty = torch.empty

        def _cpu_empty(elements, *, dtype, device):
            self.assertIs(device, arena.device)
            return original_empty(elements, dtype=dtype, device="cpu")

        with patch.object(arena_module.torch, "device", return_value=device), \
                patch.object(arena_module.torch, "npu", backend, create=True), \
                patch.object(arena_module.torch, "empty", side_effect=_cpu_empty):
            arena = PhaseArena("npu:0")
            for _ in range(2):
                token = arena.begin()
                arena.allocate("region", (1,), torch.float32)
                arena.end(token)
            backend.current_stream.return_value = other
            with self.assertRaisesRegex(RuntimeError, "original compute stream"):
                arena.release_regions(("region",))
            self.assertEqual(arena.reserved_bytes, _QUANTUM)
            backend.current_stream.return_value = current
            self.assertEqual(arena.release_regions(("region",)), _QUANTUM)
            for event in events:
                event.query.assert_not_called()
                event.synchronize.assert_not_called()
            token = arena.begin()
            events[0].query.assert_called_once_with()
            events[0].synchronize.assert_called_once_with()
            events[1].query.assert_not_called()
            backend.Event.side_effect = Mock
            arena.end(token)
            arena.close()
            events[1].synchronize.assert_called_once_with()
