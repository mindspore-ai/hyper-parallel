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
"""Unit tests for the activation checkpoint pinned memory pool."""

import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.activation_checkpoint import pinned_memory_pool
from hyper_parallel.core.activation_checkpoint.pinned_memory_pool import PinnedMemoryPool


class _Event:
    """Controllable event implementing the pool's minimal event protocol."""

    def __init__(self, complete: bool) -> None:
        self.complete = complete
        self.query_count = 0
        self.synchronize_count = 0

    def query(self) -> bool:
        self.query_count += 1
        return self.complete

    def synchronize(self) -> None:
        self.synchronize_count += 1
        self.complete = True


class _SyntheticStorage:
    """Storage exposing a caller-controlled identity."""

    def __init__(self, key: int) -> None:
        self.key = key

    def data_ptr(self) -> int:
        return self.key


class _SyntheticView:
    """Tensor-like object used to emulate framework storage views."""

    def __init__(self, key: int) -> None:
        self.storage = _SyntheticStorage(key)

    def untyped_storage(self) -> _SyntheticStorage:
        return self.storage


class _SyntheticBuffer(_SyntheticView):
    """Buffer whose slices expose identities different from the base buffer."""

    def __init__(self, key: int, view_keys) -> None:
        super().__init__(key)
        self.view_keys = iter(view_keys)

    def __getitem__(self, index) -> _SyntheticView:
        del index
        return _SyntheticView(next(self.view_keys))


class TestPinnedMemoryPool(unittest.TestCase):
    """Hardware-independent tests using ordinary CPU tensors as pinned buffers."""

    def setUp(self) -> None:
        self.mock_platform = MagicMock()
        self.mock_platform.tensor_dtype.uint8 = torch.uint8
        self.mock_platform.alloc_tensor_buffer.side_effect = (
            lambda numel, dtype, device, pin_memory: torch.empty(numel, dtype=dtype)
        )
        self.platform_patch = patch.object(pinned_memory_pool, "platform", self.mock_platform)
        self.platform_patch.start()

    def tearDown(self) -> None:
        self.platform_patch.stop()

    def test_validates_constructor_and_acquire_arguments(self) -> None:
        """Reject invalid constructor limits and acquisition sizes."""
        for invalid in (0, -1, 1.5, True):
            with self.subTest(max_host_bytes=invalid), self.assertRaises(ValueError):
                PinnedMemoryPool(invalid)
            with self.subTest(align_limit=invalid), self.assertRaises(ValueError):
                PinnedMemoryPool(1024, align_limit=invalid)

        pool = PinnedMemoryPool(1024)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                pool.acquire(invalid)

    def test_alignment_policy_and_properties(self) -> None:
        """Apply alignment rules and expose configured pool properties."""
        pool = PinnedMemoryPool(8192, align_limit=1500)
        first = pool.acquire(600)
        second = pool.acquire(1200)
        third = pool.acquire(2500)

        self.assertEqual(first.numel(), 600)
        self.assertEqual(second.numel(), 1200)
        self.assertEqual(third.numel(), 2500)
        self.assertEqual(pool.total_allocated, 1024 + 1500 + 2500)
        self.assertEqual(pool.max_host_bytes, 8192)
        self.assertEqual(pool.align_limit, 1500)

    def test_exact_capacity_checkout_returns_base_buffer(self) -> None:
        buffer = _SyntheticBuffer(100, view_keys=())
        self.mock_platform.alloc_tensor_buffer.side_effect = [buffer]
        pool = PinnedMemoryPool(1024)

        acquired = pool.acquire(1024)

        self.assertIs(acquired, buffer)

    def test_available_blocks_use_best_fit(self) -> None:
        """Reuse the smallest available block that satisfies each request."""
        pool = PinnedMemoryPool(4096)
        small = pool.acquire(600)
        large = pool.acquire(1500)
        small_ptr = small.untyped_storage().data_ptr()
        large_ptr = large.untyped_storage().data_ptr()
        pool.release(small)
        pool.release(large)

        reused_small = pool.acquire(600)
        reused_large = pool.acquire(1200)

        self.assertEqual(reused_small.untyped_storage().data_ptr(), small_ptr)
        self.assertEqual(reused_large.untyped_storage().data_ptr(), large_ptr)
        self.assertEqual(pool.total_allocated, 3072)

    def test_capacity_limit_raises_when_no_block_can_be_reused(self) -> None:
        """Raise when the capacity limit prevents allocation or block reuse."""
        pool = PinnedMemoryPool(1500, align_limit=1500)
        pool.acquire(600)

        with self.assertRaisesRegex(RuntimeError, "PinnedMemoryPool capacity exceeded") as caught:
            pool.acquire(600)

        self.assertEqual(pool.total_allocated, 1024)
        error = str(caught.exception)
        self.assertIn("requested_bytes=600", error)
        self.assertIn("aligned_bytes=1024", error)
        self.assertIn("pooled_bytes=1024", error)
        self.assertIn("max_host_bytes=1500", error)
        self.assertEqual(self.mock_platform.alloc_tensor_buffer.call_count, 1)

    def test_completed_pending_block_is_reclaimed_without_wait(self) -> None:
        """Reuse a completed pending block without synchronizing its event."""
        pool = PinnedMemoryPool(1024)
        buffer = pool.acquire(600)
        pointer = buffer.untyped_storage().data_ptr()
        event = _Event(complete=True)
        pool.release(buffer, event=event)

        reused = pool.acquire(600)

        self.assertEqual(reused.untyped_storage().data_ptr(), pointer)
        self.assertEqual(event.synchronize_count, 0)

    def test_shared_pending_event_is_queried_once_per_reclaim(self) -> None:
        """Query a shared event only once during a reclaim scan."""
        pool = PinnedMemoryPool(3072)
        small = pool.acquire(600)
        large = pool.acquire(1500)
        small_pointer = small.untyped_storage().data_ptr()
        large_pointer = large.untyped_storage().data_ptr()
        event = _Event(complete=True)
        pool.release(small, event=event)
        pool.release(large, event=event)

        reused_small = pool.acquire(600)
        reused_large = pool.acquire(1200)

        self.assertEqual(event.query_count, 1)
        self.assertEqual(reused_small.untyped_storage().data_ptr(), small_pointer)
        self.assertEqual(reused_large.untyped_storage().data_ptr(), large_pointer)

    def test_full_pool_waits_for_sufficient_pending_block(self) -> None:
        """Wait for a sufficiently large pending block when the pool is full."""
        pool = PinnedMemoryPool(1024)
        buffer = pool.acquire(600)
        pointer = buffer.untyped_storage().data_ptr()
        event = _Event(complete=False)
        pool.release(buffer, event=event)

        reused = pool.acquire(500)

        self.assertEqual(reused.untyped_storage().data_ptr(), pointer)
        self.assertEqual(event.synchronize_count, 1)

    def test_typed_view_release_and_invalid_releases(self) -> None:
        """Support typed views and reject invalid release operations."""
        pool = PinnedMemoryPool(1024)
        other_pool = PinnedMemoryPool(1024)
        raw = pool.acquire(16)
        typed = raw.view(torch.float32)
        pool.release(typed)

        with self.assertRaisesRegex(ValueError, "already been released"):
            pool.release(raw)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            other_pool.release(raw)
        with self.assertRaisesRegex(ValueError, "identifiable storage"):
            pool.release(object())

    def test_tracks_storage_identity_of_each_returned_view(self) -> None:
        """Replace stale storage identities when returning a new view."""
        self.mock_platform.alloc_tensor_buffer.side_effect = [
            _SyntheticBuffer(100, view_keys=(101, 102))
        ]
        pool = PinnedMemoryPool(1024)

        first = pool.acquire(600)
        pool.release(first)
        second = pool.acquire(600)

        self.assertNotIn(101, pool._blocks)  # pylint: disable=protected-access
        self.assertIn(102, pool._blocks)  # pylint: disable=protected-access
        pool.release(second)

    def test_allocation_failure_rolls_back_reserved_capacity(self) -> None:
        pool = PinnedMemoryPool(1024)
        self.mock_platform.alloc_tensor_buffer.side_effect = RuntimeError("allocation failed")

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            pool.acquire(600)
        self.assertEqual(pool.total_allocated, 0)

    def test_pool_instances_are_isolated(self) -> None:
        first = PinnedMemoryPool(1024)
        second = PinnedMemoryPool(2048)

        first.acquire(600)

        self.assertEqual(first.total_allocated, 1024)
        self.assertEqual(second.total_allocated, 0)
        second.acquire(1500)
        self.assertEqual(second.total_allocated, 2048)

    def test_concurrent_acquire_release_keeps_pool_consistent(self) -> None:
        """Keep allocation accounting consistent during concurrent reuse."""
        pool = PinnedMemoryPool(8 * 1024)
        errors = []

        def worker() -> None:
            try:
                for _ in range(20):
                    buffer = pool.acquire(600)
                    pool.release(buffer)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(pool.total_allocated, pool.max_host_bytes)
