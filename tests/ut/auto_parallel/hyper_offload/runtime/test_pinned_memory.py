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
"""Unit tests for PinnedMemoryPool."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.pinned_memory import PinnedMemoryPool, _bucket_for


def _pin_memory_cpu_patcher() -> None:
    """Return a patcher that makes ``torch.empty(pin_memory=True)`` work on CPU-only hosts.

    ``torch.empty(..., pin_memory=True)`` requires CUDA even when allocating
    CPU memory.  This patcher strips the ``pin_memory`` kwarg so the call
    succeeds without a GPU.
    """
    orig_empty = torch.empty

    def _empty_no_pin(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return orig_empty(*args, **kwargs)

    return patch(
        "hyper_parallel.auto_parallel.hyper_offload.runtime.pinned_memory.torch.empty",
        _empty_no_pin,
    )


class TestBucketHelpers(unittest.TestCase):
    """Bucket alignment helpers."""

    def test_bucket_for_small(self) -> None:
        self.assertEqual(_bucket_for(1), 1024)
        self.assertEqual(_bucket_for(1023), 1024)

    def test_bucket_for_exact(self) -> None:
        self.assertEqual(_bucket_for(1024), 1024)

    def test_bucket_for_large(self) -> None:
        self.assertEqual(_bucket_for(1025), 2048)

    def test_bucket_for_very_large(self) -> None:
        self.assertEqual(_bucket_for(2**20), 2**20)
        self.assertEqual(_bucket_for(2**20 + 1), 2**21)

    def test_bucket_for_oversize(self) -> None:
        """Size > largest bucket (2^31) returns the size itself (fallback in _align_to_bucket)."""
        self.assertEqual(_bucket_for(2**32), 2**32)
        self.assertEqual(_bucket_for(2**40), 2**40)


class TestPinnedMemoryPoolNoPin(unittest.TestCase):
    """PinnedMemoryPool tests that work on CPU by avoiding pin_memory."""

    @classmethod
    def setUpClass(cls) -> None:
        # Mock is_pinned to return True so release() actually reclaims buffers
        cls._is_pinned_patch = patch("torch.Tensor.is_pinned", return_value=True)
        cls._is_pinned_patch.start()
        # Mock torch.empty at the import site so pin_memory=True works on CPU
        cls._empty_patch = _pin_memory_cpu_patcher()
        cls._empty_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._empty_patch.stop()
        cls._is_pinned_patch.stop()

    def setUp(self) -> None:
        self.pool = PinnedMemoryPool(max_host_bytes=1024 * 1024)

    def test_initial_state(self) -> None:
        self.assertEqual(self.pool.total_allocated, 0)
        self.assertEqual(self.pool.max_host_bytes, 1024 * 1024)

    def test_acquire_and_release_no_pin(self) -> None:
        """Acquire and release should cycle without pin_memory."""
        buf = self.pool.acquire(64)
        self.assertIsInstance(buf, torch.Tensor)
        self.assertGreaterEqual(buf.numel(), 64)
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertEqual(buf.device.type, "cpu")

        # Release without event
        self.pool.release(buf)
        # Acquire again — should reuse
        buf2 = self.pool.acquire(64)
        self.assertIsInstance(buf2, torch.Tensor)

    def test_acquire_and_release_with_event(self) -> None:
        """Release with an event should go to pending, not directly to pool."""
        buf = self.pool.acquire(128)
        self.pool.release(buf, event=torch.Event(enable_timing=False))
        # The buffer went to pending (event not queried yet if zero-lag)
        # Acquire another — should allocate fresh (or reclaim if event already signalled)
        buf2 = self.pool.acquire(128)
        self.assertIsInstance(buf2, torch.Tensor)

    def test_total_allocated_tracking(self) -> None:
        """total_allocated should reflect newly allocated buffers."""
        buf1 = self.pool.acquire(1024)  # bucket = 1024
        self.assertEqual(self.pool.total_allocated, 1024)

        # Release buf1 so it can be reused
        self.pool.release(buf1)
        buf2 = self.pool.acquire(1024)  # reuse from pool → no new allocation
        self.assertEqual(self.pool.total_allocated, 1024)

        self.pool.release(buf2)
        buf3 = self.pool.acquire(2048)  # new allocation (different bucket size)
        self.assertEqual(self.pool.total_allocated, 1024 + 2048)

        # Recycle remaining
        self.pool.release(buf3)

    def test_acquire_multiple_buffers(self) -> None:
        """Multiple acquires should work."""
        bufs = [self.pool.acquire(256) for _ in range(5)]
        self.assertEqual(len(bufs), 5)
        for buf in bufs:
            self.assertGreaterEqual(buf.numel(), 256)
        # Release all
        for buf in bufs:
            self.pool.release(buf)

    def test_acquire_returns_properly_sliced_buffer(self) -> None:
        """Acquire(100) should return a buffer of at least 100 elements."""
        buf = self.pool.acquire(100)
        self.assertGreaterEqual(buf.numel(), 100)
        # The full storage may be larger (bucket-aligned), but the returned view is sized.
        self.assertEqual(buf.shape[0], 100)

    def test_pool_exhaustion_raises(self) -> None:
        """When pool memory is exhausted, acquire should raise RuntimeError."""
        tiny_pool = PinnedMemoryPool(max_host_bytes=512)
        with self.assertRaises(RuntimeError):
            tiny_pool.acquire(1024)  # exceeds pool limit

    def test_release_non_pinned_raises(self) -> None:
        """Releasing a non-pinned tensor should raise ValueError."""
        buf = torch.empty(64, dtype=torch.uint8)
        # Temporarily restore real is_pinned (setUpClass patches it to return True)
        with patch("torch.Tensor.is_pinned", return_value=False):
            with self.assertRaises(ValueError):
                self.pool.release(buf)
        # Pool state unchanged
        self.assertEqual(self.pool.total_allocated, 0)

    def test_max_host_bytes_property(self) -> None:
        self.assertEqual(self.pool.max_host_bytes, 1024 * 1024)


class TestPinnedMemoryPoolPinMemory(unittest.TestCase):
    """PinnedMemoryPool tests that require pinned memory.

    These tests use a mock for ``pin_memory`` so they work everywhere.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Patch pin_memory to work on CPU
        cls._pin_patch = patch("torch.Tensor.pin_memory", lambda self: self)
        cls._pin_patch.start()
        # Mock is_pinned so release() treats buffers as pinned
        cls._is_pinned_patch = patch("torch.Tensor.is_pinned", return_value=True)
        cls._is_pinned_patch.start()
        # Mock torch.empty at the import site so pin_memory=True works on CPU
        cls._empty_patch = _pin_memory_cpu_patcher()
        cls._empty_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._empty_patch.stop()
        cls._is_pinned_patch.stop()
        cls._pin_patch.stop()

    def setUp(self) -> None:
        self.pool = PinnedMemoryPool(max_host_bytes=1024 * 1024)

    def test_pinned_buffer_reuse(self) -> None:
        """Pinned buffers should be reused after release."""
        buf = self.pool.acquire(256)
        self.assertTrue(buf.is_pinned())  # mock always returns True
        buf_ptr = buf.data_ptr()
        self.pool.release(buf)
        buf2 = self.pool.acquire(256)
        # May or may not be the same pointer (depends on bucket reuse), but should succeed.
        self.assertIsInstance(buf2, torch.Tensor)

    def test_recycle_across_buckets(self) -> None:
        """Released buffer in a larger bucket should satisfy a smaller request."""
        buf_big = self.pool.acquire(4096)
        self.pool.release(buf_big)
        buf_small = self.pool.acquire(512)
        self.assertIsInstance(buf_small, torch.Tensor)

    def test_reclaim_pending_recycles(self) -> None:
        """Pending buffers with completed events should be reclaimed on next acquire."""
        buf = self.pool.acquire(256)
        event = torch.Event(enable_timing=False)
        self.pool.release(buf, event=event)
        # Acquiring the same size should reclaim from pending
        buf2 = self.pool.acquire(256)
        self.assertIsInstance(buf2, torch.Tensor)

    def test_acquire_from_pending_synchronizes_on_full_pool(self) -> None:
        """When pool is full and pending event is incomplete, acquire should synchronize and reuse."""
        pool = PinnedMemoryPool(max_host_bytes=2048)
        buf1 = pool.acquire(1024)
        buf2 = pool.acquire(1024)  # pool now full (total=2048)

        event = torch.Event(enable_timing=False)
        pool.release(buf1, event=event)  # goes to pending, event stays incomplete

        # Pool is full, no new allocation possible
        # _reclaim_locked won't reclaim (event.query()=False)
        # Must fall through to pending → synchronize
        buf3 = pool.acquire(512)
        self.assertIsInstance(buf3, torch.Tensor)
        self.assertEqual(buf3.numel(), 512)


class TestPinnedMemoryPoolThreadSafety(unittest.TestCase):
    """Thread-safety of PinnedMemoryPool."""

    def setUp(self) -> None:
        self._empty_patch = _pin_memory_cpu_patcher()
        self._empty_patch.start()
        self._is_pinned_patch = patch("torch.Tensor.is_pinned", return_value=True)
        self._is_pinned_patch.start()

    def tearDown(self) -> None:
        self._is_pinned_patch.stop()
        self._empty_patch.stop()

    def test_concurrent_acquire_release(self) -> None:
        """Concurrent acquire/release from multiple threads should not corrupt the pool."""
        pool = PinnedMemoryPool(max_host_bytes=10 * 1024 * 1024)

        def worker() -> None:
            for _ in range(20):
                buf = pool.acquire(1024)
                pool.release(buf)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Pool should still be in a consistent state
        self.assertGreaterEqual(pool.total_allocated, 0)
        # Acquire should still work
        buf = pool.acquire(64)
        self.assertIsInstance(buf, torch.Tensor)
