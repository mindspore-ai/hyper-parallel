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
"""Pinned host memory pool with bucket-based management.

Provides a thread-safe pool of host page-locked CPU memory buffers for
efficient host-device data transfers during activation offloading.
"""

import logging
import threading

import torch

logger = logging.getLogger(__name__)

_BUCKET_SIZES = [2**i for i in range(10, 32)]


def _align_to_bucket(size: int) -> int:
    """Find the smallest bucket size >= size."""
    for bucket in _BUCKET_SIZES:
        if bucket >= size:
            return bucket
    return size


def _bucket_for(size: int) -> int:
    """Return the bucket size a buffer belongs to."""
    return _align_to_bucket(size)


class PinnedMemoryPool:
    """Thread-safe pool of pinned host CPU memory buffers with deferred recycling via CUDA events.

    If an acquire request exceeds the pool's remaining capacity defined by
    ``max_host_bytes``, further acquires fall back to regular pageable CPU
    memory.
    """

    def __init__(self, max_host_bytes: int) -> None:
        self._pool: dict[int, list[torch.Tensor]] = {}
        self._pending: dict[int, list[tuple[torch.Tensor, torch.Event]]] = {}
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._max_host_bytes = max_host_bytes

    @property
    def total_allocated(self) -> int:
        """Total host bytes currently held by the allocator."""
        return self._total_allocated

    @property
    def max_host_bytes(self) -> int:
        """Hard limit on host memory in bytes."""
        return self._max_host_bytes

    def _reclaim_locked(self, bucket: int) -> None:
        """Move completed tensors from pending to available pool."""
        if bucket not in self._pending:
            return

        still_pending = []
        for tensor, event in self._pending[bucket]:
            if event.query():
                self._pool.setdefault(bucket, []).append(tensor)
            else:
                still_pending.append((tensor, event))
        self._pending[bucket] = still_pending

    def acquire(self, size: int) -> torch.Tensor:
        """Obtain a buffer of at least *size* bytes from the pool."""
        bucket = _bucket_for(size)
        with self._lock:
            for bucket_size in (bucket, *(b for b in _BUCKET_SIZES if b > bucket)):
                self._reclaim_locked(bucket_size)
                entries = self._pool.get(bucket_size)
                if entries:
                    return entries.pop()[:size]

            aligned = _align_to_bucket(size)
            if self._total_allocated + aligned <= self._max_host_bytes:
                self._total_allocated += aligned
                logger.debug(
                    "PinnedMemoryPool: allocate %d bytes (total=%d, limit=%d)",
                    aligned,
                    self._total_allocated,
                    self._max_host_bytes,
                )
                return torch.empty(aligned, dtype=torch.uint8, pin_memory=True)[:size]

            self._reclaim_locked(bucket)
            if bucket in self._pending and self._pending[bucket]:
                tensor, event = self._pending[bucket].pop(0)
                event.synchronize()
                return tensor[:size]

            raise RuntimeError(
                f"PinnedMemoryPool exhausted: total_allocated={self._total_allocated}, "
                f"max_host_bytes={self._max_host_bytes}, requested={size}"
            )

    def release(self, tensor: torch.Tensor, event: torch.Event | None = None) -> None:
        """Return a previously acquired buffer to the pool for reuse."""
        if not tensor.is_pinned():
            raise ValueError(
                "release() expects a pinned (page-locked) tensor, "
                f"got tensor on {tensor.device} with pin_memory={tensor.is_pinned()}"
            )

        storage = tensor.untyped_storage()
        full_tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
        full_tensor.set_(storage, 0, (storage.size(),), (1,))

        bucket = _bucket_for(full_tensor.numel())
        with self._lock:
            if event is not None:
                self._pending.setdefault(bucket, []).append((full_tensor, event))
            else:
                self._pool.setdefault(bucket, []).append(full_tensor)
