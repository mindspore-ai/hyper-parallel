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
"""Framework-independent pool for reusable pinned host memory."""

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.platform import get_platform

platform = get_platform()

_MIN_BUCKET_BYTES = 1024
_IN_USE = "in_use"
_AVAILABLE = "available"
_PENDING = "pending"


@dataclass
class _Block:
    """A complete allocation owned by :class:`PinnedMemoryPool`."""

    buffer: Any
    capacity: int
    state: str
    event: Optional[Any] = None
    storage_key: Optional[int] = None


def _valid_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}.")


def _storage_key(tensor: Any) -> int:
    """Return a stable identity for a tensor's underlying storage."""
    try:
        return int(tensor.untyped_storage().data_ptr())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("release() expects a tensor with an identifiable storage.") from exc


class PinnedMemoryPool:
    """Thread-safe, capacity-bounded pool of pinned CPU byte buffers.

    The pool keeps complete one-dimensional ``uint8`` allocations and returns
    length-limited views from :meth:`acquire`.  A returned view may be cast to
    another dtype by the caller, but no view may be used after :meth:`release`.
    When no reusable block is available and a new pinned allocation would
    exceed ``max_host_bytes``, :meth:`acquire` raises ``RuntimeError``.

    Args:
        max_host_bytes: Maximum total capacity retained by the pool.
        align_limit: Largest size used by the power-of-two alignment policy.

    Raises:
        ValueError: If a constructor or ``acquire`` argument is invalid.
        RuntimeError: If an allocation would exceed ``max_host_bytes`` and no
            sufficient reusable block exists.
    """

    def __init__(self, max_host_bytes: int, align_limit: int = 2 * 1024**3) -> None:
        _valid_positive_int(max_host_bytes, "max_host_bytes")
        _valid_positive_int(align_limit, "align_limit")
        self._max_host_bytes = max_host_bytes
        self._align_limit = align_limit
        self._available: Dict[int, List[_Block]] = defaultdict(list)
        self._pending: Dict[int, List[_Block]] = defaultdict(list)
        self._blocks: Dict[int, _Block] = {}
        self._total_allocated = 0
        self._lock = threading.Lock()

    @property
    def total_allocated(self) -> int:
        """Return total aligned pinned capacity held by this pool."""
        with self._lock:
            return self._total_allocated

    @property
    def max_host_bytes(self) -> int:
        """Return the pool's hard capacity limit in bytes."""
        return self._max_host_bytes

    @property
    def align_limit(self) -> int:
        """Return the largest alignment bucket size in bytes."""
        return self._align_limit

    def _aligned_size(self, size: int) -> int:
        if size > self._align_limit:
            return size
        minimum = max(size, _MIN_BUCKET_BYTES)
        power = 1 << (minimum - 1).bit_length()
        aligned = min(power, self._align_limit)
        return max(aligned, size)

    def _reclaim_completed_locked(self, minimum_capacity: int) -> None:
        """Move completed pending blocks into the available buckets."""
        # A transfer batch may release many blocks with one event. Retain the
        # event objects so each runtime event is queried only once per scan.
        event_results: Dict[int, Tuple[Any, bool]] = {}
        for capacity in list(self._pending):
            if capacity < minimum_capacity:
                continue
            still_pending = []
            for block in self._pending[capacity]:
                event = block.event
                event_key = id(event)
                event_result = event_results.get(event_key)
                if event_result is None:
                    completed = event.query()
                    event_results[event_key] = (event, completed)
                else:
                    completed = event_result[1]
                if completed:
                    block.event = None
                    block.state = _AVAILABLE
                    self._available[capacity].append(block)
                else:
                    still_pending.append(block)
            if still_pending:
                self._pending[capacity] = still_pending
            else:
                del self._pending[capacity]

    def _register_view_locked(self, block: _Block, view: Any) -> Any:
        """Associate a checked-out view with its owning allocation."""
        key = _storage_key(view)
        owner = self._blocks.get(key)
        if owner is not None and owner is not block:
            raise RuntimeError("Two host buffers exposed the same storage identity.")
        old_key = block.storage_key
        if old_key is not None and old_key != key and self._blocks.get(old_key) is block:
            del self._blocks[old_key]
        self._blocks[key] = block
        block.storage_key = key
        return view

    def _checkout_locked(self, block: _Block, size: int) -> Any:
        """Create and register the exact view returned by :meth:`acquire`."""
        view = block.buffer if size == block.capacity else block.buffer[:size]
        return self._register_view_locked(block, view)

    def _find_available_locked(self, minimum_capacity: int) -> Optional[_Block]:
        for capacity in sorted(self._available):
            if capacity >= minimum_capacity and self._available[capacity]:
                block = self._available[capacity].pop()
                if not self._available[capacity]:
                    del self._available[capacity]
                block.state = _IN_USE
                return block
        return None

    def _find_pending_locked(self, minimum_capacity: int) -> Optional[_Block]:
        for capacity in sorted(self._pending):
            if capacity >= minimum_capacity and self._pending[capacity]:
                block = self._pending[capacity].pop(0)
                if not self._pending[capacity]:
                    del self._pending[capacity]
                block.state = _IN_USE
                return block
        return None

    def acquire(self, size: int) -> Any:
        """Acquire a pooled pinned CPU ``uint8`` view."""
        _valid_positive_int(size, "size")
        aligned_size = self._aligned_size(size)

        with self._lock:
            self._reclaim_completed_locked(aligned_size)
            block = self._find_available_locked(aligned_size)
            if block is not None:
                try:
                    return self._checkout_locked(block, size)
                except Exception:
                    block.state = _AVAILABLE
                    self._available[block.capacity].append(block)
                    raise

            if self._total_allocated + aligned_size <= self._max_host_bytes:
                self._total_allocated += aligned_size
                reserved = True
            else:
                reserved = False
                block = self._find_pending_locked(aligned_size)
                if block is None:
                    raise RuntimeError(
                        "PinnedMemoryPool capacity exceeded: "
                        f"requested_bytes={size}, aligned_bytes={aligned_size}, "
                        f"pooled_bytes={self._total_allocated}, "
                        f"max_host_bytes={self._max_host_bytes}."
                    )

        if reserved:
            try:
                buffer = platform.alloc_tensor_buffer(
                    aligned_size,
                    platform.tensor_dtype.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            except Exception:
                with self._lock:
                    self._total_allocated -= aligned_size
                raise
            block = _Block(buffer, aligned_size, _IN_USE)
            try:
                with self._lock:
                    view = self._checkout_locked(block, size)
            except Exception:
                with self._lock:
                    self._total_allocated -= aligned_size
                raise
            return view

        event = block.event
        try:
            event.synchronize()
        except Exception:
            with self._lock:
                block.state = _PENDING
                block.event = event
                self._pending[block.capacity].insert(0, block)
            raise
        block.event = None
        try:
            with self._lock:
                return self._checkout_locked(block, size)
        except Exception:
            with self._lock:
                block.state = _AVAILABLE
                self._available[block.capacity].append(block)
            raise

    def release(self, tensor: Any, event: Optional[Any] = None) -> None:
        """Return an acquired view to the pool, optionally after an async event."""
        key = _storage_key(tensor)
        with self._lock:
            block = self._blocks.get(key)
            if block is None:
                raise ValueError("The tensor does not belong to this PinnedMemoryPool.")
            if block.state != _IN_USE:
                raise ValueError("The tensor has already been released to this PinnedMemoryPool.")
            if event is None:
                block.state = _AVAILABLE
                self._available[block.capacity].append(block)
            else:
                block.state = _PENDING
                block.event = event
                self._pending[block.capacity].append(block)
