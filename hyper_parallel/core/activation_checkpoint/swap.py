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
"""Swap tensor and swap manager implementation for activation checkpointing"""
# pylint: disable=W0212

import functools
import threading
import warnings

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from hyper_parallel.platform import get_platform

platform = get_platform()

# ---------------------------------------------------------------------------
# Module-level buffer pools — process-local, no locking needed for single-
# stream training.  Each GPU process owns its own Python interpreter, so
# these dicts are never shared across processes.
#
# _CPU_PINNED_POOL: a list of available pinned CPU tensors per dtype_key.
#   Created via alloc_tensor_buffer(pin_memory=True) on the first miss; the
#   base tensor is returned here after wait_load and reused in the next
#   launch_offload, avoiding repeated cudaHostAlloc / cudaFreeHost calls.
# ---------------------------------------------------------------------------
_CPU_PINNED_POOL: Dict[str, List[Any]] = defaultdict(list)
# Cap each group-swap staging allocation. 32 MiB keeps DMA chunks large
# while avoiding one huge per-dtype staging tensor in large models.
_GROUP_SWAP_MAX_BULK_COPY_BYTES = 32 * 1024 * 1024


def _get_cpu_pinned_buf(dtype_key: str, total_numel: int, dtype):
    """Pop the smallest sufficient pinned buffer from the pool, or allocate.

    Best-fit selection minimises wasted pinned memory.  When no buffer in the
    pool is large enough, an undersized entry is discarded before allocating a
    fresh buffer via alloc_tensor_buffer.

    Returns the *full* buffer (capacity >= total_numel).  Callers must slice
    ``buf[:total_numel]`` for the actual copy so the returned reference can be
    passed back to :func:`_return_cpu_pinned_buf` without any platform-specific
    introspection.
    """
    pool = _CPU_PINNED_POOL[dtype_key]
    best_i = -1
    for i, buf in enumerate(pool):
        if buf.numel() >= total_numel:
            if best_i == -1 or buf.numel() < pool[best_i].numel():
                best_i = i
    if best_i != -1:
        return pool.pop(best_i)
    # No suitable buffer — discard one stale undersized entry.
    if pool:
        pool.pop()
    return platform.alloc_tensor_buffer(total_numel, dtype, device='cpu', pin_memory=True)


def _return_cpu_pinned_buf(buf):
    """Return a full pinned CPU buffer to the pool for reuse."""
    if buf is None:
        return
    _CPU_PINNED_POOL[str(buf.dtype)].append(buf)


def _collect_device_storage_ptrs(tensors: Any) -> Set[int]:
    """Collect device storage pointers from a nested tensor structure."""
    storage_ptrs = set()

    def _collect(x):
        if isinstance(x, platform.Tensor) and str(x.device).lower() != "cpu":
            storage_ptrs.add(x.untyped_storage().data_ptr())
        return x

    platform.tree_map(_collect, tensors)
    return storage_ptrs


class SwapTensor:
    """A tensor that can be swapped between device and host memory asynchronously."""
    STATE_DEVICE = "device"
    STATE_HOST = "host"
    STATE_D2H = "d2h"
    STATE_H2D = "h2d"
    STATE_NON_TENSOR = "non_tensor"

    def __init__(self, val: Any, funcname: str, group_swap: bool = False) -> None:
        self.val = val
        self.funcname = funcname
        self._keep_on_device = False
        self._duplicate_swap = False
        self._group_managed = False # True when this tensor is handled by SwapGroup bulk copy
        self.group_swap = group_swap # opt-in for group copy fusion (MUST_SWAP tensors only)
        if isinstance(val, platform.Tensor) and str(val.device).lower() != 'cpu':
            self.ver = val._version
            self._state = self.STATE_DEVICE
            val_storage = val.untyped_storage()
            self.storage_size = val_storage.size()
            self.is_slice_tensor = self.storage_size != val.numel() * platform.get_element_size(val)
            self.val_cpu = None
        else:
            self.ver = None
            self._state = self.STATE_NON_TENSOR
            self.val_cpu = None
            self.is_slice_tensor = False
            self.storage_size = 0

    def dedup_key(self):
        """Return a stable identity key for duplicate-swap detection."""
        if self._state == self.STATE_NON_TENSOR:
            return None
        val_storage = self.val.untyped_storage()
        return (
            str(self.val.device),
            val_storage.data_ptr(),
            self.val.storage_offset(),
            val_storage.size(),
            tuple(self.val.stride()),
        )

    def mark_duplicate_swap(self) -> None:
        """Mark this wrapper as a duplicate registration in the same swap group."""
        self._duplicate_swap = True

    def protect_if_aliases(self, alias_storage_ptrs: Set[int]) -> None:
        """Keep tensors that alias externally-owned tensors on device."""
        if self._state == self.STATE_NON_TENSOR:
            return
        if self.val.untyped_storage().data_ptr() in alias_storage_ptrs:
            self._keep_on_device = True

    def get_val(self) -> Any:
        """Return the underlying tensor value.

        Raises RuntimeError if the tensor is not currently in the 'device' state.
        Non-tensor values are returned directly regardless of state.
        """
        if self._state == self.STATE_NON_TENSOR:
            return self.val
        if self._state != self.STATE_DEVICE:
            raise RuntimeError(
                f"Cannot call get_val(): tensor is in '{self._state}' state. "
                f"Must be in 'device' state."
            )
        return self.val

    def resize_device_storage(self):
        """Reallocate device memory on compute stream."""
        if self._state == self.STATE_NON_TENSOR or self._duplicate_swap:
            return
        if self._group_managed:
            return

        if self._state != self.STATE_HOST:
            return
        storage = self.val.untyped_storage()
        if storage.size() == self.storage_size:
            return
        storage.resize_(self.storage_size)

    def async_load(self):
        """async load tensor from host to device"""
        if self._state == self.STATE_NON_TENSOR or self._keep_on_device or self._duplicate_swap:
            return
        if self._group_managed:
            return

        if self._state != self.STATE_HOST:
            warnings.warn(
                f"[SwapTensor.async_load] Invalid state: current={self._state}, "
                f"expected 'host'. Operation skipped."
            )
            return

        if self.val_cpu is None:
            raise ValueError("val_cpu must not be None during async_load")
        with platform.preserve_version_counter(self.val):
            if self.is_slice_tensor:
                self.val.data.copy_(self.val_cpu, non_blocking=True)
            else:
                self.val.untyped_storage().copy_(self.val_cpu.untyped_storage(), non_blocking=True)
        self._state = self.STATE_H2D

    def wait_load(self):
        """change state to device after async load is done"""
        if self._state == self.STATE_NON_TENSOR or self._keep_on_device or self._duplicate_swap:
            return

        if self._state == self.STATE_DEVICE:
            return  # already loaded
        if self._state != self.STATE_H2D:
            warnings.warn(
                f"[SwapTensor.wait_load] Called in invalid state: {self._state}. "
                f"Expected 'h2d'. Skipped."
            )
            return
        self._state = self.STATE_DEVICE

    def async_offload(self):
        """async offload tensor from device to host"""
        if self._state == self.STATE_NON_TENSOR or self._keep_on_device or self._duplicate_swap:
            return
        if self._group_managed:
            return

        if self._state != self.STATE_DEVICE:
            warnings.warn(
                f"[SwapTensor.async_offload] Invalid state: current={self._state}, "
                f"expected 'device'. Operation skipped."
            )
            return

        if self.storage_size != self.val.untyped_storage().size():
            raise RuntimeError(
                f"There is a tensor from {self.funcname} cannot be SWAPPED! Its storage has been resized "
                f"presize:{self.storage_size}, current size:{self.val.untyped_storage().size()}"
            )
        if self.ver != self.val._version:
            raise RuntimeError(
                f"There is a tensor from {self.funcname} cannot be SWAPPED! In-place modification happened "
                f"preversion:{self.ver}, current version:{self.val._version}"
            )

        if self.val_cpu is None:
            self.val_cpu = platform.empty_like(
                self.val, device="cpu", pin_memory=True
            )
        if self.is_slice_tensor:
            self.val_cpu.copy_(self.val, non_blocking=True)
        else:
            self.val_cpu.untyped_storage().copy_(self.val.untyped_storage(), non_blocking=True)
        self._state = self.STATE_D2H

    def wait_offload(self):
        """wait offload to host and free device memory"""
        if self._state == self.STATE_NON_TENSOR or self._keep_on_device or self._duplicate_swap:
            return

        if self._state == self.STATE_HOST:
            return
        if self._state != self.STATE_D2H:
            warnings.warn(
                f"[SwapTensor.wait_offload] Called in invalid state: {self._state}. "
                f"Expected 'd2h'. Skipped."
            )
            return
        storage = self.val.untyped_storage()
        if storage.size() != 0:
            storage.resize_(0)
        self._state = self.STATE_HOST

    @property
    def state(self) -> str:
        """Return the current swap state of this tensor (device, host, d2h, h2d, or non_tensor)."""
        return self._state

    def __repr__(self):
        if self._state == self.STATE_NON_TENSOR:
            return f"<SwapTensor state=non_tensor, val_type={type(self.val).__name__}>"
        return (
            f"<SwapTensor state={self._state}, duplicate={self._duplicate_swap}, "
            f"device_val={'exists' if self.val is not None else 'None'}>"
        )


class Storage:
    """Manage a collection of tensors for swapping operations.

    Supports dict-like access: ``storage[key].append(item)``, ``storage.clear()``,
    ``for batch in storage.values(): ...``.
    """

    def __init__(self):
        self._data: Dict[Any, List[Any]] = defaultdict(list)

    def __getitem__(self, key: Any) -> List[Any]:
        return self._data[key]

    def values(self):
        """Return an iterable view of all stored lists."""
        return self._data.values()

    def clear(self):
        """Remove all entries from the storage."""
        self._data.clear()

    def iter_swap_tensors(self):
        """Iterate all SwapTensor objects stored in this storage."""
        collected = []

        def _collect(x):
            if isinstance(x, SwapTensor):
                collected.append(x)
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_collect, item)
        return collected

    def mark_duplicate_swaps(self, seen_keys) -> int:
        """Mark tensors already registered in the same swap group as duplicates."""
        duplicate_count = 0
        for swap_tensor in self.iter_swap_tensors():
            dedup_key = swap_tensor.dedup_key()
            if dedup_key is None:
                continue
            if dedup_key in seen_keys:
                swap_tensor.mark_duplicate_swap()
                duplicate_count += 1
                continue
            seen_keys.add(dedup_key)
        return duplicate_count

    def protect_alias_storage_ptrs(self, alias_storage_ptrs: Set[int]):
        """Avoid offloading swap entries that alias externally-owned storage."""
        if not alias_storage_ptrs:
            return

        def _protect_tensor(x):
            if isinstance(x, SwapTensor):
                x.protect_if_aliases(alias_storage_ptrs)
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_protect_tensor, item)

    def launch_load(self):
        """launch async load for all tensors in swap storage"""
        def _async_load(x):
            if isinstance(x, SwapTensor):
                x.async_load()
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_async_load, item)

    def resize_device_storage(self):
        """Resize device storage for all swap tensors (runs on compute stream)."""
        def _resize(x):
            if isinstance(x, SwapTensor):
                x.resize_device_storage()
            return x
        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_resize, item)

    def wait_load(self):
        """wait load for all tensors in swap storage"""
        def _wait_load(x):
            if isinstance(x, SwapTensor):
                x.wait_load()
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_wait_load, item)
        self.clear()

    def wait_offload(self):
        """wait offload for all tensors in swap storage"""
        def _wait_offload(x):
            if isinstance(x, SwapTensor):
                x.wait_offload()
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_wait_offload, item)

    def launch_offload(self):
        """launch async offload for all tensors in swap storage"""
        def _async_offload(x):

            if isinstance(x, SwapTensor):
                x.async_offload()
            return x

        for storage_list in self.values():
            for item in storage_list:
                platform.tree_map(_async_offload, item)


class SwapGroup:
    """Manager for a group of storages to coordinate swap operations.

    Non-slice tensors within the group are packed into bounded contiguous device
    buffers before D2H transfer, and loaded back from bounded H2D buffers.
    Each tensor then aliases its slice of the relevant buffer via
    ``Tensor.set_()``, avoiding per-tensor memory fragmentation.

    Slice tensors (storage larger than logical data) fall back to the original
    per-tensor copy path.
    """

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.is_last_group: bool = False
        self._storages: List[Storage] = []
        self._load_event: Optional[Any] = None
        self._offload_event: Optional[Any] = None
        # Group-level contiguous buffers for non-slice tensors.
        self._packed_tensor_info: List = []   # [(SwapTensor, bucket_key, element_offset), ...]
        self._packed_buckets: Dict[str, Dict[str, Any]] = {}
        self._group_cpu_buf = None            # pinned CPU bufs; live offload→load
        self._group_device_buf = None         # temp device bufs; cleared after each phase
        # Persistent dedup set accumulated across add() calls; avoids O(N²) rebuild.
        # mark_duplicate_swaps mutates it in-place, so new keys are added automatically.
        # Reset at wait_load() so stale data_ptrs don't leak into the next iteration.
        self._seen_dedup_keys: set = set()
        # Per-bucket SwapTensor lists built in _collect_packable_tensors and consumed
        # in launch_offload, eliminating a redundant pass over _packed_tensor_info.
        self._packed_by_bucket: Dict[str, List] = {}

    def add(self, storage):
        """Add a storage to the swap group."""
        duplicate_count = storage.mark_duplicate_swaps(self._seen_dedup_keys)
        if duplicate_count > 0:
            warnings.warn(
                f"SwapGroup '{self.group_name}' skipped {duplicate_count} duplicate tensor swap registration(s)."
            )
        self._storages.append(storage)

    def protect_alias_tensors(self, tensors: Any):
        """Protect externally-owned tensors from premature offload."""
        alias_storage_ptrs = _collect_device_storage_ptrs(tensors)
        if not alias_storage_ptrs:
            return
        for storage in self._storages:
            storage.protect_alias_storage_ptrs(alias_storage_ptrs)

    def _collect_packable_tensors(self) -> int:
        """Identify tensors eligible for group packing and mark them for bulk copy.

        A tensor is eligible only when it is contiguous, not a slice tensor,
        not a duplicate, not sharing storage with another live swap tensor, and
        has ``group_swap=True``.  Dtype buckets are split before their staging
        allocation would exceed ``_GROUP_SWAP_MAX_BULK_COPY_BYTES``.  A packed
        bucket with fewer than two tensors is left on the original per-tensor
        path because it has no batch-copy benefit.  Non-contiguous
        tensors are excluded because the packing step copies storage-order
        bytes while restore uses the original stride; those tensors fall back to
        the per-tensor copy path.
        Shared-storage tensors also fall back together because group packing
        frees the original storage after packing, which would invalidate any
        non-packed aliases such as transpose views before their own offload.

        Side effects: marks each eligible tensor with ``_group_managed=True``
        and ``_state=STATE_D2H``, and populates ``_packed_tensor_info`` /
        ``_packed_buckets``.

        Returns:
            Total byte count of all packable tensors.
        """
        candidate_buckets: Dict[str, List[Dict[str, Any]]] = {}
        packed_info: List = []
        packed_buckets: Dict[str, Dict[str, Any]] = {}
        packed_by_bucket: Dict[str, List] = {}
        total_bytes = 0

        def _try_pack(x):
            if not isinstance(x, SwapTensor):
                return x
            no_pack = (not x.group_swap or x._state != SwapTensor.STATE_DEVICE or x._keep_on_device
                       or x.is_slice_tensor or x._duplicate_swap or x.storage_size >= _GROUP_SWAP_MAX_BULK_COPY_BYTES
                       or not x.val.is_contiguous())
            if no_pack:
                return x
            if x.storage_size != x.val.untyped_storage().size():
                raise RuntimeError(
                    f"There is a tensor from {x.funcname} cannot be SWAPPED! Its storage has been resized "
                    f"presize:{x.storage_size}, current size:{x.val.untyped_storage().size()}"
                )
            if x.ver != x.val._version:
                raise RuntimeError(
                    f"There is a tensor from {x.funcname} cannot be SWAPPED! In-place modification happened "
                    f"preversion:{x.ver}, current version:{x.val._version}"
                )
            dtype_key = str(x.val.dtype)
            dtype_buckets = candidate_buckets.setdefault(dtype_key, [])
            if (not dtype_buckets or
                    dtype_buckets[-1]["total_bytes"] + x.storage_size > _GROUP_SWAP_MAX_BULK_COPY_BYTES):
                dtype_buckets.append({
                    "bucket_key": f"{dtype_key}#{len(dtype_buckets)}",
                    "dtype": x.val.dtype,
                    "dtype_key": dtype_key,
                    "device": x.val.device,
                    "tensors": [],
                    "total_bytes": 0,
                    "total_numel": 0,
                })
            bucket = dtype_buckets[-1]
            bucket["tensors"].append(x)
            bucket["total_bytes"] += x.storage_size
            bucket["total_numel"] += x.val.numel()
            return x

        for storage in self._storages:
            for storage_list in storage.values():
                for item in storage_list:
                    platform.tree_map(_try_pack, item)

        for dtype_bucket_list in candidate_buckets.values():
            for candidate_bucket in dtype_bucket_list:
                tensors = candidate_bucket["tensors"]
                if len(tensors) < 2:
                    continue
                bucket_key = candidate_bucket["bucket_key"]
                packed_buckets[bucket_key] = {
                    "dtype": candidate_bucket["dtype"],
                    "dtype_key": candidate_bucket["dtype_key"],
                    "device": candidate_bucket["device"],
                    "total_numel": candidate_bucket["total_numel"],
                }
                element_offset = 0
                for tensor in tensors:
                    tensor._group_managed = True
                    tensor._state = SwapTensor.STATE_D2H
                    packed_info.append((tensor, bucket_key, element_offset))
                    element_offset += tensor.val.numel()
                packed_by_bucket[bucket_key] = tensors
                total_bytes += candidate_bucket["total_bytes"]

        self._packed_tensor_info = packed_info
        self._packed_buckets = packed_buckets
        self._packed_by_bucket = packed_by_bucket
        return total_bytes

    def launch_offload(self, copy_stream):
        """Launch async offload for all storages in the group.

        Non-slice tensors are first packed into bounded contiguous device
        buffers, then transferred to pinned CPU memory.  Slice tensors are
        offloaded individually via the existing per-tensor path.
        """
        total_bytes = self._collect_packable_tensors()
        with platform.no_grad():
            if total_bytes > 0:
                group_device_bufs = {}
                group_cpu_bufs = {}
                for bucket_key, swap_tensors in self._packed_by_bucket.items():
                    group_device_bufs[bucket_key] = platform.cat(
                        [st.val.reshape(-1) for st in swap_tensors], dim=0
                    )

        compute_event = platform.new_event()
        compute_event.record(platform.get_current_stream())
        self._offload_event = platform.new_event()
        stream_context = platform.get_stream_context()
        with platform.no_grad(), stream_context(copy_stream):
            compute_event.wait(copy_stream)

            if total_bytes > 0:
                # One-shot D2H per packed bucket. MindSpore requires tensor/storage dtype consistency.
                for bucket_key, bucket in self._packed_buckets.items():
                    dtype_key = bucket["dtype_key"]
                    numel = bucket["total_numel"]
                    cpu_buf = _get_cpu_pinned_buf(dtype_key, numel, bucket["dtype"])
                    group_cpu_bufs[bucket_key] = cpu_buf
                    cpu_buf[:numel].copy_(group_device_bufs[bucket_key], non_blocking=True)
                self._group_device_buf = group_device_bufs
                self._group_cpu_buf = group_cpu_bufs

            # Slice tensors use the existing per-tensor path.
            # Group-managed tensors are already STATE_D2H so async_offload is a no-op.
            for storage in self._storages:
                storage.launch_offload()
            self._offload_event.record(copy_stream)

    def wait_offload(self):
        """Wait for offload to complete for all storages in the group."""
        if self._offload_event is None:
            raise RuntimeError(
                f"SwapGroup '{self.group_name}' wait_offload() called before launch_offload()."
            )
        compute_stream = platform.get_current_stream()
        stream_context = platform.get_stream_context()
        with platform.no_grad(), stream_context(compute_stream):
            self._offload_event.wait(compute_stream)
            self._offload_event = None
            for storage in self._storages:
                storage.wait_offload()
        # Release the temporary device packing buffer; _group_cpu_buf persists until launch_load.
        self._group_device_buf = None

    def launch_load(self, copy_stream):
        """Prepare storage and launch async load for all storages in the group.

        Non-slice tensors are loaded from pinned CPU memory into bounded
        contiguous device buffers.  Tensors will alias their slice of the
        relevant buffer after ``wait_load``.  Slice tensors use the existing
        per-tensor path.
        """
        # Resize device storage for slice tensors only.
        # Group-managed tensors skip resize_device_storage via _group_managed flag.
        with platform.no_grad():
            for storage in self._storages:
                storage.resize_device_storage()

        compute_event = platform.new_event()
        compute_event.record(platform.get_current_stream())
        self._load_event = platform.new_event()
        stream_context = platform.get_stream_context()
        with platform.no_grad(), stream_context(copy_stream):
            compute_event.wait(copy_stream)

            if self._packed_tensor_info and self._group_cpu_buf is not None:
                group_device_bufs = {}
                for bucket_key, bucket in self._packed_buckets.items():
                    cpu_buf = self._group_cpu_buf.get(bucket_key)
                    if cpu_buf is None:
                        continue
                    numel = bucket["total_numel"]
                    group_device_bufs[bucket_key] = platform.alloc_tensor_buffer(
                        numel, bucket["dtype"], bucket["device"]
                    )
                    # One-shot H2D per packed bucket.
                    group_device_bufs[bucket_key].copy_(cpu_buf[:numel], non_blocking=True)
                self._group_device_buf = group_device_bufs
                # Mirror async_load's STATE_H2D transition: H2D is in flight.
                for st, _, _ in self._packed_tensor_info:
                    st._state = SwapTensor.STATE_H2D

            # Slice tensors use the existing per-tensor path.
            # Group-managed tensors skip async_load via _group_managed flag.
            for storage in self._storages:
                storage.launch_load()    # Only copy, no resize
            self._load_event.record(copy_stream)

    def wait_load(self):
        """Wait for load to complete for all storages in the group.

        After the H2D transfer completes, each group-managed tensor is made to
        alias its slice of the contiguous device buffer via ``Tensor.set_()``.
        The buffer stays alive through the tensors' own storage references after
        ``_group_device_buf`` is cleared here.
        """
        if self._load_event is None:
            raise RuntimeError(
                f"SwapGroup '{self.group_name}' wait_load() called before launch_load()."
            )
        compute_stream = platform.get_current_stream()
        stream_context = platform.get_stream_context()
        with platform.no_grad(), stream_context(compute_stream):
            self._load_event.wait(compute_stream)
            self._load_event = None
            # Restore group-managed tensors: alias into the contiguous device buffer.
            if self._group_device_buf is not None:
                prev_key = None
                group_storage = None
                for st, bucket_key, element_offset in self._packed_tensor_info:
                    if bucket_key != prev_key:
                        group_device_buf = self._group_device_buf.get(bucket_key)
                        group_storage = group_device_buf.untyped_storage() if group_device_buf is not None else None
                        prev_key = bucket_key
                    if group_storage is None:
                        continue
                    with platform.preserve_version_counter(st.val):
                        st.val.set_(group_storage, element_offset, st.val.shape, st.val.stride())
                    st._state = SwapTensor.STATE_DEVICE
            for storage in self._storages:
                storage.wait_load()
        self._storages.clear()
        # Return CPU pinned buffers to the pool.  By the time wait_load
        # returns, _load_event has fired on the compute stream, which
        # means the copy stream's H2D transfer has completed and the CPU
        # buffer is no longer being read by the DMA engine.  The next
        # launch_offload (start of the following iteration) will pop these
        # buffers from the pool, well after the current H2D is done.
        if self._group_cpu_buf is not None:
            for buf in self._group_cpu_buf.values():
                _return_cpu_pinned_buf(buf)
        self._group_cpu_buf = None
        # Device buffer: the pool holds the staging reference; just drop
        # the local reference.  Tensors aliasing _group_device_buf's
        # storage keep it alive via their own storage references until
        # they are consumed in backward.
        self._group_device_buf = None
        self._packed_tensor_info = []
        self._packed_buckets = {}
        self._packed_by_bucket = {}
        self._seen_dedup_keys = set()


class SwapManager:
    """Singleton manager for swap groups and their operations."""
    _instance: Optional["SwapManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, '_groups'):
            return
        self._groups: Dict[str, SwapGroup] = {}
        self._current_group_name: str = ""
        self._layer_count: int = 0
        self._copy_stream: Optional[Any] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def add_storage(self, group_name: str, storage: Storage) -> None:
        """Add a storage to a specified swap group."""
        self.ensure_group(group_name)
        self._groups[group_name].add(storage)

    def ensure_group(self, group_name: str) -> None:
        """Create the swap group if it does not exist yet."""
        if group_name not in self._groups:
            self._groups[group_name] = SwapGroup(group_name)

    def launch_offload(self, group_name: str, copy_stream=None):
        """Launch async offload for a specified swap group."""
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        if copy_stream is None:
            copy_stream = self._get_copy_stream()
        group.launch_offload(copy_stream)

    def protect_alias_tensors(self, group_name: str, tensors: Any):
        """Keep tensors that alias externally-owned tensors on device."""
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        group.protect_alias_tensors(tensors)

    def wait_offload(self, group_name: str):
        """Wait for offload to complete for a specified swap group."""
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        group.wait_offload()

    def launch_load(self, group_name: str, copy_stream=None):
        """Launch async load for a specified swap group."""
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        if copy_stream is None:
            copy_stream = self._get_copy_stream()
        group.launch_load(copy_stream)

    def wait_load(self, group_name: str):
        """Wait for load to complete for a specified swap group."""
        group = self._groups.get(group_name)
        if group is None:
            raise RuntimeError(f"Group {group_name} does not exist.")
        group.wait_load()

    def release_group_storage(self, group_name: str) -> None:
        """Release live storage references held by the swap group.

        Called at the end of backward to free Storage objects that were never
        released via wait_load (e.g. the last layer, which has no next layer
        and therefore never goes through the offload-load cycle).
        """
        group = self._groups.get(group_name)
        if group is not None:
            group._storages.clear()

    def get_current_group_name(self) -> str:
        """Return the name of the currently active swap group."""
        return self._current_group_name

    def set_current_group_name(self, group_name: str) -> None:
        """Set the name of the currently active swap group."""
        self._current_group_name = group_name

    def is_last_group(self, group_name: Optional[str] = None) -> bool:
        """Return whether the specified swap group is the terminal group in the chain."""
        group_name = self._current_group_name if group_name is None else group_name
        group = self._groups.get(group_name)
        if group is None:
            return False
        return group.is_last_group

    def set_forward_prefetch_layer(self, first_layer, second_layer):
        """
        Configure prefetching and offloading order between two consecutive layers.

        Usage:
            for i in range(len(model.layers) - 1):
                set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])

        Ensures idempotency: safe to call multiple times on the same layer pair.
        """
        if first_layer is second_layer:
            warnings.warn(
                "set_forward_prefetch_layer: "
                "Prefetching between identical layers has no effect.",
                UserWarning,
                stacklevel=2,
            )

        def _ensure_group_name(module):
            """Assign a unique swap group name to the module if not already assigned."""
            if not hasattr(module, "_swap_group_name"):
                name = f"swap_group_{self._layer_count}"
                self._layer_count += 1
                module._swap_group_name = name
                module._swap_group_order = {"prev": None, "next": None}
            return module._swap_group_name
        first_name = _ensure_group_name(first_layer)
        second_name = _ensure_group_name(second_layer)

        if first_name not in self._groups:
            self._groups[first_name] = SwapGroup(first_name)
        if second_name not in self._groups:
            self._groups[second_name] = SwapGroup(second_name)

        if first_layer._swap_group_order["next"] is None:
            first_layer._swap_group_order["next"] = second_name
        if second_layer._swap_group_order["prev"] is None:
            second_layer._swap_group_order["prev"] = first_name

        self._groups[first_name].is_last_group = first_layer._swap_group_order["next"] is None
        self._groups[second_name].is_last_group = second_layer._swap_group_order["next"] is None

        def _forward_pre_hook(group_name, module, _):  # pylint: disable=W0613
            if getattr(module, "_swap_state", None) == "pre_backward":
                return
            SwapManager().set_current_group_name(group_name)

        def _forward_hook(group_name, module, args, output):  # pylint: disable=W0613
            """
            Forward post-hook executed immediately after forward computation
            of the current layer finishes.

            Execution timeline (example with 3 layers, forward order: L0 → L1 → L2):

                Time →
                Forward Compute Stream:
                    | Fwd L0 | post(L0) | Fwd L1 | post(L1) | Fwd L2 |

                Copy Stream (offload):
                            | Offload L0 |    -    | Offload L1 |
                                ↑                ↑
                            offload at post(L0)  offload at post(L1)

            Swap rules:
            1. After forward computation of the current layer completes:
            - If a next layer exists, asynchronously offload the activations
                of the current layer (launch_offload).

            Example:
            - At post-forward of L0, offload activations of L0.
            - At post-forward of L1, offload activations of L1.

            2. To limit device memory peak:
            - If a previous layer exists, wait until its offload operation
                has completed (wait_offload).

            Notes:
            - Offload operations are issued on the copy stream to overlap data transfer
            with forward computation of subsequent layers.
            - If the module is already in 'pre_backward' state, this hook is skipped
            to avoid triggering offload during backward phase.
            """
            if getattr(module, "_swap_state", None) == "pre_backward":
                return
            next_name = module._swap_group_order.get('next', None)
            if next_name:
                SwapManager().protect_alias_tensors(group_name, output)
                SwapManager().launch_offload(group_name)
            prev_name = module._swap_group_order.get('prev', None)
            if prev_name:
                SwapManager().wait_offload(prev_name)

        def _backward_pre_hook(group_name, module, grad_input):  # pylint: disable=W0613
            """
            Pre-backward hook executed immediately before backward computation
            of the current layer starts.

            Execution timeline (example with 3 layers, backward order: L2 → L1 → L0):

                Time →
                Backward Compute Stream:
                    | pre(L2) | Grad L2 | pre(L1) | Grad L1 | pre(L0) | Grad L0 |

                Copy Stream (load):
                            | Load  L1 |    -    | Load  L0 |
                                ↑              ↑
                        prefetch at pre(L2)   prefetch at pre(L1)

            Swap rules:
            1. At the beginning of backward for the current layer:
            - If a previous layer exists in backward order, asynchronously
                prefetch its activations (launch_load).

            Example:
            - At pre-backward of L2, prefetch activations of L1.
            - At pre-backward of L1, prefetch activations of L0.

            2. Before starting backward computation of the current layer:
            - Ensure that the activations of the current layer have already
                been loaded back to device memory (wait_load).

            Notes:
            - Load operations are issued on the copy stream to overlap data transfer
            with backward computation of the current layer.
            - The swap state is marked as 'pre_backward' to prevent forward hooks
            from issuing offload operations during backward phase.
            """
            module._swap_state = "pre_backward"
            prev_name = module._swap_group_order.get('prev', None)
            if prev_name:
                SwapManager().launch_load(prev_name)

            next_name = module._swap_group_order.get('next', None)
            if next_name:
                SwapManager().wait_load(group_name)
            SwapManager().release_group_storage(group_name)

        def _backward_hook(group_name, module, grad_input, grad_output):  # pylint: disable=W0613
            module._swap_state = "backward"

        def _register_hooks_once(module, group_name):
            hooks = [
                ("_swap_forward_pre_hook_handle",
                 lambda h: platform.register_forward_pre_hook(module, h, prepend=True),
                 functools.partial(_forward_pre_hook, group_name)),

                ("_swap_forward_hook_handle",
                 module.register_forward_hook,
                 functools.partial(_forward_hook, group_name)),

                ("_swap_backward_pre_hook_handle",
                 lambda h: platform.register_full_backward_pre_hook(module, h, prepend=True),
                 functools.partial(_backward_pre_hook, group_name)),

                ("_swap_backward_hook_handle",
                 lambda h: platform.register_full_backward_hook(module, h),
                 functools.partial(_backward_hook, group_name)),
            ]

            for attr_name, register_func, hook in hooks:
                if not hasattr(module, attr_name):
                    handle = register_func(hook)
                    setattr(module, attr_name, handle)
        # Register for both layers
        _register_hooks_once(first_layer, first_name)
        _register_hooks_once(second_layer, second_name)

    def _get_copy_stream(self):
        """Return a singleton copy stream, created on first access."""
        if self._copy_stream is None:
            self._copy_stream = platform.new_stream()
        return self._copy_stream
