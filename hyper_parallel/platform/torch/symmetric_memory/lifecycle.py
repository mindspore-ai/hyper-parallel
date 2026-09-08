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
"""Minimal process-wide lifecycle for Torch whole-world SHMEM."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

_DEFAULT_ENDPOINT = "tcp://127.0.0.1:8662"
_DEFAULT_HEAP_SIZE = 1024**3


@dataclass
class _ShmemProcessState:
    """Native communicator state shared by every Torch SHMEM owner."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    manager: Any | None = None
    group: Any | None = None
    rank: int | None = None
    world_size: int | None = None
    heap_size: int | None = None
    clients: int = 0

    @property
    def initialized(self) -> bool:
        """Return whether the native communicator is active."""
        return self.manager is not None

    def reset(self) -> None:
        """Clear the finalized communicator state."""
        self.manager = None
        self.group = None
        self.rank = None
        self.world_size = None
        self.heap_size = None
        self.clients = 0


_PROCESS_STATE = _ShmemProcessState()


def _load_manager() -> Any:
    """Return the lazily loaded native manager."""
    binding = import_module(
        "hyper_parallel.platform.torch.symmetric_memory.symmetric_memory"
    )
    getter = getattr(binding, "_get_manager", None)
    if not callable(getter):
        raise TypeError("the Torch SHMEM binding does not expose its manager.")
    return getter()


def _resolve_world(group: Any | None) -> tuple[Any | None, int, int]:
    """Resolve and validate the one whole-world group supported here."""
    import torch.distributed as dist  # pylint: disable=C0415

    if not dist.is_initialized():
        if group is not None:
            raise RuntimeError(
                "a SHMEM process group cannot be selected before "
                "torch.distributed initialization."
            )
        return None, 0, 1

    selected_group = dist.group.WORLD if group is None else group
    rank = dist.get_rank(selected_group)
    world_size = dist.get_world_size(selected_group)
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise ValueError(
            "Torch SHMEM currently supports only a process group covering the "
            "whole distributed world with identical rank ordering."
        )
    return selected_group, rank, world_size


def _heap_size() -> int:
    """Read the configured process-local SHMEM heap size."""
    configured = os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(_DEFAULT_HEAP_SIZE))
    try:
        heap_size = int(configured)
    except ValueError as error:
        raise ValueError(
            "SYMMETRIC_MEMORY_HEAP_SIZE must be a positive integer number "
            f"of bytes, got {configured!r}."
        ) from error
    if heap_size <= 0:
        raise ValueError(
            f"SYMMETRIC_MEMORY_HEAP_SIZE must be positive, got {heap_size}."
        )
    return heap_size


def _barrier(group: Any | None, world_size: int) -> None:
    """Synchronize the active whole-world communicator when available."""
    import torch.distributed as dist  # pylint: disable=C0415

    if dist.is_initialized() and world_size > 1:
        dist.barrier(group=group)


class _TorchSymmetricMemoryOwner:
    """Reference-counted owner of the process-wide Torch SHMEM runtime."""

    def __init__(self, group: Any | None) -> None:
        """Acquire one owner reference for the whole-world group."""
        self._group: Any | None = None
        self._world_size: int | None = None
        self._manager_ref: Any | None = None
        self._closed = False
        self._allocations: dict[int, Any] = {}
        self._acquire(group)

    @property
    def closed(self) -> bool:
        """Return whether this owner has released its runtime reference."""
        return self._closed

    def _acquire(self, requested_group: Any | None) -> None:
        """Initialize or join the process-wide native communicator."""
        group, rank, world_size = _resolve_world(requested_group)
        heap_size = _heap_size()
        with _PROCESS_STATE.lock:
            if _PROCESS_STATE.initialized:
                self._join(rank, world_size, heap_size)
                return

            manager = _load_manager()
            status = manager.attr_init(
                rank,
                world_size,
                heap_size,
                os.getenv("SHMEM_IP_PORT", _DEFAULT_ENDPOINT),
            )
            if status != 0:
                raise RuntimeError(
                    f"aclshmem initialization failed with status {status}."
                )
            _PROCESS_STATE.manager = manager
            _PROCESS_STATE.group = group
            _PROCESS_STATE.rank = rank
            _PROCESS_STATE.world_size = world_size
            _PROCESS_STATE.heap_size = heap_size
            _PROCESS_STATE.clients = 1
            self._group = group
            self._world_size = world_size
            self._manager_ref = manager

    def _join(self, rank: int, world_size: int, heap_size: int) -> None:
        """Join one compatible active communicator."""
        if _PROCESS_STATE.rank != rank or _PROCESS_STATE.world_size != world_size:
            raise RuntimeError(
                "active Torch SHMEM communicator does not match the whole world."
            )
        if _PROCESS_STATE.heap_size is None or _PROCESS_STATE.heap_size < heap_size:
            raise RuntimeError(
                "active Torch SHMEM heap is smaller than the requested size: "
                f"active={_PROCESS_STATE.heap_size}, requested={heap_size}."
            )
        manager = _PROCESS_STATE.manager
        if manager is None:
            raise RuntimeError("active Torch SHMEM manager is missing.")
        _PROCESS_STATE.clients += 1
        self._group = _PROCESS_STATE.group
        self._world_size = world_size
        self._manager_ref = manager

    @staticmethod
    def _normalize_shape(shape: Any) -> list[int]:
        """Normalize one native allocation shape."""
        normalized = [shape] if isinstance(shape, int) else list(shape)
        if not normalized or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in normalized
        ):
            raise ValueError(
                f"SHMEM shape must contain positive integers, got {shape!r}."
            )
        return normalized

    def _manager(self) -> Any:
        """Return this owner's active native manager."""
        if self._closed:
            raise RuntimeError("cannot use a closed Torch SHMEM owner.")
        if self._manager_ref is None:
            raise RuntimeError("Torch SHMEM owner has no native manager.")
        return self._manager_ref

    def empty(self, shape: Any, dtype: Any) -> Any:
        """Allocate a tensor from the symmetric heap."""
        tensor = self._manager().malloc(self._normalize_shape(shape), dtype)
        self._allocations[id(tensor)] = tensor
        return tensor

    def aligned_empty(self, shape: Any, dtype: Any, alignment: int) -> Any:
        """Allocate an aligned tensor from the symmetric heap."""
        if (
            not isinstance(alignment, int)
            or isinstance(alignment, bool)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise ValueError(
                f"alignment must be a positive power of two, got {alignment}."
            )
        tensor = self._manager().aligned_malloc(
            self._normalize_shape(shape),
            dtype,
            alignment,
        )
        self._allocations[id(tensor)] = tensor
        return tensor

    def free(self, tensor: Any) -> None:
        """Free and invalidate one allocation owned by this handle."""
        allocation = self._allocations.get(id(tensor))
        if allocation is None:
            raise ValueError("cannot free a tensor not owned by this SHMEM owner.")
        self._manager().free(allocation)
        self._allocations.pop(id(tensor))
        allocation.untyped_storage().resize_(0)

    def barrier(self) -> None:
        """Synchronize the whole-world communicator."""
        self._manager()
        if self._world_size is None:
            raise RuntimeError("Torch SHMEM owner has no world size.")
        _barrier(self._group, self._world_size)

    def close(self) -> None:
        """Release allocations and finalize after the last owner closes."""
        if self._closed:
            return
        if self._allocations:
            import torch  # pylint: disable=C0415

            torch.npu.synchronize()
            self.barrier()
            for tensor in tuple(self._allocations.values()):
                self.free(tensor)
            self.barrier()
        self._closed = True
        with _PROCESS_STATE.lock:
            _PROCESS_STATE.clients -= 1
            self._manager_ref = None
            if _PROCESS_STATE.clients:
                return
            manager = _PROCESS_STATE.manager
            world_size = _PROCESS_STATE.world_size
            if manager is None or world_size is None:
                raise RuntimeError("Torch SHMEM state disappeared before finalization.")
            barrier_error = None
            try:
                _barrier(_PROCESS_STATE.group, world_size)
            except RuntimeError as error:
                barrier_error = error
            try:
                status = manager.finalize()
            finally:
                _PROCESS_STATE.reset()
            if barrier_error is not None:
                if status != 0:
                    raise RuntimeError(
                        "Torch SHMEM barrier failed and native finalization returned "
                        f"status {status}."
                    ) from barrier_error
                raise barrier_error
            if status != 0:
                raise RuntimeError(
                    f"aclshmem finalization failed with status {status}."
                )


def acquire_symmetric_memory(group: Any | None = None) -> _TorchSymmetricMemoryOwner:
    """Acquire an initialized owner for the whole-world SHMEM runtime."""
    return _TorchSymmetricMemoryOwner(group)


__all__ = ["acquire_symmetric_memory"]
