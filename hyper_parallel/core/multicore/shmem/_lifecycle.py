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
"""Reference-counted Torch lifecycle for the process-wide SHMEM Runtime."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any

from ._runtime import _load_native, _torch_modules

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


_lock = threading.RLock()
_users = 0
_root_group: Any | None = None
_root_uses_distributed: bool | None = None
_root_size: int | None = None
_root_ranks: tuple[int, ...] | None = None
# Set only when Native shutdown fails and leaves the process Runtime unsafe to reinitialize.
_shutdown_failed = False


def _reference_count() -> int:
    """Return the process-local count of acquired SHMEM Runtime references."""
    with _lock:
        return _users


def _host_barrier() -> None:
    """Complete a host-synchronous HCCL barrier over the acquired Root group."""
    if _users <= 0:
        raise RuntimeError("SHMEM Runtime has no active reference to barrier")
    if _root_uses_distributed and _root_size > 1:
        _, dist = _torch_modules()
        dist.barrier(group=_root_group)


def _describe_root(root_group: Any, dist: Any) -> tuple[int, tuple[int, ...]]:
    """Validate group membership and return group-local CANN coordinates."""
    world_size = int(dist.get_world_size())
    root_rank = int(dist.get_rank(root_group))
    if root_rank < 0:
        raise RuntimeError(f"current process must belong to the SHMEM Root Group: root_rank={root_rank}")
    global_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(root_group))
    if (not global_ranks or len(set(global_ranks)) != len(global_ranks)
            or any(rank < 0 or rank >= world_size for rank in global_ranks)):
        raise RuntimeError(
            f"SHMEM Root Group has invalid ordered membership: {global_ranks}, world_size={world_size}"
        )
    if root_rank >= len(global_ranks):
        raise RuntimeError(
            f"current process must belong to the SHMEM Root Group: root_rank={root_rank}, ranks={global_ranks}"
        )
    return root_rank, global_ranks


def _subgroup_unique_id(native: Any, dist: Any, group: Any, rank: int, ranks: tuple[int, ...]) -> bytes:
    """Broadcast an independent CANN bootstrap ID without involving other PP stages."""
    payload = [None, None]
    root_error = None
    if rank == 0:
        try:
            payload[0] = native._get_unique_id()  # pylint: disable=protected-access
        except Exception as error:  # All group members must observe a root bootstrap failure.
            root_error = error
            payload[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(payload, src=ranks[0], group=group)
    if payload[1] is not None:
        raise RuntimeError(f"SHMEM bootstrap failed for ranks={ranks}: {payload[1]}") from root_error
    if not isinstance(payload[0], bytes) or not payload[0]:
        raise RuntimeError(f"SHMEM bootstrap returned an invalid unique ID for ranks={ranks}")
    return payload[0]


def _validate_active_root(selected_group: Any, dist: Any, heap_size_bytes: int | None) -> None:
    """Check that another owner can join the currently initialized root."""
    if heap_size_bytes is not None:
        config = _load_native()._debug_state()["config"]  # pylint: disable=protected-access
        if config["heap_size_bytes"] < heap_size_bytes:
            raise RuntimeError("active SHMEM heap is smaller than the requested managed layout")
    if dist.is_initialized() != _root_uses_distributed:
        raise RuntimeError("torch.distributed state changed during the active SHMEM Runtime lifecycle")
    if dist.is_initialized() and selected_group is not _root_group:
        _, selected_ranks = _describe_root(selected_group, dist)
        if selected_ranks != _root_ranks:
            raise RuntimeError(
                "SHMEM Root Group has different ordered membership from the active Runtime: "
                f"active={_root_ranks}, requested={selected_ranks}"
            )


def acquire(root_group: ProcessGroup | None = None, *, heap_size_bytes: int | None = None) -> None:
    """Acquire one reference to the Root-only process SHMEM Runtime.

    The first reference initializes the Native Runtime. Later references with the same ordered group membership
    share it. Every successful call must be paired with one :func:`release` call after the consumer has stopped using
    SHMEM and freed its Allocations.

    Args:
        heap_size_bytes: Optional explicit initialization size, overriding the environment for this lifecycle.
        root_group: The EP ProcessGroup, which may be a subgroup of Torch WORLD. ``None`` selects WORLD when
            Torch distributed is initialized, or a one-PE Root otherwise. Disjoint groups initialize independently;
            one process cannot participate in two different active SHMEM roots simultaneously.

    Raises:
        RuntimeError: If the Root identity or framework state differs from the active lifecycle, or a previous Native
            shutdown failed.
    """
    global _users, _root_group, _root_uses_distributed, _root_size  # pylint: disable=global-statement
    global _root_ranks  # pylint: disable=global-statement

    with _lock:
        if heap_size_bytes is not None and (isinstance(heap_size_bytes, bool)
                                            or not isinstance(heap_size_bytes, int) or heap_size_bytes <= 0):
            raise ValueError("heap_size_bytes must be a positive integer")
        if _shutdown_failed:
            raise RuntimeError("SHMEM Runtime cannot be acquired after a Native shutdown failure")

        _, dist = _torch_modules()
        uses_distributed = bool(dist.is_initialized())
        if uses_distributed:
            selected_group = dist.group.WORLD if root_group is None else root_group
        else:
            if root_group is not None:
                raise RuntimeError("a SHMEM Root Group cannot be selected before torch.distributed initialization")
            selected_group = None

        if _users > 0:
            _validate_active_root(selected_group, dist, heap_size_bytes)
            _users += 1
            return

        if uses_distributed:
            root_rank, root_ranks = _describe_root(selected_group, dist)
            root_size = len(root_ranks)
            dist.barrier(group=selected_group)
        else:
            root_rank, root_size = 0, 1
            root_ranks = None

        # Native initialization failure is retryable and must not create a consumer reference.
        native = _load_native()
        if heap_size_bytes is not None:
            unique_id = (_subgroup_unique_id(native, dist, selected_group, root_rank, root_ranks)
                         if uses_distributed else b"")
            native._initialize(root_rank, root_size, unique_id, heap_size_bytes)  # pylint: disable=protected-access
        elif uses_distributed and root_ranks != tuple(range(dist.get_world_size())):
            unique_id = _subgroup_unique_id(native, dist, selected_group, root_rank, root_ranks)
            native._initialize(root_rank, root_size, unique_id)  # pylint: disable=protected-access
        else:
            native._initialize(root_rank, root_size)  # pylint: disable=protected-access
        _root_group = selected_group
        _root_uses_distributed = uses_distributed
        _root_size = root_size
        _root_ranks = root_ranks
        _users = 1


def _runtime_access(function):
    """Exclude SHMEM API submissions while the managed heap is being rebuilt."""
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        """Serialize one API submission against managed heap teardown."""
        with _lock:
            if _shutdown_failed:
                raise RuntimeError("SHMEM Runtime is unsafe after a shutdown or reconfiguration failure")
            return function(*args, **kwargs)
    return guarded


@contextmanager
def _reconfiguration():
    """Serialize managed teardown with allocation and lifecycle API calls."""
    with _lock:
        if _shutdown_failed:
            raise RuntimeError("SHMEM Runtime is unsafe after a shutdown or reconfiguration failure")
        yield


def _reconfiguration_stage(operation: Any) -> float:
    """Converge stage errors before any rank enters the next vendor collective."""
    started = time.perf_counter()
    error = None
    try:
        operation()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors = [error]
    if _root_uses_distributed:
        _, dist = _torch_modules()
        errors = [None] * _root_size
        dist.all_gather_object(errors, error, group=_root_group)
    if any(item is not None for item in errors):
        raise RuntimeError(f"SHMEM reconfiguration stage failed: {errors}")
    return (time.perf_counter() - started) * 1000


def _invalidate_runtime() -> None:
    """Prevent all APIs from using a partially destroyed or rebuilt runtime."""
    global _shutdown_failed  # pylint: disable=global-statement
    with _lock:
        _shutdown_failed = True


def _reinitialize(heap_size_bytes: int) -> dict[str, float]:
    """Replace an empty, quiescent heap while preserving its logical owner references.

    The managed coordinator holds ``_reconfiguration`` and has collectively
    validated all owners, freed every allocation and synchronized the device.
    """
    with _lock:
        if _users <= 0 or _shutdown_failed:
            raise RuntimeError("SHMEM reconfiguration requires a healthy active Runtime")
        if isinstance(heap_size_bytes, bool) or not isinstance(heap_size_bytes, int) or heap_size_bytes <= 0:
            raise ValueError("heap_size_bytes must be a positive integer")
        native = _load_native()
        _, dist = _torch_modules()
        rank = int(dist.get_rank(_root_group)) if _root_uses_distributed else 0
        timings = {}
        try:
            _reconfiguration_stage(native._validate_shutdown)  # pylint: disable=protected-access
            timings["finalize_ms"] = _reconfiguration_stage(native._shutdown)  # pylint: disable=protected-access
            started = time.perf_counter()
            # The vendor bootstrap singleton must be finalized before requesting its next ID.
            unique_id = (_subgroup_unique_id(native, dist, _root_group, rank, _root_ranks)
                         if _root_uses_distributed else b"")
            timings["bootstrap_ms"] = (time.perf_counter() - started) * 1000
            timings["initialize_ms"] = _reconfiguration_stage(
                lambda: native._initialize(rank, _root_size, unique_id, heap_size_bytes))  # pylint: disable=protected-access
        except Exception:
            _invalidate_runtime()
            raise
        return timings


def release() -> None:
    """Release one SHMEM Runtime reference and shut down the last reference.

    The last consumer must free every symmetric Allocation and stop all SHMEM work before calling this function. All
    Root ranks must release their final reference in the same collective order. Failures before Native shutdown retain
    the final reference so the caller can correct the condition and retry.

    Raises:
        RuntimeError: If no reference is active, framework state changed, shutdown preconditions fail, or Native
            shutdown fails.
    """
    global _users, _shutdown_failed, _root_group, _root_uses_distributed, _root_size  # pylint: disable=global-statement
    global _root_ranks  # pylint: disable=global-statement

    with _lock:
        if _users <= 0:
            raise RuntimeError("SHMEM Runtime has no active reference to release")
        if _shutdown_failed:
            raise RuntimeError("SHMEM Runtime is unsafe after a shutdown or reconfiguration failure")
        if _users > 1:
            _users -= 1
            return

        torch, dist = _torch_modules()
        distributed_is_initialized = bool(dist.is_initialized())
        if distributed_is_initialized != _root_uses_distributed:
            raise RuntimeError("torch.distributed state changed during the active SHMEM Runtime lifecycle")

        native = _load_native()
        native._validate_shutdown()  # pylint: disable=protected-access
        torch.npu.synchronize()
        if _root_uses_distributed:
            dist.barrier(group=_root_group)

        shutdown_succeeded = False
        try:
            native._shutdown()  # pylint: disable=protected-access
            shutdown_succeeded = True
        finally:
            _users = 0
            _root_group = None
            _root_uses_distributed = None
            _root_size = None
            _root_ranks = None
            _shutdown_failed = not shutdown_succeeded


__all__ = [
    "acquire",
    "release",
]
