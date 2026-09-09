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
from typing import TYPE_CHECKING, Any

from ._runtime import _load_native, _torch_modules

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


_lock = threading.Lock()
_users = 0
_root_group: Any | None = None
_root_uses_distributed: bool | None = None
# Set only when Native shutdown fails and leaves the process Runtime unsafe to reinitialize.
_shutdown_failed = False


def _reference_count() -> int:
    """Return the process-local count of acquired SHMEM Runtime references."""
    with _lock:
        return _users


def _describe_root(root_group: Any, dist: Any) -> tuple[int, int]:
    """Validate one complete Torch WORLD and return its CANN Root rank and size."""
    world_size = int(dist.get_world_size())
    expected_ranks = tuple(range(world_size))
    global_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(root_group))
    if global_ranks != expected_ranks:
        raise RuntimeError(
            "SHMEM Root Group must cover the complete Torch WORLD in global-rank order: "
            f"expected={expected_ranks}, actual={global_ranks}"
        )
    root_rank = int(dist.get_rank(root_group))
    if root_rank < 0 or root_rank >= world_size:
        raise RuntimeError(
            f"current process must belong to the SHMEM Root Group: root_rank={root_rank}, root_size={world_size}"
        )
    return root_rank, world_size


def acquire(root_group: ProcessGroup | None = None) -> None:
    """Acquire one reference to the Root-only process SHMEM Runtime.

    The first reference initializes the Native Runtime. Later references with the same ordered complete Torch WORLD
    share it. Every successful call must be paired with one :func:`release` call after the consumer has stopped using
    SHMEM and freed its Allocations.

    Args:
        root_group: A ProcessGroup covering the complete Torch WORLD in global-rank order. ``None`` selects WORLD
            when Torch distributed is initialized, or a one-PE Root otherwise.

    Raises:
        RuntimeError: If the Root identity or framework state differs from the active lifecycle, or a previous Native
            shutdown failed.
    """
    global _users, _root_group, _root_uses_distributed  # pylint: disable=global-statement

    with _lock:
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
            if uses_distributed != _root_uses_distributed:
                raise RuntimeError("torch.distributed state changed during the active SHMEM Runtime lifecycle")
            if uses_distributed and selected_group is not _root_group:
                _describe_root(selected_group, dist)
            _users += 1
            return

        if uses_distributed:
            root_rank, root_size = _describe_root(selected_group, dist)
            dist.barrier(group=selected_group)
        else:
            root_rank, root_size = 0, 1

        # Native initialization failure is retryable and must not create a consumer reference.
        _load_native()._initialize(root_rank, root_size)  # pylint: disable=protected-access
        _root_group = selected_group
        _root_uses_distributed = uses_distributed
        _users = 1


def release() -> None:
    """Release one SHMEM Runtime reference and shut down the last reference.

    The last consumer must free every symmetric Allocation and stop all SHMEM work before calling this function. All
    Root ranks must release their final reference in the same collective order. Failures before Native shutdown retain
    the final reference so the caller can correct the condition and retry.

    Raises:
        RuntimeError: If no reference is active, framework state changed, shutdown preconditions fail, or Native
            shutdown fails.
    """
    global _users, _shutdown_failed, _root_group, _root_uses_distributed  # pylint: disable=global-statement

    with _lock:
        if _users <= 0:
            raise RuntimeError("SHMEM Runtime has no active reference to release")
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
            _shutdown_failed = not shutdown_succeeded


__all__ = [
    "acquire",
    "release",
]
