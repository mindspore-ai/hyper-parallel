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
"""Internal Torch allocation and one-sided SHMEM APIs."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._debug import _log_allocation_site
from ._runtime import _load_native, _torch_modules

if TYPE_CHECKING:
    import torch


def _normalize_size(size: tuple[Any, ...]) -> tuple[int, ...]:
    """Normalize Torch-style variadic dimensions or one dimension sequence."""
    shape: Sequence[Any]
    if len(size) == 1 and isinstance(size[0], Sequence) and not isinstance(size[0], (str, bytes)):
        shape = size[0]
    else:
        shape = size

    normalized = []
    for dimension_index, dimension in enumerate(shape):
        if isinstance(dimension, bool):
            raise TypeError(f"size[{dimension_index}] must be an integer, got bool")
        try:
            normalized.append(operator.index(dimension))
        except TypeError as error:
            raise TypeError(
                f"size[{dimension_index}] must be an integer, got {type(dimension).__name__}"
            ) from error
    return tuple(normalized)


def _normalize_alignment(alignment: Any | None) -> int | None:
    """Normalize an optional explicit base-address alignment."""
    if alignment is None:
        return None
    if isinstance(alignment, bool):
        raise TypeError("alignment must be an integer, got bool")
    try:
        return operator.index(alignment)
    except TypeError as error:
        raise TypeError(f"alignment must be an integer, got {type(alignment).__name__}") from error


def empty(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    alignment: int | None = None,
) -> torch.Tensor:
    """Collectively allocate one contiguous Tensor from the Root WORLD symmetric Heap.

    Args:
        size: Torch-style variadic dimensions or one dimension sequence.
        dtype: Tensor dtype; ``None`` uses Torch's default dtype.
        alignment: Optional Allocation-base alignment in Byte.

    Returns:
        A symmetric Tensor that must later be passed to :func:`free` collectively.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the Allocation.

    Note:
        Call ``shmem.acquire()`` before allocating and keep that reference until this Allocation is freed.
        The final ``shmem.release()`` must not run concurrently with this call.
        With ``HYPER_PARALLEL_SHMEM_LOG_LEVEL=0``, one best-effort DEBUG stderr line records the direct caller so an
        Allocation leaked without ``free`` can be traced back to its code site via ``allocation_base``.
    """
    torch, _ = _torch_modules()
    shape = _normalize_size(size)
    effective_dtype = torch.get_default_dtype() if dtype is None else dtype
    tensor = _load_native()._empty(  # pylint: disable=protected-access
        shape,
        effective_dtype,
        _normalize_alignment(alignment),
    )
    _log_allocation_site(tensor)
    return tensor


def free(tensor: torch.Tensor) -> None:
    """Collectively release one complete symmetric Allocation after Stream quiescence.

    Args:
        tensor: The complete Tensor returned by :func:`empty`; Tensor views cannot be released independently.

    Raises:
        RuntimeError: If no SHMEM reference is active or ``tensor`` cannot be released.

    Warning:
        Post-free use is undefined behavior. ``free`` only returns the block to the symmetric heap: the mapping,
        the data, and the Tensor's shape, Storage, and data pointer all stay intact, so a stale Tensor still looks
        usable. Passing it to any SHMEM operation (``put``/``get``/``signal``/``wait_signal``/``all_gather`` or a
        repeated ``free``) is rejected deterministically, but plain local Torch use (e.g. ``x += 1``) is not guarded
        and silently writes into the freed block. The heap allocator reuses freed blocks best-fit, so the next
        same-size :func:`empty` receives the same address with certainty: any stale reference then aliases and
        corrupts the new Allocation, and the corruption reaches remote PEs through subsequent RMA. Drop every
        reference to the Tensor at ``free`` time.

    Note:
        Established usage model: allocate symmetric Tensors once (typically at first use) and free them once at
        teardown, collectively on every rank in a consistent order; high-frequency alloc/free cycling is not
        supported. A SHMEM reference must remain active through this call, and the final ``shmem.release()`` must not
        run concurrently with it.
    """
    _load_native()._free(tensor)  # pylint: disable=protected-access


def barrier(*, blocking: bool = True) -> None:
    """Enqueue a Root WORLD barrier on the current NPU Stream.

    Args:
        blocking: If ``True`` (the default), synchronize the current NPU Stream after enqueueing the barrier. The call
            then returns only after the barrier and all earlier work on that Stream have completed. Pass ``False``
            only when subsequent work is deliberately ordered after the barrier on the same Stream.

    Runtime rejects a Stream from a device other than the one frozen at initialization. With ``blocking=False``, Host
    return only confirms that the CANN Host API directly enqueued the barrier on the current ACL Stream; the barrier
    is not submitted through the torch_npu task queue and device completion is not Host-visible. Subsequent work is
    safely ordered after it only when that work is submitted to the same Stream.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the barrier.

    Note:
        Call ``shmem.acquire()`` first. The final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._barrier(blocking=blocking)  # pylint: disable=protected-access


def put(remote_dst: torch.Tensor, local_src: torch.Tensor, target_pe: int) -> None:
    """Enqueue a byte-for-byte Put to one Root WORLD PE.

    Args:
        remote_dst: Contiguous symmetric destination Tensor or view.
        local_src: Contiguous local NPU source Tensor with the same byte count.
        target_pe: Destination in CANN Root WORLD coordinates.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the operation.

    Note:
        Return only confirms enqueue on the current NPU Stream. Call ``shmem.acquire()`` first;
        the final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._put(remote_dst, local_src, target_pe)  # pylint: disable=protected-access


def get(local_dst: torch.Tensor, remote_src: torch.Tensor, source_pe: int) -> None:
    """Enqueue a byte-for-byte Get from one Root WORLD PE.

    Args:
        local_dst: Contiguous local NPU destination Tensor.
        remote_src: Contiguous symmetric source Tensor or view with the same byte count.
        source_pe: Source in CANN Root WORLD coordinates.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the operation.

    Note:
        Return only confirms enqueue on the current NPU Stream. Call ``shmem.acquire()`` first;
        the final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._get(local_dst, remote_src, source_pe)  # pylint: disable=protected-access


def signal(remote_signal: torch.Tensor, value: int, target_pe: int, *, operation: str = "set") -> None:
    """Enqueue a Set or Add update to one symmetric Signal.

    Args:
        remote_signal: Contiguous symmetric Tensor view containing one ``int32`` element.
        value: Signed 32-bit update value.
        target_pe: Destination in CANN Root WORLD coordinates.
        operation: Exactly ``"set"`` or ``"add"``.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the operation.

    Note:
        Concurrent Signal slots must be spaced by at least 64 Byte by the consumer. Call ``shmem.acquire()`` first;
        the final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._signal(  # pylint: disable=protected-access
        remote_signal,
        value,
        target_pe,
        operation=operation,
    )


def wait_signal(
    signal_tensor: torch.Tensor,
    value: int,
    *,
    comparison: str = "eq",
) -> None:
    """Enqueue a wait for one local symmetric Signal.

    Args:
        signal_tensor: One local symmetric ``int32`` Signal element.
        value: Signed 32-bit comparison value.
        comparison: One of ``"eq"``, ``"ne"``, ``"gt"``, ``"ge"``, ``"lt"`` or ``"le"``.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the operation.

    Note:
        Call ``shmem.acquire()`` first. The final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._wait_signal(  # pylint: disable=protected-access
        signal_tensor,
        value,
        comparison=comparison,
    )


def all_gather(output: torch.Tensor, input: torch.Tensor) -> None:
    """Enqueue a Root WORLD AllGather using symmetric ``output`` storage.

    Args:
        output: Contiguous symmetric NPU Tensor with ``input.numel() * root_size`` elements.
        input: Contiguous local NPU Tensor contributed by the calling Root PE.

    Raises:
        RuntimeError: If no SHMEM reference is active or the Runtime rejects the operation.

    Note:
        Output slots follow CANN Root WORLD order. Return means the Kernel and exit barrier were enqueued on the
        current NPU Stream; Host-visible completion requires Stream synchronization. A zero-byte ``input`` is a
        no-op after validation: no Kernel and no barrier are enqueued. Call ``shmem.acquire()`` first;
        the final ``shmem.release()`` must not run concurrently with this call.
    """
    _load_native()._all_gather(output, input)  # pylint: disable=protected-access


__all__ = [
    "all_gather",
    "barrier",
    "empty",
    "free",
    "get",
    "put",
    "signal",
    "wait_signal",
]
