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

__all__ = [
    "acquire",
    "release",
]

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, NoReturn, Optional

from .. import _collective
from ._runtime import _load_native, _torch_modules

if TYPE_CHECKING:
    import torch
    from torch.distributed import ProcessGroup


_lock = threading.RLock()
_users = 0
_root_group: Any | None = None
_root_uses_distributed: bool | None = None
_root_size: int | None = None
_root_ranks: tuple[int, ...] | None = None
_device_index: int | None = None
# Blocks SHMEM access after physical, device, or Root convergence failures.
_runtime_invalidated = False
_NATIVE_ERROR_PREFIX = "[HYPER_PARALLEL_SHMEM_ERROR]"
_NATIVE_MESSAGE_FIELD = " message="
_CANN_DETAIL_FIELD = ", cann_detail=["
_RETRYABLE_RUNTIME_HINT = "The active SHMEM Runtime remains usable."
_TERMINAL_RUNTIME_HINT = "The SHMEM Runtime is no longer usable; restart the process."
_CONVERGENCE_FAILURE_HINT = (
    "The SHMEM Runtime can no longer be managed safely after Root convergence failed; restart the process."
)
_INVALIDATED_RUNTIME_MESSAGE = (
    "SHMEM Runtime is no longer usable after a shutdown or restart failure; restart the process"
)


@dataclass(frozen=True)
class _AllocationSpec:
    """Physical properties of one symmetric Allocation after restart."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    alignment_bytes: int


@dataclass(frozen=True)
class _AllocationReplacement:
    """Replace one complete active Allocation with a target specification."""

    current: torch.Tensor
    target: _AllocationSpec


@dataclass(frozen=True)
class _RestartResult:
    """Symmetric Tensors and stage timings produced by a successful restart."""

    tensors: tuple[torch.Tensor, ...]
    stages_ms: dict[str, float]


class _RuntimeUnavailableError(RuntimeError):
    """Mark a stage failure after which the active Runtime cannot be used safely."""


def _exception_summary(error: Exception) -> str:
    """Extract display text without changing the exception or its cause."""
    message = str(error)
    if message.startswith(_NATIVE_ERROR_PREFIX):
        _, separator, native_message = message.partition(_NATIVE_MESSAGE_FIELD)
        if separator:
            message = native_message
            message, _, _ = message.partition(_CANN_DETAIL_FIELD)
    return message


def _raise_lifecycle_failure(message: str, error: Exception, *, hint: str) -> NoReturn:
    """Raise a lifecycle error with recovery guidance and the available local cause."""
    message = f"{message.rstrip('.')}. {hint}"
    if error.__cause__ is not None:
        raise RuntimeError(message) from error.__cause__
    if error.__suppress_context__:
        raise RuntimeError(message) from None
    raise RuntimeError(message) from error


def _reference_count() -> int:
    """Return the process-local count of acquired SHMEM Runtime references."""
    with _lock:
        return _users


def _runtime_is_healthy() -> bool:
    """Return whether the process Runtime accepts SHMEM calls."""
    with _lock:
        return _users > 0 and not _runtime_invalidated


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


def root_members(group: ProcessGroup | None) -> tuple[int, ...]:
    """Return ordered SHMEM membership without relying on ProcessGroup identity.

    Args:
        group: Root process group, or None for the default root.
    """
    _, dist = _torch_modules()
    return _describe_root(dist.group.WORLD if group is None else group, dist)[1] if dist.is_initialized() else (0,)


def _subgroup_unique_id(
    native: Any,
    dist: Any,
    group: Any,
    rank: int,
    ranks: tuple[int, ...],
    *,
    failure_prefix: str = "SHMEM bootstrap failed",
) -> bytes:
    """Broadcast an independent CANN bootstrap ID without involving other PP stages."""
    payload = [None, None]
    root_error = None
    if rank == 0:
        # Relay every vendor bootstrap failure so peers cannot wait indefinitely for a root ID.
        try:
            payload[0] = native._get_unique_id()  # pylint: disable=protected-access
        except Exception as error:  # pylint: disable=broad-exception-caught
            root_error = error
            payload[1] = _collective.failure_report(error, message=_exception_summary(error))
    try:
        dist.broadcast_object_list(payload, src=ranks[0], group=group)
    except Exception as error:
        raise RuntimeError(
            f"{failure_prefix} on root_rank={rank}: broadcast failed: {_exception_summary(error)}"
        ) from error
    if payload[1] is not None:
        details = _collective.format_collective_failures(
            [payload[1]],
            rank_label="root_rank",
            ranks_label="root_ranks",
        )
        raise RuntimeError(f"{failure_prefix} on {details}") from root_error
    if not isinstance(payload[0], bytes) or not payload[0]:
        raise RuntimeError(f"{failure_prefix} on root_rank=0: returned an invalid unique ID")
    return payload[0]


def _validate_active_root(selected_group: Any, dist: Any, heap_size_bytes: int | None) -> None:
    """Validate another owner against the initialized root."""
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
        RuntimeError: If the Root identity or framework state differs from the active lifecycle, or the Runtime was
            invalidated by a shutdown or restart failure.
    """
    global _users, _root_group, _root_uses_distributed, _root_size  # pylint: disable=global-statement
    global _root_ranks, _device_index  # pylint: disable=global-statement

    with _lock:
        if heap_size_bytes is not None and (isinstance(heap_size_bytes, bool)
                                            or not isinstance(heap_size_bytes, int) or heap_size_bytes <= 0):
            raise ValueError("heap_size_bytes must be a positive integer")
        if _runtime_invalidated:
            raise RuntimeError(_INVALIDATED_RUNTIME_MESSAGE)

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
        native_state = native._debug_state()  # pylint: disable=protected-access
        _device_index = int(native_state["device_index"])
        _users = 1


def _validate_shutdown_preconditions() -> None:
    """Validate local shutdown state without modifying the active Runtime."""
    _, dist = _torch_modules()
    if bool(dist.is_initialized()) != _root_uses_distributed:
        raise _RuntimeUnavailableError(
            "torch.distributed state changed during the active SHMEM Runtime lifecycle"
        )
    _load_native()._validate_shutdown()  # pylint: disable=protected-access


def _synchronize_device(device: Any = None) -> None:
    """Synchronize NPU work and classify a device failure as Runtime-unavailable."""
    torch, _ = _torch_modules()
    try:
        if device is None:
            torch.npu.synchronize()
        else:
            torch.npu.synchronize(device)
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise _RuntimeUnavailableError(_exception_summary(error)) from error


def _runtime_access(function):
    """Exclude SHMEM API submissions during a managed Runtime restart."""
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        """Serialize one API submission against managed heap teardown."""
        with _lock:
            if _runtime_invalidated:
                raise RuntimeError(_INVALIDATED_RUNTIME_MESSAGE)
            return function(*args, **kwargs)
    return guarded


_StagePayload = tuple[Any, Optional[_collective.FailureReport], bool]


def _execute_local_stage(operation: Callable[[], Any]) -> tuple[_StagePayload, Exception | None]:
    """Execute one local stage and preserve its non-serializable exception cause."""
    try:
        return (operation(), None, False), None
    except Exception as exception:  # pylint: disable=broad-exception-caught
        runtime_unavailable = isinstance(exception, _RuntimeUnavailableError)
        diagnostic = exception.__cause__ if runtime_unavailable and exception.__cause__ is not None else exception
        local_error = exception.__cause__ if runtime_unavailable else exception
        if runtime_unavailable and diagnostic is exception:
            report = "RuntimeError", _exception_summary(diagnostic)
        else:
            report = _collective.failure_report(diagnostic, message=_exception_summary(diagnostic))
        return (None, report, runtime_unavailable), local_error


def _evaluate_stage_results(peers: list[_StagePayload], local_value: Any, local_error: Exception | None,
                            action: str, stage: str) -> None:
    """Raise a local diagnostic for peer failures or inconsistent stage values."""
    if not any(item[1] is not None for item in peers):
        if any(item[0] != local_value for item in peers):
            raise RuntimeError(f"{action} failed during {stage}: values differ across Root ranks")
        return
    runtime_unavailable = any(item[2] for item in peers)
    details = _collective.format_collective_failures(
        [item[1] for item in peers],
        rank_label="root_rank",
        ranks_label="root_ranks",
    )
    error_type = _RuntimeUnavailableError if runtime_unavailable else RuntimeError
    raise error_type(f"{action} failed during {stage} on {details}") from local_error


def _reconfiguration_stage(
    operation: Callable[[], Any],
    *,
    action: str,
    stage: str,
    root_group: Any,
) -> float:
    """Execute a stage and agree on its returned manifest; non-manifest stages return None."""
    started = time.perf_counter()
    payload, local_error = _execute_local_stage(operation)
    peers = _collective.exchange(
        payload,
        group=root_group,
        failure_prefix=f"{action} failed during {stage} while converging Root rank results",
    )
    _evaluate_stage_results(peers, payload[0], local_error, action, stage)
    return (time.perf_counter() - started) * 1000


def _validate_allocation_spec(specification: _AllocationSpec) -> tuple[tuple[int, ...], str, int, int]:
    """Validate one target Allocation and return its rank-independent manifest."""
    if not isinstance(specification.shape, tuple) or not specification.shape:
        raise ValueError("Allocation shape must be a non-empty tuple")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
           for dimension in specification.shape):
        raise ValueError(f"Allocation shape must contain positive integers, got {specification.shape}")
    alignment = specification.alignment_bytes
    if (isinstance(alignment, bool) or not isinstance(alignment, int)
            or alignment <= 0 or alignment & (alignment - 1)):
        raise ValueError(f"Allocation alignment_bytes must be a positive power of two, got {alignment!r}")
    try:
        payload_bytes = specification.dtype.itemsize
        for dimension in specification.shape:
            payload_bytes *= dimension
    except (AttributeError, TypeError) as error:
        raise TypeError(f"Allocation dtype must be a torch dtype, got {specification.dtype!r}") from error
    return specification.shape, str(specification.dtype), alignment, payload_bytes


def _validate_replacements(
    replacements: tuple[_AllocationReplacement, ...],
    native_state: dict[str, Any],
    heap_size_bytes: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Validate complete active Allocations and return current and target manifests."""
    if not isinstance(replacements, tuple):
        raise TypeError("replacements must be a tuple")
    active_records = {
        int(record["allocation_base"]): int(record["allocation_bytes"])
        for record in native_state["active_allocations"]
    }
    current_manifest = []
    target_manifest = []
    current_bases = set()
    target_footprint = 0
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, _AllocationReplacement):
            raise TypeError(f"replacements[{index}] must be an _AllocationReplacement")
        current = replacement.current
        if current is None:
            raise ValueError(f"replacements[{index}].current must be an active Tensor")
        if getattr(current, "_base", None) is not None:
            raise ValueError(f"replacements[{index}].current must be a complete Allocation, not a Tensor view")
        try:
            allocation_base = int(current.data_ptr())
            allocation_bytes = int(current.numel()) * int(current.element_size())
            current_shape = tuple(int(dimension) for dimension in current.shape)
            current_dtype = str(current.dtype)
            contiguous = bool(current.is_contiguous())
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError(f"replacements[{index}].current must be a Tensor") from error
        if not contiguous or active_records.get(allocation_base) != allocation_bytes:
            raise ValueError(f"replacements[{index}].current must be one complete active Allocation")
        if allocation_base in current_bases:
            raise ValueError("replacements must not contain duplicate current Allocations")
        current_bases.add(allocation_base)
        current_manifest.append((current_shape, current_dtype, allocation_bytes))
        target_shape, target_dtype, alignment, payload_bytes = _validate_allocation_spec(replacement.target)
        target_manifest.append((target_shape, target_dtype, alignment, payload_bytes))
        target_footprint += payload_bytes + alignment - 1
    if current_bases != set(active_records):
        raise RuntimeError("replacements must cover every active SHMEM Allocation exactly once")
    if target_footprint > heap_size_bytes:
        raise RuntimeError(
            "replacement Allocations exceed the target SHMEM heap: "
            f"required={target_footprint}, heap_size_bytes={heap_size_bytes}"
        )
    return tuple(current_manifest), tuple(target_manifest)


def _restart(
    *,
    heap_size_bytes: int,
    replacements: tuple[_AllocationReplacement, ...],
    preflight_hook: Callable[[], object],
) -> _RestartResult:
    """Collectively rebuild the sole user's symmetric Allocations.

    Args:
        heap_size_bytes: Positive byte capacity of the restarted symmetric Heap.
        replacements: Complete active Allocations and their target physical specifications.
        preflight_hook: Read-only business-state snapshot executed once while the lifecycle lock is held.

    Returns:
        Replacement Tensors in request order and generic restart stage timings.

    Raises:
        RuntimeError: If the Runtime is unhealthy, another user exists, rank state differs, or restart fails.

    Note:
        Ordinary preflight validation failures leave a healthy Runtime usable only when Root results converge.
        Existing Runtime invalidation, state drift, device synchronization errors and convergence failures require
        a process restart even before Allocation release. Any failure after release begins invalidates the Runtime.
        All Root ranks must call this function in the same order.
    """
    global _runtime_invalidated  # pylint: disable=global-statement
    with _lock:
        native = _load_native()
        torch, dist = _torch_modules()
        destructive_started = False
        current_stage = "Preflight"

        def run_stage(stage: str, operation: Callable[[], Any]) -> float:
            """Run one converged restart stage under a stable diagnostic label."""
            nonlocal current_stage
            current_stage = stage
            return _reconfiguration_stage(
                operation,
                action="SHMEM restart",
                stage=stage,
                root_group=_root_group,
            )

        def preflight() -> tuple[Any, ...]:
            """Capture and validate the complete local pre-destructive state."""
            if _runtime_invalidated:
                raise _RuntimeUnavailableError(_INVALIDATED_RUNTIME_MESSAGE)
            if _users != 1:
                raise RuntimeError("SHMEM restart requires one healthy Runtime user")
            if (isinstance(heap_size_bytes, bool)
                    or not isinstance(heap_size_bytes, int) or heap_size_bytes <= 0):
                raise ValueError("heap_size_bytes must be a positive integer")
            if bool(dist.is_initialized()) != _root_uses_distributed:
                raise _RuntimeUnavailableError(
                    "torch.distributed state changed during the active SHMEM Runtime lifecycle"
                )
            try:
                native_state = native._debug_state()  # pylint: disable=protected-access
            except Exception as error:  # pylint: disable=broad-exception-caught
                raise _RuntimeUnavailableError(_exception_summary(error)) from error
            if native_state.get("state") != "Ready":
                raise _RuntimeUnavailableError(
                    f"SHMEM Runtime state changed during the active lifecycle: state={native_state.get('state')!r}"
                )
            if _device_index is None or native_state.get("device_index") != _device_index:
                raise _RuntimeUnavailableError("SHMEM Runtime device changed during the active lifecycle")
            if int(torch.npu.current_device()) != _device_index:
                raise RuntimeError("current NPU device differs from the SHMEM Runtime device")
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("SHMEM restart is unavailable during graph capture")
            business_state = preflight_hook()
            current_manifest, target_manifest = _validate_replacements(
                replacements,
                native_state,
                heap_size_bytes,
            )
            return heap_size_bytes, business_state, current_manifest, target_manifest

        try:
            stages_ms = {
                "preflight_ms": run_stage("Preflight", preflight),
            }
            device = torch.device("npu", _device_index)
            stages_ms["quiesce_ms"] = run_stage("Quiesce", lambda: _synchronize_device(device))

            # _api imports lifecycle guards, so the local import avoids a cycle while reusing Allocation validation
            # and diagnostics.
            from . import _api  # pylint: disable=import-outside-toplevel

            replacement_tensors = []

            def release_allocations() -> None:
                """Release every declared Allocation and verify an empty registry."""
                for replacement in replacements:
                    _api.free(replacement.current)
                if native._debug_state()["active_allocations"]:  # pylint: disable=protected-access
                    raise RuntimeError("SHMEM restart release left active Allocations")

            destructive_started = True
            stages_ms["release_ms"] = run_stage(
                "ReleaseAllocations",
                release_allocations,
            )
            validate_ms = run_stage(
                "ValidateShutdown",
                native._validate_shutdown,  # pylint: disable=protected-access
            )
            shutdown_ms = run_stage(
                "Shutdown",
                native._shutdown,  # pylint: disable=protected-access
            )
            stages_ms["finalize_ms"] = validate_ms + shutdown_ms
            current_stage = "Bootstrap"
            started = time.perf_counter()
            rank = int(dist.get_rank(_root_group)) if _root_uses_distributed else 0
            unique_id = (_subgroup_unique_id(
                native,
                dist,
                _root_group,
                rank,
                _root_ranks,
                failure_prefix="SHMEM restart failed during Bootstrap",
            )
                         if _root_uses_distributed else b"")
            stages_ms["bootstrap_ms"] = (time.perf_counter() - started) * 1000

            def initialize() -> None:
                """Initialize the target Heap and verify the frozen device."""
                native._initialize(  # pylint: disable=protected-access
                    rank,
                    _root_size,
                    unique_id,
                    heap_size_bytes,
                )
                native_state = native._debug_state()  # pylint: disable=protected-access
                if int(native_state["device_index"]) != _device_index:
                    raise RuntimeError("SHMEM Runtime restarted on a different NPU device")

            stages_ms["initialize_ms"] = run_stage(
                "Initialize",
                initialize,
            )

            def restore_allocations() -> None:
                """Create replacement Tensors in their declared order."""
                replacement_tensors.extend(
                    _api.empty(
                        replacement.target.shape,
                        dtype=replacement.target.dtype,
                        alignment=replacement.target.alignment_bytes,
                    )
                    for replacement in replacements
                )

            stages_ms["restore_ms"] = run_stage(
                "RestoreAllocations",
                restore_allocations,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            convergence_failed = isinstance(error, _collective.CollectiveConvergenceError)
            runtime_unavailable = isinstance(error, _RuntimeUnavailableError)
            if destructive_started or convergence_failed or runtime_unavailable:
                _runtime_invalidated = True
            prefix = f"SHMEM restart failed during {current_stage}"
            message = str(error)
            if not message.startswith(prefix):
                message = f"{prefix}: {_exception_summary(error)}"
            if convergence_failed:
                hint = _CONVERGENCE_FAILURE_HINT
            elif _runtime_invalidated:
                hint = _TERMINAL_RUNTIME_HINT
            else:
                hint = _RETRYABLE_RUNTIME_HINT
            _raise_lifecycle_failure(message, error, hint=hint)
        return _RestartResult(tuple(replacement_tensors), stages_ms)


def release() -> None:
    """Release one SHMEM Runtime reference and shut down the last reference.

    The last consumer must free every symmetric Allocation and stop all SHMEM work before calling this function. All
    Root ranks must release their final reference in the same collective order. Retryable validation failures retain
    the final reference so the caller can correct the condition. Root state changes, NPU synchronization failures and
    failures to converge Root results invalidate lifecycle management and require a process restart.

    Raises:
        RuntimeError: If no reference is active, framework state changed, shutdown preconditions fail, or Native
            shutdown fails.
    """
    global _users, _runtime_invalidated, _root_group, _root_uses_distributed  # pylint: disable=global-statement
    global _root_size  # pylint: disable=global-statement
    global _root_ranks, _device_index  # pylint: disable=global-statement

    with _lock:
        if _runtime_invalidated:
            raise RuntimeError(_INVALIDATED_RUNTIME_MESSAGE)
        if _users <= 0:
            raise RuntimeError("SHMEM Runtime has no active reference to release")
        if _users > 1:
            _users -= 1
            return

        try:
            _reconfiguration_stage(
                _validate_shutdown_preconditions,
                action="SHMEM release",
                stage="Preflight",
                root_group=_root_group,
            )
            _reconfiguration_stage(
                _synchronize_device,
                action="SHMEM release",
                stage="Quiesce",
                root_group=_root_group,
            )
        except RuntimeError as error:
            convergence_failed = isinstance(error, _collective.CollectiveConvergenceError)
            runtime_unavailable = isinstance(error, _RuntimeUnavailableError)
            if convergence_failed or runtime_unavailable:
                _runtime_invalidated = True
            hint = (_CONVERGENCE_FAILURE_HINT if convergence_failed else
                    _TERMINAL_RUNTIME_HINT if runtime_unavailable else _RETRYABLE_RUNTIME_HINT)
            _raise_lifecycle_failure(str(error), error, hint=hint)
        _runtime_invalidated = True
        try:
            # All peers become terminal if any Native shutdown fails; none may reinitialize alone.
            _reconfiguration_stage(
                _load_native()._shutdown,  # pylint: disable=protected-access
                action="SHMEM release",
                stage="Shutdown",
                root_group=_root_group,
            )
        except RuntimeError as error:
            _raise_lifecycle_failure(
                str(error),
                error,
                hint=_TERMINAL_RUNTIME_HINT,
            )
        finally:
            _users = 0
            _root_group = None
            _root_uses_distributed = None
            _root_size = None
            _root_ranks = None
            _device_index = None
        _runtime_invalidated = False
