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
"""Owned directional workspace for managed Torch MegaMoe execution."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from hyper_parallel.core.multicore.profiling import EVENT_COUNTER_BYTES

from .spec import MegaMoeSpec, _resolve_receive_capacity


_GMM_WORKSPACE_BYTES = 32 * 1024 * 1024
_SWIGLU_GRAD_WORKSPACE_BYTES = 16 * 1024 * 1024
_WORKSPACE_ALIGNMENT = 512
_HEAP_GRANULARITY_BYTES = 64 * 1024 * 1024


def _round_up(value: int, granularity: int) -> int:
    """Round ``value`` up to a positive ``granularity``."""
    return (value + granularity - 1) // granularity * granularity


def _spec_workspace_bytes(specification: Mapping[str, Any], element_size: int) -> int:
    """Return planned symmetric bytes for one execution-resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = _resolve_receive_capacity(
        specification["expert_capacity_factor"],
        routed_slots,
        specification["ep_size"],
    )
    tensor_bytes = (capacity + routed_slots) * specification["hidden_size"] * element_size
    return tensor_bytes + 2 * EVENT_COUNTER_BYTES + 4 * (_WORKSPACE_ALIGNMENT - 1)


def configure_symmetric_heap(
    active_specifications: tuple[Any, ...],
    tensor: torch.Tensor,
) -> int:
    """Size the fixed SHMEM heap once for all active resource groups.

    Args:
        active_specifications: Specifications for every live compatible resource group.
        tensor: First input tensor used to determine element size.

    Returns:
        Configured symmetric heap size in bytes.

    Raises:
        ValueError: If the configured heap size is invalid.
        RuntimeError: If an explicit heap is smaller than the required workspace.
    """
    required_bytes = sum(
        _spec_workspace_bytes(specification, tensor.element_size()) for specification in active_specifications
    )
    required_bytes = _round_up(required_bytes, _HEAP_GRANULARITY_BYTES)
    configured = os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE")
    if configured is None:
        os.environ["SYMMETRIC_MEMORY_HEAP_SIZE"] = str(required_bytes)
        return required_bytes
    try:
        configured_bytes = int(configured)
    except ValueError as error:
        raise ValueError(
            "SYMMETRIC_MEMORY_HEAP_SIZE must be a positive integer number of bytes, " f"got {configured!r}."
        ) from error
    if configured_bytes <= 0:
        raise ValueError(f"SYMMETRIC_MEMORY_HEAP_SIZE must be positive, got {configured_bytes}.")
    if configured_bytes < required_bytes:
        raise RuntimeError(
            "SYMMETRIC_MEMORY_HEAP_SIZE is too small for active MegaMoe resources: "
            f"configured {configured_bytes} bytes, requires at least {required_bytes} bytes."
        )
    return configured_bytes


@dataclass
class MegaMoeWorkspace:
    """Buffers owned by one standalone or explicitly shared resource group."""

    symmetric_memory: Any
    shared: bool
    dtype: Any | None = None
    device: Any | None = None
    expert_capacity: int = 0
    routed_slots: int = 0
    expert_buffer: Any | None = None
    routed_buffer: Any | None = None
    forward_event_counters: Any | None = None
    backward_event_counters: Any | None = None
    gmm_workspace: Any | None = None
    swiglu_grad_workspace: Any | None = None
    completion_event: Any | None = None
    in_use: bool = False
    used: bool = False
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def ensure(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate the fixed configured route capacity once.

        Args:
            spec: First-forward-bound shape and capacity specification.
            dtype: Element dtype for symmetric data buffers.
            device: Device that owns local and symmetric buffers.
        """
        requested_capacity = spec.receive_capacity
        if self.expert_buffer is not None:
            compatible = (
                self.dtype == dtype
                and self.device == device
                and self.routed_slots == spec.routed_slots
                and self.expert_capacity == requested_capacity
            )
            if not compatible:
                raise ValueError("MegaMoe workspace cannot change device, dtype, routed shape, or capacity.")
            return

        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = spec.routed_slots
        try:
            self.expert_buffer = self.symmetric_memory.aligned_empty(
                (requested_capacity, spec.hidden_size),
                dtype,
                _WORKSPACE_ALIGNMENT,
            )
            self.routed_buffer = self.symmetric_memory.aligned_empty(
                (spec.routed_slots, spec.hidden_size),
                dtype,
                _WORKSPACE_ALIGNMENT,
            )
            self.forward_event_counters = self.symmetric_memory.aligned_empty(
                (EVENT_COUNTER_BYTES,),
                torch.uint8,
                _WORKSPACE_ALIGNMENT,
            )
            self.backward_event_counters = self.symmetric_memory.aligned_empty(
                (EVENT_COUNTER_BYTES,),
                torch.uint8,
                _WORKSPACE_ALIGNMENT,
            )
            self.gmm_workspace = torch.empty(
                (_GMM_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.swiglu_grad_workspace = torch.empty(
                (_SWIGLU_GRAD_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.completion_event = torch.npu.Event()
        except Exception:
            self._free_symmetric_tensors()
            self._free_local_tensors()
            raise

    def claim(self) -> None:
        """Claim the serial workspace and order it after the previous stream."""
        with self.lock:
            if self.in_use:
                raise RuntimeError("MegaMoe execution resources do not support concurrent calls.")
            if self.used:
                torch.npu.current_stream().wait_event(self.completion_event)
            self.in_use = True

    def release(self) -> None:
        """Record completion ordering and release the serial workspace lease."""
        with self.lock:
            if not self.in_use:
                raise RuntimeError("MegaMoe workspace was released without an active lease.")
            self.completion_event.record(torch.npu.current_stream())
            self.in_use = False
            self.used = True

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        for field_name in (
            "expert_buffer",
            "routed_buffer",
            "forward_event_counters",
            "backward_event_counters",
        ):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            self.symmetric_memory.free(tensor)
            setattr(self, field_name, None)
        self.expert_capacity = 0
        self.routed_slots = 0

    def _free_local_tensors(self) -> None:
        """Release local-only workspaces after all queued kernels complete."""
        for field_name in ("gmm_workspace", "swiglu_grad_workspace"):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            tensor.untyped_storage().resize_(0)
            setattr(self, field_name, None)

    def close(self) -> None:
        """Synchronize, release all buffers, and reset the workspace."""
        with self.lock:
            if self.in_use:
                raise RuntimeError("cannot close MegaMoe workspace during an active call.")
            if self.expert_buffer is None and self.gmm_workspace is None:
                return
        torch.npu.synchronize(self.device)
        self.symmetric_memory.barrier()
        self._free_symmetric_tensors()
        self.symmetric_memory.barrier()
        self._free_local_tensors()
        self.dtype = None
        self.device = None
        self.completion_event = None
        self.used = False
