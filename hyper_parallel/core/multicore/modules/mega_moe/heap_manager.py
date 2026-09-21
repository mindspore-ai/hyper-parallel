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
"""Coordinate grow-only replacement of all managed workspaces in one SHMEM root."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.shmem import _lifecycle

from .spec import _align_capacity, _resolve_receive_capacity
from .workspace import _HEAP_GRANULARITY_BYTES, _round_up, _spec_workspace_bytes, configure_symmetric_heap

_MANAGER = None
_SYMMETRIC_FIELDS = ("source_buffer", "expert_buffer", "routed_buffer",
                     "forward_event_counters", "backward_event_counters")


def root_members(group: Any) -> tuple[int, ...]:
    """Resolve ordered root membership independently of Python ProcessGroup identity.

    Args:
        group: EP process group, or None for the default root.
    """
    if not dist.is_initialized():
        return (0,)
    return tuple(dist.get_process_group_ranks(dist.group.WORLD if group is None else group))


@dataclass
class _Entry:
    specification: dict[str, Any]
    capacity: int
    resource: Any = None


class MegaMoeHeapManager:
    """Keep logical workspaces alive while replacing their collective physical heap."""

    def __init__(self, spec: Any, tensor: torch.Tensor, specifications: tuple[Any, ...]) -> None:
        """Reserve the initial layout without taking a native owner reference."""
        self.group = spec.ep_group
        self.members = root_members(spec.ep_group)
        self.device = tensor.device
        self.element_size = tensor.element_size()
        self.entries: list[_Entry] = []
        self.epoch = 0
        self.state = "ready"
        self.growth_records: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._rebuild_thread = None
        self.heap_bytes = configure_symmetric_heap(specifications, tensor, write_environment=False)
        configured = os.getenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE")
        self.heap_limit = int(configured) if configured is not None else None
        self._reserve(specifications)

    @property
    def resources(self) -> list[Any]:
        """Return bound owners in their deterministic allocation order."""
        return [entry.resource for entry in self.entries if entry.resource is not None]

    @contextmanager
    def access(self) -> Iterator[None]:
        """Reject concurrent submissions while allowing the rebuild thread to bind buffers."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("MegaMoe root does not support concurrent heap operations")
        try:
            if self.state == "failed" or (self.state != "ready" and self._rebuild_thread != threading.get_ident()):
                raise RuntimeError(f"MegaMoe heap is {self.state}; no further execution is allowed")
            yield
        finally:
            self._lock.release()

    def _reserve(self, specifications: tuple[Any, ...]) -> None:
        for item in specifications:
            if not any(entry.specification is item for entry in self.entries):
                capacity = _resolve_receive_capacity(item["initial_capacity_factor"],
                                                     item["local_num_tokens"] * item["top_k"], item["ep_size"])
                self.entries.append(_Entry(item, capacity))

    def reserve(self, specifications: tuple[Any, ...]) -> None:
        """Include lazy owners before they allocate from the shared heap.

        Args:
            specifications: Ordered specifications of all known root resource groups.
        """
        with self.access():
            previous = len(self.entries)
            self._reserve(specifications)
            capacities = [entry.capacity for entry in self.entries]
            required = self._required_bytes(capacities)
            if required <= self.heap_bytes:
                return
            try:
                if not any(entry.specification.get("dispatch_mode", "push") == "push" for entry in self.entries):
                    raise RuntimeError("Create all pull MegaMoe resources before initializing the heap")
                target = self._budget(required)
                self._rebuild(capacities, target)
            except Exception:
                if self.state == "ready":
                    del self.entries[previous:]
                raise

    def bind(self, resource: Any, specification: dict[str, Any]) -> None:
        """Register one acquired owner and preserve its stable workspace identity.

        Args:
            resource: Execution resources owning one native reference and workspace.
            specification: The original reserved specification object.
        """
        for entry in self.entries:
            if entry.specification is specification:
                if entry.resource is not None:
                    raise RuntimeError("MegaMoe heap owner is already bound")
                entry.resource = resource
                resource.workspace.heap_manager = self
                resource.workspace.capacity_floor = entry.capacity
                self.heap_bytes = shmem.debug_state()["config"]["heap_size_bytes"]
                return
        raise RuntimeError("MegaMoe heap owner was not reserved")

    def remove(self, resource: Any) -> None:
        """Forget a cleanly closed owner after its symmetric allocations are freed.

        Args:
            resource: Owner whose native reference has already been released.
        """
        self.entries[:] = [entry for entry in self.entries if entry.resource is not resource]

    def _required_bytes(self, capacities: list[int]) -> int:
        size = sum(_spec_workspace_bytes(entry.specification, self.element_size, receive_capacity=capacity)
                   for entry, capacity in zip(self.entries, capacities))
        return _round_up(size, _HEAP_GRANULARITY_BYTES)

    def _budget(self, required: int) -> int:
        if self.heap_limit is not None and required > self.heap_limit:
            raise RuntimeError(
                f"MegaMoe growth exceeds explicit heap budget: required={required}, limit={self.heap_limit}")
        return self.heap_limit if self.heap_limit is not None else max(self.heap_bytes, required)

    def ensure_capacity(self, resource: Any, maximum_received_slots: int) -> None:
        """Grow before launching a route that cannot fit the current symmetric receive buffer.

        Args:
            resource: Managed owner about to launch its forward.
            maximum_received_slots: Maximum destination load from the existing root count exchange.
        """
        if maximum_received_slots <= resource.workspace.capacity_floor:
            return
        with self.access():
            index = next(i for i, entry in enumerate(self.entries) if entry.resource is resource)
            capacities = [entry.capacity for entry in self.entries]
            current = capacities[index]
            if maximum_received_slots <= current:
                return
            spec = resource.spec
            upper = _align_capacity(spec.ep_size * spec.routed_slots)
            if maximum_received_slots > upper:
                raise ValueError("MegaMoe route exceeds the lossless token bound")
            requested = max(maximum_received_slots, math.ceil(min(upper, current * spec.capacity_growth_factor)))
            capacities[index] = min(upper, _align_capacity(requested))
            required = self._required_bytes(capacities)
            if self.heap_limit is not None and required > self.heap_limit:
                capacities[index] = _align_capacity(maximum_received_slots)
                required = self._required_bytes(capacities)
            # Identical gathered counts and the validated root layout make this decision collective.
            self._rebuild(capacities, self._budget(required))

    def _consensus(self, value: Any, error: str | None = None) -> None:
        payload = (value, error)
        peers = [payload]
        if len(self.members) > 1:
            peers = [None] * len(self.members)
            dist.all_gather_object(peers, payload, group=self.group)
        failures = [(rank, item[1]) for rank, item in zip(self.members, peers) if item[1] is not None]
        if failures:
            raise RuntimeError(f"MegaMoe heap reconfiguration failed: {failures}")
        if any(item[0] != value for item in peers):
            raise RuntimeError("MegaMoe heap layout, epoch or capacity differs across EP ranks")

    def _preflight(self, capacities: list[int], heap_bytes: int) -> None:
        error = None
        try:
            state = shmem.debug_state()
            if state["reference_count"] != len(self.resources):
                raise RuntimeError("unmanaged SHMEM owner prevents heap growth")
            pointers = set()
            for resource in self.resources:
                workspace = resource.workspace
                if workspace.in_use:
                    raise RuntimeError("active workspace lease prevents heap growth")
                for name in _SYMMETRIC_FIELDS:
                    tensor = getattr(workspace, name)
                    if tensor is not None:
                        pointers.add(tensor.data_ptr())
            if pointers != {record["allocation_base"] for record in state["active_allocations"]}:
                raise RuntimeError("unmanaged SHMEM allocation prevents heap growth")
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("MegaMoe heap growth is unavailable during graph capture")
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        layout = tuple((tuple(sorted((key, value) for key, value in entry.specification.items() if key != "ep_group")),
                        capacity, entry.resource is not None,
                        entry.resource is not None and entry.resource.workspace.completion_event is not None)
                       for entry, capacity in zip(self.entries, capacities))
        self._consensus((self.epoch, heap_bytes, layout), error)

    def _stage(self, name: str, operation: Any, timings: dict[str, float]) -> None:
        start = time.perf_counter()
        error = None
        try:
            operation()
        except Exception as exception:
            error = f"{name}: {type(exception).__name__}: {exception}"
        self._consensus(name, error)
        timings[name] = (time.perf_counter() - start) * 1000

    def _rebuild(self, capacities: list[int], heap_bytes: int) -> None:
        with _lifecycle._reconfiguration():
            start = time.perf_counter()
            self._preflight(capacities, heap_bytes)
            timings = {}
            self._stage("quiesce", lambda: torch.npu.synchronize(self.device), timings)
            self.state = "rebuilding"
            self._rebuild_thread = threading.get_ident()
            allocated = [(entry.resource, entry.resource.workspace.dtype, entry.resource.workspace.device)
                         for entry in self.entries if entry.resource is not None
                         and entry.resource.workspace.completion_event is not None]
            old_bytes = self.heap_bytes
            try:
                for index, (resource, _, _) in enumerate(allocated):
                    self._stage(f"free_{index}", resource.workspace.close, timings)
                self._stage("reinitialize", lambda: timings.update(_lifecycle._reinitialize(heap_bytes)), timings)
                for entry, capacity in zip(self.entries, capacities):
                    entry.capacity = capacity
                    if entry.resource is not None:
                        entry.resource.workspace.capacity_floor = capacity
                for index, (resource, dtype, device) in enumerate(allocated):
                    self._stage(f"allocate_{index}",
                                lambda r=resource, t=dtype, d=device: r.workspace.ensure(r.spec, t, d), timings)
                self._consensus(("commit", self.epoch + 1))
                self.heap_bytes = heap_bytes
                self.epoch += 1
                self.state = "ready"
                self.growth_records.append({"epoch": self.epoch, "old_heap_bytes": old_bytes,
                                            "heap_bytes": heap_bytes, "capacities": capacities,
                                            "stages_ms": timings, "total_ms": (time.perf_counter() - start) * 1000})
            except Exception:
                self.state = "failed"
                _lifecycle._invalidate_runtime()
                raise
            finally:
                self._rebuild_thread = None


def get_heap_manager(spec: Any, tensor: torch.Tensor, specifications: tuple[Any, ...]) -> MegaMoeHeapManager:
    """Select the sole active root coordinator and reserve all known resource groups.

    Args:
        spec: Bound shape and topology of the requesting owner.
        tensor: Activation defining this root's device and element size.
        specifications: All live specifications in deterministic resource order.
    """
    global _MANAGER  # pylint: disable=global-statement
    if _MANAGER is None or not _MANAGER.resources:
        _MANAGER = MegaMoeHeapManager(spec, tensor, specifications)
    if _MANAGER.members != root_members(spec.ep_group) or _MANAGER.device != tensor.device:
        raise RuntimeError("MegaMoe heap requires one device and ordered SHMEM root per process")
    if _MANAGER.element_size != tensor.element_size():
        raise ValueError("MegaMoe heap owners must use the same element size")
    _MANAGER.reserve(specifications)
    return _MANAGER
