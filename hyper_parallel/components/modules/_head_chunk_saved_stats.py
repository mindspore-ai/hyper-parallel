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
"""Invocation-local FA-state offload with bounded, event-ordered device slots."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.core.activation_memory import PinnedMemoryPool

logger = logging.getLogger(__name__)


class _OffloadStatistics:
    """Integer-only counters; never retain an invocation or any tensor/event."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = dict.fromkeys((
            "active_stores", "d2h_stages", "d2h_bytes", "h2d_stages", "h2d_bytes",
            "live_host_bytes", "peak_host_bytes", "device_slots", "peak_device_slots",
            "device_slot_bytes", "peak_device_slot_bytes",
            "d2h_source_tickets", "live_d2h_source_tickets", "peak_d2h_source_tickets",
            "live_d2h_source_bytes", "peak_d2h_source_bytes", "d2h_source_retired_tickets",
            "d2h_source_fallbacks", "d2h_source_layout_fallbacks", "d2h_source_cross_stream_fallbacks",
        ), 0)

    def update(self, **changes: int) -> None:
        """Apply live-capacity deltas and cumulative transfer counters."""
        with self._lock:
            for key, delta in changes.items():
                value = self._values[key] + delta
                if value < 0:
                    raise RuntimeError(f"Unbalanced saved-state ownership counter: {key}")
                self._values[key] = value
            for current, peak in (("live_host_bytes", "peak_host_bytes"),
                                  ("device_slots", "peak_device_slots"),
                                  ("device_slot_bytes", "peak_device_slot_bytes"),
                                  ("live_d2h_source_tickets", "peak_d2h_source_tickets"),
                                  ("live_d2h_source_bytes", "peak_d2h_source_bytes")):
                self._values[peak] = max(self._values[peak], self._values[current])

    def snapshot(self) -> dict[str, int]:
        """Copy counters without keeping the runtime objects they describe."""
        with self._lock:
            return dict(self._values)


@dataclass
class _HostTensor:
    """One typed lease; released state makes exception cleanup idempotent."""

    buffer: torch.Tensor
    view: torch.Tensor
    released: bool = False


@dataclass(frozen=True)
class SavedStatsLoadTicket:
    """One generation of a device slot, ordered by an H2D completion event."""

    slot: int
    tensors: tuple[torch.Tensor, ...]
    completion_event: Any


@dataclass(frozen=True)
class _D2HSourceTicket:
    """Keep one stage's private FA storage until a delayed owner-stream wait."""

    tensors: tuple[torch.Tensor, ...]
    owner_stream: Any
    completion_event: Any
    storage_bytes: int


class SavedStatsOffloadStore:
    """Save one forward's FA states on host and consume them once in backward.

    The pinned pool may be shared by multiple outstanding forwards, but all
    leases, copy-stream events and device slots are invocation-local. Slots
    are sized to the largest stage assigned to them, including tail stages.
    Host leases are returned only after H2D completes. ``close`` drains the
    private copy stream, not the whole device, and releases unused leases.

    Optional source retirement relies on private FA outputs allocated on the
    caller's compute stream. Tensor exposes no allocation-stream query. Only
    distinct contiguous storages qualify; other layouts retain the original
    allocator protection. At the next stage, the owner stream waits for the
    previous D2H before its copy-stream marker is erased. ``finish_forward``
    retires the tail so source references never accumulate across layers.

    Args:
        device: NPU device for this invocation.
        pool: Capacity-bounded pinned pool shared by this attention module.
        slots: One for demand loading, or two for one-stage look-ahead.
        statistics: Optional module-owned integer-only aggregate counters.
        retire_d2h_sources: Retire one-stage-delayed D2H source protection.
            Requires the private same-allocation-stream FA-output contract.
    """

    def __init__(
        self, device: torch.device, pool: PinnedMemoryPool, slots: int = 2,
        *, statistics: _OffloadStatistics | None = None,
        retire_d2h_sources: bool = False,
    ) -> None:
        self._closed = True
        if device.type != "npu":
            raise ValueError(f"saved-state offload requires an NPU device, got {device}")
        if isinstance(slots, bool) or not isinstance(slots, int) or slots not in (1, 2):
            raise ValueError("saved-state prefetch slots must be one or two")
        if not isinstance(retire_d2h_sources, bool):
            raise ValueError("retire_d2h_sources must be a boolean")
        self.device, self.pool, self.slots = device, pool, slots
        self._statistics = _OffloadStatistics()
        self._aggregate = statistics
        self._stages: dict[int, tuple[_HostTensor, ...]] = {}
        self._write_events: dict[int, Any] = {}
        self._slot_capacities: dict[int, tuple[tuple[int, torch.dtype], ...]] = {}
        self._device_slots: dict[int, tuple[torch.Tensor, ...]] = {}
        self._reusable_events: dict[int, Any] = {}
        self._active_slots: dict[int, SavedStatsLoadTicket] = {}
        self._loading = False
        self._next_stage = 0
        self._retire_d2h_sources = retire_d2h_sources
        self._d2h_source: _D2HSourceTicket | None = None
        self.copy_stream = torch.npu.Stream(device=device)
        self._record(active_stores=1)
        self._closed = False

    def _record(self, **changes: int) -> None:
        self._statistics.update(**changes)
        if self._aggregate is not None:
            self._aggregate.update(**changes)

    @staticmethod
    def _bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("saved-state offload store is closed; offloaded graphs support one backward only")

    def _allocate_host(self, tensor: torch.Tensor) -> _HostTensor:
        size = self._bytes(tensor)
        buffer = self.pool.acquire(size)
        try:
            view = buffer.view(tensor.dtype).view(tensor.shape)
        except Exception:
            self.pool.release(buffer)
            raise
        self._record(live_host_bytes=size)
        return _HostTensor(buffer, view)

    def _release_host(self, entry: _HostTensor, event: Any | None = None) -> None:
        if not entry.released:
            self.pool.release(entry.buffer, event=event)
            entry.released = True
            self._record(live_host_bytes=-self._bytes(entry.view))

    @staticmethod
    def _erase_copy_stream(tensors: tuple[torch.Tensor, ...], stream: Any) -> None:
        # The allocator API is optional until this NPU-only feature executes.
        import torch_npu  # pylint: disable=import-outside-toplevel

        for tensor in tensors:
            torch_npu.erase_stream(tensor, stream)

    def _retire_d2h_source(self) -> None:
        ticket = self._d2h_source
        if ticket is None:
            return
        current = torch.npu.current_stream(self.device)
        if current == ticket.owner_stream:
            # Delay the wait until the next FA has been submitted, preserving
            # its overlap with the previous D2H without host synchronization.
            current.wait_event(ticket.completion_event)
            self._erase_copy_stream(ticket.tensors, self.copy_stream)
            self._record(d2h_source_retired_tickets=1)
        else:
            # A wait on another stream cannot protect allocation-stream reuse.
            # Keep the original marker and let the allocator retire this copy.
            self._record(d2h_source_fallbacks=1, d2h_source_cross_stream_fallbacks=1)
        self._drop_d2h_source()

    def _drop_d2h_source(self) -> None:
        ticket = self._d2h_source
        if ticket is None:
            return
        self._record(live_d2h_source_tickets=-1, live_d2h_source_bytes=-ticket.storage_bytes)
        self._d2h_source = None

    def _retain_d2h_source(self, tensors: tuple[torch.Tensor, ...], completion: Any, owner: Any) -> None:
        storages = tuple(tensor.untyped_storage() for tensor in tensors)
        if (any(not tensor.is_contiguous() for tensor in tensors)
                or len({storage.data_ptr() for storage in storages}) != len(storages)):
            self._record(d2h_source_fallbacks=1, d2h_source_layout_fallbacks=1)
            return
        storage_bytes = sum(storage.nbytes() for storage in storages)
        self._d2h_source = _D2HSourceTicket(
            tuple(tensor.detach() for tensor in tensors), owner, completion, storage_bytes,
        )
        self._record(d2h_source_tickets=1, live_d2h_source_tickets=1, live_d2h_source_bytes=storage_bytes)

    def finish_forward(self) -> None:
        """Retire the final D2H source without holding device storage into backward."""
        self._check_open()
        self._retire_d2h_source()

    def offload_stage(self, stage_index: int, tensors: tuple[torch.Tensor, ...]) -> None:
        """Enqueue D2H; optionally retain this source until the next stage boundary."""
        self._check_open()
        if self._loading or stage_index != self._next_stage:
            raise ValueError("saved states must be offloaded in consecutive forward order, before backward")
        if not tensors or any(tensor.device != self.device or tensor.numel() == 0 for tensor in tensors):
            raise ValueError("saved-state tensors must be nonempty and reside on the invocation device")
        if self._retire_d2h_sources:
            self._retire_d2h_source()
        slot = stage_index % self.slots
        capacities = tuple((tensor.numel(), tensor.dtype) for tensor in tensors)
        previous = self._slot_capacities.get(slot)
        if previous is not None:
            if len(previous) != len(capacities) or any(old[1] != new[1] for old, new in zip(previous, capacities)):
                raise ValueError("saved-state slot tensor counts and dtypes must remain consistent")
            capacities = tuple((max(old[0], new[0]), new[1]) for old, new in zip(previous, capacities))
        entries = []
        try:
            for tensor in tensors:
                entries.append(self._allocate_host(tensor))
            owner = torch.npu.current_stream(self.device)
            producer = torch.npu.Event()
            producer.record(owner)
            with torch.npu.stream(self.copy_stream):
                self.copy_stream.wait_event(producer)
                for entry, tensor in zip(entries, tensors):
                    entry.view.copy_(tensor.contiguous(), non_blocking=True)
                completion = torch.npu.Event()
                completion.record(self.copy_stream)
            for tensor in tensors:
                tensor.record_stream(self.copy_stream)
            if self._retire_d2h_sources:
                self._retain_d2h_source(tensors, completion, owner)
        except Exception:
            self.copy_stream.synchronize()
            for entry in entries:
                self._release_host(entry)
            raise
        self._stages[stage_index] = tuple(entries)
        self._write_events[stage_index] = completion
        self._slot_capacities[slot] = capacities
        self._next_stage += 1
        self._record(d2h_stages=1, d2h_bytes=sum(self._bytes(tensor) for tensor in tensors))

    def prefetch_stage(self, stage_index: int) -> SavedStatsLoadTicket:
        """Enqueue H2D after producer/reuse events into one bounded device slot."""
        self._check_open()
        self._loading = True
        if stage_index not in self._stages:
            raise ValueError(f"unknown or already consumed saved-state stage {stage_index}")
        slot = stage_index % self.slots
        if slot in self._active_slots:
            raise RuntimeError(f"saved-state device slot {slot} is still active")
        entries = self._stages[stage_index]
        buffers = self._device_slots.get(slot)
        if buffers is None:
            buffers = tuple(torch.empty(count, dtype=dtype, device=self.device)
                            for count, dtype in self._slot_capacities[slot])
            self._device_slots[slot] = buffers
            self._record(device_slots=1, device_slot_bytes=sum(self._bytes(tensor) for tensor in buffers))
        tensors = tuple(buffer[:entry.view.numel()].view(entry.view.shape)
                        for buffer, entry in zip(buffers, entries))
        allocation_ready = torch.npu.Event()
        allocation_ready.record(torch.npu.current_stream(self.device))
        with torch.npu.stream(self.copy_stream):
            self.copy_stream.wait_event(self._write_events[stage_index])
            reuse = self._reusable_events.get(slot)
            if reuse is not None:
                self.copy_stream.wait_event(reuse)
            self.copy_stream.wait_event(allocation_ready)
            for tensor, entry in zip(tensors, entries):
                tensor.copy_(entry.view, non_blocking=True)
            completion = torch.npu.Event()
            completion.record(self.copy_stream)
        for tensor in tensors:
            tensor.record_stream(self.copy_stream)
        # Keep each lease's released bit until the entire transfer of ownership
        # succeeds, so an exception cannot make close return a lease twice.
        for entry in entries:
            self._release_host(entry, event=completion)
        del self._stages[stage_index]
        del self._write_events[stage_index]
        self._reusable_events.pop(slot, None)
        ticket = SavedStatsLoadTicket(slot, tensors, completion)
        self._active_slots[slot] = ticket
        self._record(h2d_stages=1, h2d_bytes=sum(self._bytes(tensor) for tensor in tensors))
        return ticket

    @staticmethod
    def consume(ticket: SavedStatsLoadTicket) -> tuple[torch.Tensor, ...]:
        """Make the compute stream wait before reading a prefetched stage."""
        stream = torch.npu.current_stream(ticket.tensors[0].device)
        stream.wait_event(ticket.completion_event)
        for tensor in ticket.tensors:
            tensor.record_stream(stream)
        return ticket.tensors

    def release_slot(self, ticket: SavedStatsLoadTicket) -> None:
        """Order the next slot overwrite after its current-stream consumer."""
        self._check_open()
        if self._active_slots.get(ticket.slot) is not ticket:
            raise RuntimeError("saved-state slot ticket is not the active generation")
        completion = torch.npu.Event()
        completion.record(torch.npu.current_stream(self.device))
        self._reusable_events[ticket.slot] = completion
        del self._active_slots[ticket.slot]

    def statistics(self) -> dict[str, int]:
        """Return integer transfer counts and actual owned-buffer capacities."""
        result = self._statistics.snapshot()
        result.update(pool_capacity_bytes=self.pool.total_allocated, pool_limit_bytes=self.pool.max_host_bytes)
        return result

    def close(self) -> None:
        """Drain copies and return all leases, including abandoned forward state."""
        if self._closed:
            return
        self.copy_stream.synchronize()
        # Copies are complete, so retained markers are safe. Cleanup must not
        # depend on the optional erase API succeeding a second time.
        self._drop_d2h_source()
        for entries in self._stages.values():
            for entry in entries:
                self._release_host(entry)
        self._stages.clear()
        self._write_events.clear()
        device_bytes = sum(self._bytes(tensor) for buffers in self._device_slots.values() for tensor in buffers)
        self._record(active_stores=-1, device_slots=-len(self._device_slots), device_slot_bytes=-device_bytes)
        # consume() recorded compute-stream ownership. Dropping buffers here
        # cannot recycle their storage before the last enqueued FA consumer.
        self._active_slots.clear()
        self._device_slots.clear()
        self._reusable_events.clear()
        self._slot_capacities.clear()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.exception("Failed to release Head Chunk saved-state offload resources")


__all__ = ["SavedStatsLoadTicket", "SavedStatsOffloadStore"]
