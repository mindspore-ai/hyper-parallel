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
"""Owned state packets over a process-local Ascend SHMEM runtime.

Initialization and close are collective. Each arena is an explicit owner; no
communication-slot storage is borrowed by an autograd consumer.
"""

from __future__ import annotations

# Context parallel is Torch-only; this optional component specifically targets NPU.
# pylint: disable=forbidden-backend-import

import importlib
import os
import threading
import weakref
from typing import Any

import torch
import torch.distributed as dist


_SIGNAL_STRIDE = 128


def _validate_arena_inputs(group: Any, state_shape: tuple[int, int, int, int],
                           packets: int, bootstrap_uri: str) -> tuple[int, ...]:
    """Validate the local arena contract before collective initialization."""
    ranks = tuple(dist.get_process_group_ranks(group))
    if len(ranks) < 2 or dist.get_rank(group) < 0:
        raise ValueError("State pull requires a participating member of a CP group of size >= 2.")
    if str(dist.get_backend(group)).lower() != "hccl":
        raise ValueError("State pull requires an initialized HCCL group.")
    if (len(state_shape) != 4 or state_shape[0] != 1 or state_shape[-2] != state_shape[-1]
            or any(dim <= 0 for dim in state_shape) or not 0 < packets <= state_shape[1]):
        raise ValueError("Expected [1,H,K,K] and between one and H nonempty head packets.")
    if not bootstrap_uri or not os.environ.get("SHMEM_UID_SESSION_ID"):
        raise ValueError("Specify bootstrap_uri and a separate SHMEM_UID_SESSION_ID endpoint.")
    return ranks


class _PullRuntime:
    """Share one native SDK instance while keeping arena allocations independent."""

    current = None
    lock = threading.Lock()

    def __init__(self, group: Any, endpoint: str, heap_bytes: int) -> None:
        """Initialize one explicitly selected native SDK runtime."""
        self.group = group
        self.endpoint = endpoint
        self.heap_bytes = heap_bytes
        self.device = torch.npu.current_device()
        self.references = 0
        self.allocated_bytes = 0
        self.submission_lock = threading.Lock()
        # This optional NPU extension must not be imported by the package root.
        self.native = importlib.import_module("_hp_state_pull")
        self.native.init(dist.get_rank(group), dist.get_world_size(group), heap_bytes, endpoint)

    @classmethod
    def acquire(cls, group: Any, endpoint: str, heap_bytes: int) -> _PullRuntime:
        """Share only an identical initialization contract within this process."""
        with cls.lock:
            runtime = cls.current
            if runtime is None:
                runtime = cls(group, endpoint, heap_bytes)
                cls.current = runtime
            elif (runtime.group is not group or runtime.endpoint != endpoint
                  or runtime.heap_bytes != heap_bytes or runtime.device != torch.npu.current_device()):
                raise ValueError("Active SHMEM runtime has a different group, device or initialization contract.")
            runtime.references += 1
            return runtime

    def release(self) -> None:
        """Finalize the SDK after its last arena releases all allocations."""
        with type(self).lock:
            if self.references == 1:
                self.native.release()
                type(self).current = None
            self.references -= 1


class _AutogradLease:
    """Keep the arena alive and reject close before its first backward finishes."""

    def __init__(self, arena: StatePullArena) -> None:
        """Track the first backward without making the arena retain the graph."""
        self.arena = arena
        self.pending_backward = True

    def check(self) -> None:
        """Reject backward if the explicitly owned transport was closed."""
        self.arena.check_stream()

    def complete(self) -> None:
        """Mark first backward complete, allowing collective shutdown."""
        self.pending_backward = False


class StatePullArena:
    """Own symmetric state slots and per-invocation receive tensors.

    Args:
        group: Chronological HCCL group; SDK PE numbers are group-local ranks.
        state_shape: Fixed contiguous FP32 state shape [1,H,K,K].
        packets: Number of nonempty, approximately balanced head packets.
        bootstrap_uri: Explicit SDK bootstrap endpoint, distinct from HCCL.
        heap_bytes: Total SDK heap, shared by compatible arena instances.
        separate_relay_stream: Allow summary production alongside packet relay.

    Note:
        Forward/backward calls must have the same per-direction order on peers.
        Calls may have multiple autograd graphs in flight, but concurrent host
        submission is rejected. Close is collective and checks unfinished graphs.
    """

    def __init__(self, group: Any, state_shape: tuple[int, int, int, int], *, packets: int = 4,
                 bootstrap_uri: str, heap_bytes: int = 192 * 2**20,
                 separate_relay_stream: bool = True) -> None:
        """Allocate matched symmetric slots after collective contract validation."""
        ranks = _validate_arena_inputs(group, state_shape, packets, bootstrap_uri)
        self.group = group
        self.ranks = ranks
        self.rank = dist.get_rank(group)
        self.size = len(ranks)
        self.state_shape = tuple(state_shape)
        self.packets = packets
        self.packet_offsets = tuple(index * state_shape[1] // packets for index in range(packets + 1))
        self.slots = 2
        self.device = torch.device("npu", torch.npu.current_device())
        self.caller_stream = torch.npu.current_stream()
        self.relay_stream = torch.npu.Stream() if separate_relay_stream else self.caller_stream
        self.closed = False
        self.sequence = [0, 0]
        self.last_send = [[[0] * packets for _ in range(self.slots)] for _ in range(2)]
        self.pending = []
        self.active = set()
        self.graphs = weakref.WeakSet()
        contracts = [None] * self.size
        contract = (ranks, self.state_shape, packets, heap_bytes, separate_relay_stream,
                    bootstrap_uri, os.environ["SHMEM_UID_SESSION_ID"])
        dist.all_gather_object(contracts, contract, group=group)
        if any(other != contract for other in contracts):
            raise ValueError("Peers supplied different state arena contracts.")
        torch.npu.synchronize()
        self.runtime = _PullRuntime.acquire(group, bootstrap_uri, heap_bytes)
        self.symmetric_bytes = 2 * self.slots * state_shape[1] * state_shape[2] * state_shape[3] * 4
        self.symmetric_bytes += 2 * (2 * self.slots * packets + 1) * _SIGNAL_STRIDE * 4
        if self.runtime.allocated_bytes + self.symmetric_bytes > heap_bytes:
            self.runtime.release()
            raise ValueError("State arenas exceed the shared SHMEM heap budget; increase heap_bytes collectively.")
        self.buffer = self.runtime.native.empty([2, self.slots, *state_shape], torch.float32)
        shape = [2 * self.slots * packets + 1, _SIGNAL_STRIDE]
        self.ready = self.runtime.native.empty(shape, torch.int32)
        self.credit = self.runtime.native.empty(shape, torch.int32)
        self.ready.zero_()
        self.credit.zero_()
        self.runtime.allocated_bytes += self.symmetric_bytes
        torch.npu.synchronize()
        dist.barrier(group=group)
        peers = [peer for peer in (self.rank - 1, self.rank + 1) if 0 <= peer < self.size]
        accessible = all(self.runtime.native.can_access(self.buffer, peer) for peer in peers)
        vote = torch.tensor(int(accessible), dtype=torch.int32, device=self.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        if not vote.item():
            self.close()
            raise RuntimeError("An adjacent state peer is not mapped for SDMA; this fabric is unsupported.")

    def check_stream(self) -> None:
        """Validate ownership without synchronizing the device in the hot path."""
        if (self.closed or torch.npu.current_device() != self.device.index
                or torch.npu.current_stream().stream_id != self.caller_stream.stream_id):
            raise RuntimeError("State arena is closed or its caller device/stream changed.")

    def retain_autograd(self) -> _AutogradLease:
        """Track a graph weakly; discarded inference/checkpoint graphs cannot leak."""
        self.check_stream()
        lease = _AutogradLease(self)
        self.graphs.add(lease)
        return lease

    def begin(self, direction: int, receives: bool, sends: bool) -> _StatePullExchange:
        """Allocate an invocation-owned input and reserve a directional epoch."""
        self.check_stream()
        if direction not in (0, 1):
            raise ValueError("Direction must be forward (0) or backward (1).")
        # Ownership passes to the invocation and is released by finish().
        if not self.runtime.submission_lock.acquire(blocking=False):  # pylint: disable=consider-using-with
            raise RuntimeError("Concurrent or unfinished state-pull host submission is unsupported.")
        try:
            epoch = self.sequence[direction] + 1
            if epoch >= 2**31:
                raise RuntimeError("State-pull epoch must not wrap; recreate the arena collectively.")
            self.collect_completed()
            exchange = _StatePullExchange(self, direction, epoch, receives, sends)
            self.sequence[direction] = epoch
            self.active.add(id(exchange))
            return exchange
        except Exception:
            self.runtime.submission_lock.release()
            raise

    def collect_completed(self) -> None:
        """Release relay temporaries whose device events have completed, without waiting."""
        self.check_stream()
        self.pending = [(event, values) for event, values in self.pending if not event.query()]

    def memory_stats(self) -> dict[str, int]:
        """Report owned SHMEM bytes separately from Torch allocator statistics.

        The heap is shared by arenas; it is not an additional per-arena allocation.
        Pending storage counts unique retained Torch allocations, including views.
        Saved autograd H/M and independent D are accounted by Torch, not this report.
        """
        self.check_stream()
        self.collect_completed()
        storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                    for _, values in self.pending for tensor in values}
        return {"arena_symmetric_bytes": self.symmetric_bytes,
                "runtime_symmetric_bytes": self.runtime.allocated_bytes,
                "runtime_heap_bytes": self.runtime.heap_bytes,
                "pending_payload_bytes": sum(storages.values()),
                "pending_payload_batches": len(self.pending),
                "pending_backward_graphs": sum(lease.pending_backward for lease in self.graphs)}

    def close(self) -> None:
        """Collectively reject unfinished work before freeing any symmetric storage."""
        if self.closed:
            return
        self.check_stream()
        busy = bool(self.active) or any(lease.pending_backward for lease in self.graphs)
        vote = torch.tensor(int(busy), dtype=torch.int32, device=self.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MAX, group=self.group)
        if vote.item():
            raise RuntimeError("State arena still has an unfinished invocation or pending backward graph.")
        torch.npu.synchronize()
        self.pending.clear()
        for tensor in (self.buffer, self.ready, self.credit):
            self.runtime.native.free(tensor)
        self.buffer = self.ready = self.credit = None
        self.closed = True
        self.runtime.allocated_bytes -= self.symmetric_bytes
        self.runtime.release()


class _StatePullExchange:
    """One acyclic directional invocation; its input never aliases the arena."""

    def __init__(self, arena: StatePullArena, direction: int, epoch: int,
                 receives: bool, sends: bool) -> None:
        """Reserve owned receive storage and its allocation-stream dependency."""
        self.arena = arena
        self.direction = direction
        self.epoch = epoch
        self.slot = (epoch - 1) % arena.slots
        delta = 1 if direction == 0 else -1
        self.source = arena.rank - delta
        self.destination = arena.rank + delta
        if ((receives and not 0 <= self.source < arena.size)
                or (sends and not 0 <= self.destination < arena.size)):
            raise ValueError("Requested a state peer outside the chronological group.")
        self.receives = receives
        self.sends = sends
        self.input_state = torch.empty(arena.state_shape, device=arena.device,
                                       dtype=torch.float32) if receives else None
        # record_stream protects later reuse, but not unfinished prior use on
        # the allocation stream. Receive-only ranks have no producer-ready event.
        self.input_allocated = None
        if receives and arena.relay_stream.stream_id != arena.caller_stream.stream_id:
            self.input_allocated = torch.npu.Event()
            self.input_allocated.record()
        self.next_packet = 0
        self.dependent = None
        self.last_event = None
        self.joined = False
        self.finished = False
        self.payloads = []

    def _relay(self, state: torch.Tensor, matrix: torch.Tensor | None,
               first: int, count: int, receives: bool, sends: bool) -> None:
        arena = self.arena
        begin = arena.packet_offsets[first]
        end = arena.packet_offsets[first + count]
        offsets = [value - begin for value in arena.packet_offsets[first:first + count + 1]]
        flag = (self.direction * arena.slots + self.slot) * arena.packets + 1 + first
        arena.runtime.native.relay(
            state, matrix, self.input_state[:, begin:end] if receives else None,
            arena.buffer[self.direction, self.slot, :, begin:end],
            arena.ready[flag:flag + count], arena.credit[flag:flag + count],
            arena.last_send[self.direction][self.slot][first:first + count], offsets, self.epoch,
            self.source, self.destination, receives, sends)

    def _validate_group(self, state: torch.Tensor, matrix: torch.Tensor | None,
                        first_packet: int, packet_count: int) -> bool:
        """Reject inconsistent packet publication before submitting device work."""
        arena = self.arena
        if (self.finished or self.joined or not self.sends or first_packet != self.next_packet
                or packet_count <= 0 or first_packet + packet_count > arena.packets):
            raise RuntimeError("Summary groups must cover all packets once in chronological order.")
        head_count = arena.packet_offsets[first_packet + packet_count] - arena.packet_offsets[first_packet]
        expected = (1, head_count, *arena.state_shape[2:])
        if tuple(state.shape) != expected:
            raise ValueError("Summary group does not match the declared packet shape.")
        dependent = matrix is not None and self.input_state is not None
        if self.dependent is not None and self.dependent != dependent:
            raise ValueError("One invocation cannot mix dependent and reset summary groups.")
        return dependent

    def publish_group(self, state: torch.Tensor, matrix: torch.Tensor | None,
                      first_packet: int, packet_count: int) -> None:
        """Enqueue ready group packets without waiting for later summary groups."""
        arena = self.arena
        arena.check_stream()
        dependent = self._validate_group(state, matrix, first_packet, packet_count)
        self.dependent = dependent
        separate = arena.relay_stream.stream_id != arena.caller_stream.stream_id
        if separate:
            ready_event = torch.npu.Event()
            ready_event.record()
        with torch.npu.stream(arena.relay_stream):
            if separate:
                arena.relay_stream.wait_event(ready_event)
                for tensor in (state, matrix, self.input_state):
                    if tensor is not None:
                        tensor.record_stream(arena.relay_stream)
            self._relay(state, matrix, first_packet, packet_count, dependent, True)
            self.last_event = torch.npu.Event()
            self.last_event.record()
        self.payloads.extend(tensor for tensor in (state, matrix) if tensor is not None)
        self.next_packet += packet_count

    def wait_input(self) -> torch.Tensor | None:
        """Join the relay before full-head h/dhu; reset outputs are published first."""
        arena = self.arena
        arena.check_stream()
        if self.joined:
            return self.input_state
        if self.sends and self.next_packet != arena.packets:
            raise RuntimeError("The producer did not publish all required head packets.")
        if self.receives and not self.dependent:
            with torch.npu.stream(arena.relay_stream):
                if self.input_allocated is not None:
                    arena.relay_stream.wait_event(self.input_allocated)
                self.input_state.record_stream(arena.relay_stream)
                self._relay(self.input_state, None, 0, arena.packets, True, False)
                self.last_event = torch.npu.Event()
                self.last_event.record()
        if self.last_event is not None:
            arena.caller_stream.wait_event(self.last_event)
        if self.sends:
            arena.last_send[self.direction][self.slot] = [self.epoch] * arena.packets
        self.joined = True
        return self.input_state

    def finish(self) -> None:
        """Retain asynchronous temporaries, then release host invocation ownership."""
        self.arena.check_stream()
        if self.finished or not self.joined:
            raise RuntimeError("State invocation must be joined exactly once before finish.")
        if self.payloads:
            # The last relay event covers all payload reads; local h/dhu need
            # not finish before these producer temporaries can be reclaimed.
            self.arena.pending.append((self.last_event, self.payloads))
            self.payloads = []
        self.finished = True
        self.arena.active.remove(id(self))
        self.arena.runtime.submission_lock.release()
