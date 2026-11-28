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
"""Bounded Ring transport with document-limited routes and explicit stream ownership."""

from __future__ import annotations

from functools import lru_cache
from math import gcd
from typing import Any

# Native Torch is required; the legacy lint rule predates the Torch-only migration.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import


@lru_cache(maxsize=16)
def _communication_stream(device, _gradient):
    return torch.npu.Stream(device=device)


class _Packet:
    """Protect a receive slot until its producer, readers and outstanding send finish."""

    def __init__(self, storage: torch.Tensor, owner: int, heads: int, dimension: int,
                 counts: tuple[int, ...]) -> None:
        """Bind a flat transport allocation to an owner and its active shape."""
        self.storage = storage
        self.owner = owner
        self.heads = heads
        self.dimension = dimension
        self.counts = counts
        self.receive = None
        self.send = None
        self.consumed = None
        self.produced = None
        self.producer_stream = None
        self._views = {}
        self._recorded_streams = set()

    def ready(self) -> None:
        """Wait for the latest producer, which already includes any preceding receive."""
        if self.produced is not None:
            stream = torch.npu.current_stream()
            if stream != self.producer_stream:
                self.produced.wait(stream)
        elif self.receive is not None:
            self.receive.wait()

    def _active_views(self):
        count = self.counts[self.owner]
        if count not in self._views:
            width = count * self.heads * self.dimension
            payload = self.storage[:2 * width]
            self._views[count] = (payload, payload[:width].view(count, self.heads, self.dimension),
                                 payload[width:].view(count, self.heads, self.dimension))
        return self._views[count]

    def payload(self) -> torch.Tensor:
        """Reuse the active flat send/receive view for this owner size."""
        return self._active_views()[0]

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse K/V views without reconstructing slices at every consumer."""
        return self._active_views()[1:]

    def record_stream(self, stream: Any) -> None:
        """Register each allocator stream once for this storage's lifetime.

        Args:
            stream: Device stream that may retain access to this storage.
        """
        if stream not in self._recorded_streams:
            self.storage.record_stream(stream)
            self._recorded_streams.add(stream)

    def consumed_now(self) -> None:
        """Protect every read already enqueued on the current compute stream."""
        self.consumed = torch.npu.Event()
        self.consumed.record()

    def produced_now(self) -> None:
        """One event covers both gradient production and the final accumulation read."""
        self.producer_stream = torch.npu.current_stream()
        self.produced = torch.npu.Event()
        self.produced.record(self.producer_stream)
        self.consumed = self.produced


class _PackedRingExchange:
    """Double-buffer neighbor traffic, stopping each owner at its last consumer."""

    def __init__(self, initial: torch.Tensor, plan: Any, ring: Any, stream: Any) -> None:
        """Allocate bounded receive slots and order their first write after allocation."""
        self.ring, self.stream, self.plan = ring, stream, plan
        heads, dimension = initial.shape[1:]
        self.buffers = tuple(_Packet(initial.new_empty(2 * plan.capacity * heads * dimension),
                                     ring.rank, heads, dimension, plan.counts) for _ in range(2))
        self.slot = 0
        for packet in self.buffers:
            packet.record_stream(stream)
        stream.wait_stream(torch.npu.current_stream())

    def launch(self, source: _Packet, step: int) -> _Packet:
        """Post the matching subset of neighbor sends and receives for this round.

        Args:
            source: Current owner packet whose data may be forwarded.
            step: Zero-based Ring hop used to enforce route horizons.
        """
        target = self.buffers[self.slot]
        self.slot = 1 - self.slot
        with torch.npu.stream(self.stream):
            if target.send is not None:
                target.send.wait()
            if target.consumed is not None:
                target.consumed.wait()
            target.owner = (source.owner - 1) % self.ring.size
            target.produced, target.producer_stream, target.consumed, target.send = None, None, None, None
            target.receive = None
            send = self.plan.counts[source.owner] and step < self.plan.hops[source.owner]
            receive = self.plan.counts[target.owner] and step < self.plan.hops[target.owner]
            requests = (("send", send), ("recv", receive))
            for kind, active in requests if self.ring.rank % 2 == 0 else reversed(requests):
                if not active:
                    continue
                if kind == "send":
                    source.ready()
                    source.record_stream(self.stream)
                    source.send = dist.isend(source.payload(), self.ring.next_rank, group=self.ring.group)
                else:
                    target.receive = dist.irecv(target.payload(), self.ring.previous_rank, group=self.ring.group)
        return target

    def finish(self, initial: _Packet) -> None:
        """Join outstanding transport before local storage references are released.

        Args:
            initial: Original locally produced packet to join with the receive slots.
        """
        for packet in (initial, *self.buffers):
            packet.ready()
            if packet.send is not None:
                packet.send.wait()
        torch.npu.current_stream().wait_stream(self.stream)


def _receive_first(rank, size, distance):
    # Alternate within each shifted cycle, including non-coprime return distances.
    divisor = gcd(size, distance)
    cycle = size // divisor
    position = (rank // divisor * pow(distance // divisor, -1, cycle)) % cycle
    return position % 2 != 0


def _return_ring_gradient(source, initial, plan, ring, stream, step):
    if not step:
        return
    send = plan.counts[source.owner] and step == plan.hops[source.owner]
    receive = plan.counts[ring.rank] and step == plan.hops[ring.rank]
    if not send and not receive:
        return
    requests = (("send", send), ("recv", receive))
    with torch.npu.stream(stream):
        for kind, active in reversed(requests) if _receive_first(ring.rank, ring.size, step) else requests:
            if not active:
                continue
            if kind == "send":
                source.ready()
                source.record_stream(stream)
                peer = dist.get_global_rank(ring.group, source.owner)
                source.send = dist.isend(source.payload(), peer, group=ring.group)
            else:
                # The owner reuses its initial send allocation only after that send completes.
                initial.send.wait()
                initial.produced, initial.producer_stream = None, None
                peer = dist.get_global_rank(ring.group, (ring.rank + step) % ring.size)
                initial.receive = dist.irecv(initial.payload(), peer, group=ring.group)


def _pack_ring_kv(tensors, spans, ring, counts, out=None):
    pieces = [tensor[start:end] for tensor in tensors for start, end in spans[ring.rank]]
    if out is None:
        storage = torch.cat(pieces, dim=0).reshape(-1) if pieces else tensors[0].new_empty(0)
        out = _Packet(storage, ring.rank, tensors[0].shape[1], tensors[0].shape[-1], counts)
    elif pieces:
        target = out.payload().view(-1, out.heads, out.dimension)
        torch.cat(pieces, dim=0, out=target)
    out.produced_now()
    return out
