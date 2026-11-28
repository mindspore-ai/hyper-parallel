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
"""Shared state exchange for the Ascend C KDA and GDN drivers.

Local preparation and backward math stay in the algorithm-specific drivers.
This module owns document-bounded state traffic and invocation-local buffer
lifetimes for the native FP32 128-by-128 map format. AllGather metadata is also
reused by the Triton KDA protocol without importing either kernel backend.
"""
from __future__ import annotations
from bisect import bisect_left
from collections import deque
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state


_ENTRY_CACHE_KEY = object()


class _CPEntryCache:
    """Store actual entries in checkpoint-call order, independently of autograd saves."""

    def __init__(self) -> None:
        self.entries = deque()

    def save(self, signature: tuple, entry: torch.Tensor | None) -> None:
        """Own a snapshot rather than a view of a receive or packed-state buffer."""
        self.entries.append((signature, None if entry is None else entry.detach().clone()))

    def pop(self, signature: tuple) -> torch.Tensor | None:
        """Reject missing or reordered replay before issuing any state communication."""
        if not self.entries:
            raise RuntimeError(
                "CP entry cache is missing for checkpoint replay; repeated replay after "
                "resource cleanup is unsupported.")
        expected, entry = self.entries.popleft()
        if expected != signature:
            raise RuntimeError("CP checkpoint replay changed the layer, state group, shape, or document layout.")
        return entry

    def clear(self) -> None:
        """Release consumed and unused entries on replay exit or an exception."""
        self.entries.clear()


def _apply_state(state, transition, initial):
    """Keep the established FP32 matmul/add order outside model autocast."""
    if initial is None:
        return state
    with torch.autocast(device_type=state.device.type, enabled=False):
        return torch.matmul(transition.float(), initial.float()) + state


class NativeStateAllGather:
    """Precreated communication metadata; gathered maps belong to each invocation."""

    def __init__(self, group: Any, rank: int, size: int) -> None:
        """Borrow one precreated group; do not retain gathered tensors."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or not 0 <= rank < size:
            raise ValueError("AllGather requires a positive size and valid local rank.")
        self.group, self.rank, self.size = group, rank, size

    def gather_packed(self, packed: torch.Tensor) -> torch.Tensor:
        """Gather one native [1,H,128,256] map without another packing copy."""
        output = packed.new_empty((self.size, *packed.shape[1:]))
        dist.all_gather_into_tensor(output, packed, group=self.group)
        return output


def native_state_intervals(info: Any, tokens: int, rank: int, size: int) -> tuple:
    """Return tail/head windows and document-bounded exclusive scan lengths."""
    if info is None:
        return (0, tokens), (0, tokens), rank, size - rank - 1
    first_rank = (info.start - info.preceding_sample_tokens) // tokens
    last_rank = (info.global_offsets[bisect_left(info.global_offsets, info.end)] - 1) // tokens
    return ((info.tail.start, info.tail.stop), (info.head.start, info.head.stop),
            rank - first_rank if info.continues_previous else 0,
            last_rank - rank if info.continues_next else 0)


class NativeStateSends:
    """Drain every packet while the caller retains the corresponding payloads."""

    def __init__(self, handles: list[Any]) -> None:
        """Own all outstanding handles for one communication direction."""
        self.handles = handles

    def wait(self) -> None:
        """Finish all sends before releasing this invocation's payload storage."""
        for handle in self.handles:
            handle.wait()


class NativeStateExchange:
    """Bind one invocation's sequence plan to precreated communication groups.

    Only metadata survives forward. Receive buffers and send handles are owned
    by the active direction and drained before that invocation returns.
    """

    def __init__(self, mesh: Any, executor: Any, info: Any, local_length: int, heads: int,
                 p2p_head_chunks: int = 1, *, reuse_cp_state_on_recompute: bool = False,
                 cache_key: object = None, algorithm: str = "") -> None:
        """Copy communication metadata and retain only this call's sequence plan."""
        self.executor, self.info, self.heads = executor, info, heads
        self.reuse_cp_state_on_recompute = reuse_cp_state_on_recompute
        self.cache_key, self.algorithm, self.local_length = cache_key, algorithm, local_length
        self._entry_cache = self._cache_signature = self._replay_entry = None
        self.rank = mesh.get_local_rank()
        self.size = mesh.size()
        self.ranks = tuple(mesh.rank_list)
        self.group = mesh.get_group()
        self.width = getattr(executor, "width", self.size) if executor is not None else 1
        self.block, self.local = divmod(self.rank, self.width)
        self.owner = self.block * self.width + self.width - 1
        self.is_rd = getattr(executor, "protocol", None) == "recursive_doubling"
        if (isinstance(p2p_head_chunks, bool) or not isinstance(p2p_head_chunks, int)
                or not 1 <= p2p_head_chunks <= heads):
            raise ValueError("p2p_head_chunks must be a positive integer no larger than local heads.")
        if p2p_head_chunks > 1 and (self.width != 1 or self.is_rd):
            raise ValueError("p2p_head_chunks applies only to ordinary state P2P.")
        self.head_ranges = tuple((index * heads // p2p_head_chunks, (index + 1) * heads // p2p_head_chunks)
                                 for index in range(p2p_head_chunks))
        self.sparse = None
        if getattr(executor, "protocol", None) == "alltoallv" and info is not None:
            self.sparse = executor.bind_sequence(info, state_shape=(1, heads, 128, 128))
        if reuse_cp_state_on_recompute and (self.is_rd or self.sparse is not None or self.width not in (1, self.size)):
            raise NotImplementedError("Checkpoint entry caching currently supports native P2P and AllGather.")
        self.plans = (tuple(executor.directional_plan(info, local_length, reverse=reverse) for reverse in (False, True))
                      if self.is_rd else ())

    def peer(self, forward: bool, receiving: bool) -> int | None:
        """Return a chronological peer, pruning disconnected ordinary P2P edges.

        Args:
            forward: Whether to use forward state traffic rather than backward gradient traffic.
            receiving: Whether to select the incoming peer rather than the outgoing peer.
        """
        direction = 1 if forward else -1
        peer = self.owner + (-direction if receiving else direction) * self.width
        if not 0 <= peer < self.size:
            return None
        if self.width == 1 and self.info is not None:
            needs_previous = forward == receiving
            connected = self.info.continues_previous if needs_previous else self.info.continues_next
            if not connected:
                return None
        return peer

    def needs_summary(self, forward: bool) -> bool:
        """Avoid producing a summary without an exit consumer.

        Args:
            forward: Select the forward S/M or backward G/N direction.
        """
        if forward and self._replay_entry is not None:
            return False
        if self.is_rd:
            plan = self.plans[not forward]
            return plan.shift_peers(plan.local_rank)[0] is not None
        return self.width > 1 or self.peer(forward, receiving=False) is not None

    def receive(self, reference: torch.Tensor, forward: bool) -> tuple[torch.Tensor | None, Any]:
        """Post reception before native prepare, independently of summary production.

        Args:
            reference: Local input supplying the device.
            forward: Select the chronological communication direction.
        """
        if forward and self._bind_checkpoint_entry(reference):
            return None, None
        if self.is_rd or self.width == self.size:
            return None, None
        incoming = reference.new_zeros((1, self.heads, 128, 128), dtype=torch.float32)
        peer = self.peer(forward, receiving=True)
        handle = None
        if self.rank == self.owner and peer is not None:
            end = self.head_ranges[0][1] if self.width == 1 else self.heads
            handle = dist.irecv(incoming[:, :end], src=self.ranks[peer], group=self.group)
        return incoming, handle

    def _bind_checkpoint_entry(self, reference: torch.Tensor) -> bool:
        """Bind this direction to an invocation cache, never a shared module tensor."""
        state = get_recompute_state() if self.reuse_cp_state_on_recompute else None
        if state is None:
            return False
        sequence = None if self.info is None else (
            self.info.global_offsets, self.info.start, self.info.end)
        signature = (self.cache_key, self.algorithm, self.group, self.ranks, self.rank,
                     self.heads, self.local_length, self.width, self.head_ranges,
                     sequence, reference.device, reference.dtype)
        cache = state.get_resource(_ENTRY_CACHE_KEY, _CPEntryCache)
        if state.is_recomputing:
            self._replay_entry = (cache.pop(signature),)
            return True
        self._entry_cache, self._cache_signature = cache, signature
        return False

    @staticmethod
    def apply(packed: torch.Tensor, incoming: torch.Tensor | None) -> torch.Tensor:
        """Apply forward M or native backward N directly, without transposing N.

        Args:
            packed: Packed state and transition tensors for the scan.
            incoming: State entering this rank or group.
        """
        return _apply_state(packed[..., :128], packed[..., 128:], incoming)

    def exchange(self, ops: Any, packed: torch.Tensor | None,
                 reception: tuple[torch.Tensor | None, Any], forward: bool,
                 count: int) -> tuple[torch.Tensor, Any, torch.Tensor | list[torch.Tensor] | None]:
        """Produce the local input state and retain an asynchronous outgoing payload.

        Args:
            ops: Native summary-merge operations for AllGather.
            packed: Local S/M or G/N, absent when no exit consumes it.
            reception: Previously posted receive buffer and handle.
            forward: Select the chronological communication direction.
            count: Number of maps in the relevant AllGather prefix or suffix.
        """
        if forward and self._replay_entry is not None:
            entry, = self._replay_entry
            self._replay_entry = None
            return entry, None, None
        result = self._exchange_uncached(ops, packed, reception, forward, count)
        if forward and self._entry_cache is not None:
            self._entry_cache.save(self._cache_signature, result[0])
            # The backward context must not keep the entire invocation registry alive.
            self._entry_cache = self._cache_signature = None
        return result

    def _exchange_uncached(self, ops, packed, reception, forward, count):
        """Execute the existing state protocol, including every ordinary backward."""
        if self.is_rd:
            state_shape = (1, self.heads, 128, 128)
            maps = None if packed is None else packed.reshape(1, self.heads, 128, 256)
            result = self.executor.scan_affine(maps, self.plans[not forward], state_shape, reverse=not forward)
            return result, None, None
        if self.sparse is not None:
            maps, rank = self.sparse.exchange_maps(packed.reshape(1, self.heads, 128, 256), forward)
            maps = maps.squeeze(1)
            incoming = maps.new_zeros((self.heads, 128, 128))
            count = maps.shape[0] - 1
            if count:
                ops.merge_fwd_bwd_kernel(incoming, maps, count, rank, forward=forward)
            return incoming.unsqueeze(0), None, None
        if self.width == self.size:
            maps = self.executor.gather_packed(packed.reshape(1, *packed.shape[-3:]))
            incoming = maps.new_zeros((maps.shape[1], 128, 128))
            if count:
                ops.merge_fwd_bwd_kernel(incoming, maps, count, self.rank, forward=forward)
            return incoming.unsqueeze(0), None, None
        incoming, recv_handle = reception
        if self.width > 1:
            return self._grouped_exchange(packed, incoming, recv_handle, forward)
        return self._p2p_exchange(packed, incoming, recv_handle, forward)

    def _p2p_exchange(self, packed, incoming, recv_handle, forward):
        """Relay each head packet before queuing the next receive on the HCCL stream."""
        source = self.peer(forward, receiving=True)
        destination = self.peer(forward, receiving=False)
        transfer = self.info is None or self.info.num_sequences == 1
        states = matrices = None
        if packed is not None:
            # Packed S/M columns require copies for Torch matmul/add. Move those
            # copies before the state dependency, preserving FP32 arithmetic.
            states = packed[..., :128].reshape(1, self.heads, 128, 128).contiguous()
            if transfer and source is not None:
                matrices = packed[..., 128:].reshape(1, self.heads, 128, 128).contiguous()
        handles, payloads = [], []
        for index, (begin, end) in enumerate(self.head_ranges):
            if recv_handle is not None and transfer:
                recv_handle.wait()
                recv_handle = None
            if destination is not None:
                outgoing = _apply_state(
                    states[:, begin:end], None if matrices is None else matrices[:, begin:end],
                    incoming[:, begin:end] if matrices is not None else None)
                payloads.append(outgoing)
                handles.append(dist.isend(outgoing, dst=self.ranks[destination], group=self.group))
            if recv_handle is not None:
                recv_handle.wait()
            # Posting every receive first can serialize send_i behind recv_i+1
            # on the same communication stream, defeating packet relay.
            if source is not None and index + 1 < len(self.head_ranges):
                next_begin, next_end = self.head_ranges[index + 1]
                recv_handle = dist.irecv(incoming[:, next_begin:next_end], src=self.ranks[source], group=self.group)
        return incoming, NativeStateSends(handles) if handles else None, payloads

    def _grouped_exchange(self, packed, incoming, recv_handle, forward):
        """Gather native maps within a group and relay only its exit state."""
        if self.info is not None:
            if not self.info.transfer_active or not self.info.continues_next:
                packed[..., 128:].zero_()
            connected = self.info.continues_next if forward else self.info.continues_previous
            if not connected:
                packed[..., :128].zero_()
        maps = packed.new_empty((self.width, *packed.shape[-3:]))
        dist.all_gather_into_tensor(maps, packed.reshape(1, *packed.shape[-3:]), group=self.executor.intra)
        send_handle, outgoing, result = None, None, None
        indices = range(self.width) if forward else range(self.width - 1, -1, -1)
        prefix_count = self.local if forward else self.width - self.local - 1
        if self.rank == self.owner:
            if recv_handle is not None:
                recv_handle.wait()
            peer = self.peer(forward, receiving=False)
            if peer is not None:
                outgoing, result = self._scan(maps, incoming, indices, prefix_count)
                send_handle = dist.isend(outgoing, dst=self.ranks[peer], group=self.group)
        if self.peer(forward, receiving=True) is not None:
            dist.broadcast(incoming, src=self.ranks[self.owner], group=self.executor.intra)
        if result is None:
            local_indices = range(self.local) if forward else range(self.width - 1, self.local, -1)
            result, _ = self._scan(maps, incoming, local_indices)
        return result, send_handle, outgoing

    @staticmethod
    def _scan(maps, incoming, indices, prefix_count=None):
        """Keep native M/N orientation while applying a nonzero group entry."""
        state = incoming
        prefix = incoming if prefix_count == 0 else None
        for step, index in enumerate(indices, start=1):
            state = NativeStateExchange.apply(maps[index:index + 1], state)
            if step == prefix_count:
                prefix = state
        return state, prefix
