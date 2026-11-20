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
"""Native KDA state-CP scheduling over the functional local phase contract."""
from __future__ import annotations
from bisect import bisect_left
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import local_chunk_metadata

from hyper_parallel.components.functional.kimi_delta_attention import apply_kda_state_summary
from hyper_parallel.components.functional.kimi_delta_attention_ascendc import (
    load_kda_native_ops, prepare_kda_native,
    finish_kda_native_forward, prepare_kda_native_backward, finish_kda_native_backward,
)


@lru_cache(maxsize=1)
def _load_state_ops():
    """Check every native stage before entering the first state collective."""
    ops = load_kda_native_ops()
    required = (
        "pre_process_fwd_kernel_merged", "pre_process_bwd_kernel_merged",
        "merge_fwd_bwd_kernel",
    )
    missing = [name for name in required if not callable(getattr(ops, name, None))]
    if missing:
        raise RuntimeError(f"Native KDA state CP requires these fla-npu APIs: {', '.join(missing)}.")
    return ops


def _native_ag_intervals(info, tokens, rank, size):
    """Return tail/head windows and document-bounded exclusive scan lengths."""
    if info is None:
        return (0, tokens), (0, tokens), rank, size - rank - 1
    first_rank = (info.start - info.preceding_sample_tokens) // tokens
    last_rank = (info.global_offsets[bisect_left(info.global_offsets, info.end)] - 1) // tokens
    return ((info.tail.start, info.tail.stop), (info.head.start, info.head.stop),
            rank - first_rank if info.continues_previous else 0,
            last_rank - rank if info.continues_next else 0)


class _NativeStateExchange:
    """Bind one invocation's sequence plan to precreated communication groups.

    Only metadata survives forward. Receive buffers and send handles are owned
    by the active direction and drained before that invocation returns.
    """

    def __init__(self, mesh: Any, executor: Any, info: Any, local_length: int, heads: int) -> None:
        """Copy communication metadata and retain only this call's sequence plan."""
        self.executor, self.info, self.heads = executor, info, heads
        self.rank = mesh.get_local_rank()
        self.size = mesh.size()
        self.ranks = tuple(mesh.rank_list)
        self.group = mesh.get_group()
        self.width = getattr(executor, "width", self.size) if executor is not None else 1
        self.block, self.local = divmod(self.rank, self.width)
        self.owner = self.block * self.width + self.width - 1
        self.is_rd = getattr(executor, "protocol", None) == "recursive_doubling"
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
        if self.is_rd:
            plan = self.plans[not forward]
            return plan.shift_peers(plan.local_rank)[0] is not None
        return self.width > 1 or self.peer(forward, receiving=False) is not None

    def receive(self, reference: torch.Tensor, forward: bool) -> tuple[torch.Tensor | None, Any]:
        """Post reception before native prepare, independently of summary production.

        Args:
            reference: Head-major input supplying the head count and device.
            forward: Select the chronological communication direction.
        """
        if self.is_rd or self.width == self.size:
            return None, None
        incoming = reference.new_zeros((1, reference.shape[1], 128, 128), dtype=torch.float32)
        peer = self.peer(forward, receiving=True)
        handle = None
        if self.rank == self.owner and peer is not None:
            handle = dist.irecv(incoming, src=self.ranks[peer], group=self.group)
        return incoming, handle

    @staticmethod
    def apply(packed: torch.Tensor, incoming: torch.Tensor | None) -> torch.Tensor:
        """Apply forward M or native backward N directly, without transposing N.

        Args:
            packed: Packed state and transition tensors for the scan.
            incoming: State entering this rank or group.
        """
        return apply_kda_state_summary(packed[..., :128], packed[..., 128:], incoming)

    def exchange(self, ops: Any, packed: torch.Tensor | None,
                 reception: tuple[torch.Tensor | None, Any], forward: bool,
                 count: int) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        """Produce the local input state and retain an asynchronous outgoing payload.

        Args:
            ops: Native summary-merge operations for AllGather.
            packed: Local S/M or G/N, absent when no exit consumes it.
            reception: Previously posted receive buffer and handle.
            forward: Select the chronological communication direction.
            count: Number of maps in the relevant AllGather prefix or suffix.
        """
        if self.is_rd:
            state_shape = (1, self.heads, 128, 128)
            maps = None if packed is None else packed.reshape(1, self.heads, 128, 256)
            result = self.executor.scan_affine(maps, self.plans[not forward], state_shape, reverse=not forward)
            return result, None, None
        if self.width == self.size:
            maps = self.executor.gather_packed(packed.reshape(1, *packed.shape[-3:]))
            incoming = maps.new_zeros((maps.shape[1], 128, 128))
            if count:
                ops.merge_fwd_bwd_kernel(incoming, maps, count, self.rank, forward=forward)
            return incoming.unsqueeze(0), None, None
        incoming, recv_handle = reception
        if self.width > 1:
            return self._grouped_exchange(packed, incoming, recv_handle, forward)
        transfer = self.info is None or self.info.num_sequences == 1
        if recv_handle is not None and transfer:
            recv_handle.wait()
            recv_handle = None
        send_handle, outgoing = None, None
        peer = self.peer(forward, receiving=False)
        if peer is not None:
            outgoing = self.apply(packed, incoming if transfer else None).contiguous()
            send_handle = dist.isend(outgoing, dst=self.ranks[peer], group=self.group)
        # A reset permits sending the independent tail/head before this wait,
        # but local h/dhu still consumes the received first/last fragment state.
        if recv_handle is not None:
            recv_handle.wait()
        return incoming, send_handle, outgoing

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
            state = _NativeStateExchange.apply(maps[index:index + 1], state)
            if step == prefix_count:
                prefix = state
        return state, prefix


def _exchange_native_gradient(ctx, ops, inputs, reception):
    """Own G/N production, reset injection and the stock dHT dtype boundary."""
    if ctx.exchange is None:
        return None, None, None
    gk, qg, kg, weights, grad, dv_local_head = inputs
    maps = None
    if ctx.exchange.needs_summary(False):
        # Only the stock G/N ABI rounds gk; local dhu/finalize retain FP32.
        maps = ops.pre_process_bwd_kernel_merged(
            qg, kg, weights, grad, dv_local_head, ctx.scale, 64,
            gk=gk.to(qg.dtype), cu_seqlens=ctx.head)
    terminal, send_handle, outgoing = ctx.exchange.exchange(ops, maps, reception, False, ctx.suffix)
    if ctx.exchange.info is not None:
        terminal = ctx.exchange.info.last_state(terminal)
    # Communication and summary apply retain FP32; only stock dhu requires BF16.
    terminal = None if terminal is None else terminal.to(qg.dtype)
    return terminal, send_handle, outgoing


class _AscendCKDAState(torch.autograd.Function):
    """Share native local stages; state protocols optionally inject H and D."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                scale: float, lower_bound: float, metadata: dict,
                exchange: _NativeStateExchange | None) -> torch.Tensor:
        """Share prepare with S/M and local FwdH, then release gathered maps.

        Args:
            ctx: Per-invocation autograd context.
            query: Raw head-major queries.
            key: Raw head-major keys.
            value: Head-major values.
            gate: Sequence-major gate logits.
            beta: Sequence-major beta logits.
            a_log: FP32 decay parameters.
            dt_bias: FP32 gate bias.
            scale: Attention scale.
            lower_bound: Safe-gate lower bound.
            metadata: Native packed offsets and chunk indices.
            exchange: Optional state protocol for this invocation.
        """
        ops = _load_state_ops()
        inputs = (query, key, value, gate, beta, a_log, dt_bias)
        reception = None if exchange is None else exchange.receive(query, True)
        prepared = prepare_kda_native(inputs, scale, lower_bound, metadata)
        initial = send_handle = outgoing = head = suffix = None
        if exchange is not None:
            tail, head, prefix, suffix = _native_ag_intervals(
                exchange.info, query.shape[2], exchange.rank, exchange.size)
            maps = None
            if exchange.needs_summary(True):
                maps = ops.pre_process_fwd_kernel_merged(
                    prepared.kg, prepared.w, prepared.u, gk=prepared.gk, cu_seqlens=tail)
            initial, send_handle, outgoing = exchange.exchange(ops, maps, reception, True, prefix)
            del maps, reception
            if exchange.info is not None:
                initial = exchange.info.first_state(initial)
        output = finish_kda_native_forward(ctx, inputs, prepared, initial, scale, lower_bound, metadata)
        if send_handle is not None:
            send_handle.wait()
        del outgoing
        ctx.exchange, ctx.head, ctx.suffix = exchange, head, suffix
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Collect G/N, inject nonzero dHT and finish raw-input gradients.

        Args:
            ctx: Forward-owned activations and communication metadata.
            grad_output: Sequence-major output gradient.
        """
        ops = _load_state_ops()
        # Share one unpack across phases, including checkpoint recomputation.
        saved = ctx.saved_tensors
        gk, weights, qg, kg, query = saved[4], saved[7], saved[9], saved[10], saved[12]
        reception = ctx.exchange.receive(query, False)
        grad, d_aqk, dv_local_head, dq_raw = prepare_kda_native_backward(ctx, saved, grad_output)
        terminal, send_handle, outgoing = _exchange_native_gradient(
            ctx, ops, (gk, qg, kg, weights, grad, dv_local_head), reception)
        del reception
        gradients = finish_kda_native_backward(ctx, saved, grad, d_aqk, dv_local_head, dq_raw, terminal)
        if send_handle is not None:
            send_handle.wait()
        del outgoing
        return (*gradients, None, None, None, None)


def ascendc_state_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
    dt_bias: torch.Tensor, state_cp: Any, cp_mesh: Any, sequence_info: Any = None,
    scale: float | None = None, lower_bound: float = -5.0,
) -> torch.Tensor:
    """Run native KDA with P2P, AllGather, grouped AllGather/P2P, or RD.

    Args:
        query: Raw BF16 queries [1,T,H,128].
        key: Matching raw keys.
        value: BF16 values with the query shape.
        gate: Raw gate logits with the query shape.
        beta: Raw beta logits [1,T,H].
        a_log: Per-head FP32 decay parameters.
        dt_bias: FP32 gate bias [H,128].
        state_cp: Existing gather/RD executor, or None for ordinary P2P.
        cp_mesh: Ordered state mesh after any Ulysses redistribution.
        sequence_info: Packed interval metadata after any Ulysses redistribution.
        scale: Attention scale, default 1/sqrt(128).
        lower_bound: Safe gate lower bound.

    Returns:
        Output with the query shape. Uses the unmodified native stage APIs:
        backward summary gk and local dhu dHT are converted to model dtype.
        Prepared gates, communicated states and summary apply remain FP32.
        These ABI conversions introduce rounding relative to FP32-input kernels.
    """
    if query.device.type != "npu" or query.dtype != torch.bfloat16:
        raise ValueError("Native KDA state CP requires BF16 NPU inputs.")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("Native KDA state CP requires [1,T,H,128] inputs.")
    if any(tensor.shape != query.shape for tensor in (key, value, gate)) or beta.shape != query.shape[:-1]:
        raise ValueError("Native KDA state CP requires matching Q/K/V/gate heads and beta [1,T,H].")
    metadata = local_chunk_metadata(None if sequence_info is None else sequence_info.local_offsets, query.shape[1])
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    tokens = tuple(tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    exchange = _NativeStateExchange(cp_mesh, state_cp, sequence_info, query.shape[1], query.shape[2])
    return _AscendCKDAState.apply(
        *tokens, gate.contiguous(), beta.contiguous(), a_log.reshape(-1).clone(),
        dt_bias.reshape(-1, query.shape[-1]).clone(), scale, lower_bound, metadata, exchange)
