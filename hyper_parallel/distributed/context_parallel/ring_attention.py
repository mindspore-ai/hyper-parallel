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
"""Document-aware Ring Attention with NPU fused block forward/backward.

The baseline waits for each exchange on the current stream. Optional overlap
prefetches the next KV block while computing the current one, and defers waiting
for incoming dKV until its next accumulation. KV and FP32 dKV use separate pairs
of reusable receive buffers; no remote KV survives forward.

Algorithm: https://arxiv.org/abs/2310.01889. The NPU block-kernel contract was
cross-checked against Yunchang's ring_npu_flash_attn.py:
https://github.com/feifeibear/long-context-attention/tree/main/yunchang/ring
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from typing import Any

# Native Torch is required; the legacy lint rule predates the Torch-only migration.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.ring_layout import (
    PrunedRingPlan, pruned_ring_plan, redistribute_ring_qkv, redistribute_ring_rows,
    ring_task_plan,
)
from hyper_parallel.distributed.context_parallel.ring_transport import (
    _PackedRingExchange, _communication_stream, _pack_ring_kv, _return_ring_gradient,
)


@dataclass(frozen=True)
class _RingContext:
    group: Any = None
    rank: int = 0
    size: int = 1
    next_rank: int = 0
    previous_rank: int = 0


def _ring_context(cp_mesh):
    if cp_mesh is None:
        return _RingContext()
    if cp_mesh.ndim != 1:
        raise ValueError("ring_attention requires a one-dimensional CP mesh")
    group = cp_mesh.get_group()
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if rank < 0 or size < 1:
        raise ValueError("the current rank must belong to the CP group")
    if size != cp_mesh.size() or rank != cp_mesh.get_local_rank():
        raise ValueError("CP mesh coordinates must match process-group rank order")
    if size == 1:
        return _RingContext(group, rank, size)
    return _RingContext(
        group, rank, size,
        dist.get_global_rank(group, (rank + 1) % size),
        dist.get_global_rank(group, (rank - 1) % size),
    )


class _RingExchange:
    """Rotate one packed payload with disjoint send/receive storage and bounded buffers."""

    def __init__(self, tensors: tuple[torch.Tensor, ...], ring: _RingContext) -> None:
        """Allocate two packed receive slots and persistent views of their components."""
        self._ring = ring
        sizes = tuple(tensor.numel() for tensor in tensors)
        self._buffers = tuple(tensors[0].new_empty(sum(sizes)) for _ in range(2)) if ring.size > 1 else ()
        self._views = tuple(tuple(part.view_as(tensor) for part, tensor in zip(buffer.split(sizes), tensors))
                            for buffer in self._buffers)
        self._slot = 0
        self._works = []
        self._sending = None
        self._send_buffer = None
        self._received = None

    def launch(self, tensors: tuple[torch.Tensor, ...]) -> None:
        """Post one send/receive pair and retain both source tensors and packed storage.

        HCCL records a dependency on the current stream. Once received, components
        remain views of packed storage, so later rounds forward that storage without
        repacking. KV and dKV exchanges retain independent buffer pairs.
        """
        if self._received is not None:
            raise RuntimeError("wait for the pending Ring exchange before launching another")
        self._sending = tensors
        self._received = self._views[self._slot] if self._ring.size > 1 else tensors
        if self._ring.size == 1:
            return
        previous = 1 - self._slot
        if all(tensor is view for tensor, view in zip(tensors, self._views[previous])):
            self._send_buffer = self._buffers[previous]
        else:
            self._send_buffer = torch.cat(tuple(tensor.reshape(-1) for tensor in tensors))
        buffer = self._buffers[self._slot]
        # Alternating submission order initializes matching peer links without
        # a cyclic dependency, including noncontiguous process-group ranks.
        if self._ring.rank % 2 == 0:
            send = dist.isend(self._send_buffer, dst=self._ring.next_rank, group=self._ring.group)
            receive = dist.irecv(buffer, src=self._ring.previous_rank, group=self._ring.group)
        else:
            receive = dist.irecv(buffer, src=self._ring.previous_rank, group=self._ring.group)
            send = dist.isend(self._send_buffer, dst=self._ring.next_rank, group=self._ring.group)
        self._works.extend((send, receive))

    def wait(self) -> tuple[torch.Tensor, ...]:
        """Order sends/receives complete before exposing views or reusing either slot.

        Launch and wait must use the same compute stream. Its ordering protects the
        previous consumer before buffer reuse, without a device-wide synchronization.
        """
        if self._received is None:
            raise RuntimeError("no pending Ring exchange to wait for")
        for work in self._works:
            work.wait()
        received = self._received
        self._works.clear()
        self._sending = None
        self._send_buffer = None
        self._received = None
        self._slot = 1 - self._slot
        return received

    def rotate(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Post and immediately wait for a rotation in the synchronous schedule."""
        self.launch(tensors)
        return self.wait()


@lru_cache(maxsize=8)
def _causal_mask(device):
    # Sparse mode 2 uses a fixed compressed mask, independent of global sequence length.
    return torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)


def _kernel_options(query, input_layout, diagonal_causal):
    return {
        "head_num": query.shape[2 if input_layout == "BSND" else 1],
        "input_layout": input_layout,
        "atten_mask": _causal_mask(query.device) if diagonal_causal else None,
        "sparse_mode": 2 if diagonal_causal else 0,
        "pre_tockens": 2147483647,
        "next_tockens": 0 if diagonal_causal else 2147483647,
        "keep_prob": 1.0,
    }


def _merge_blocks(state, block, input_layout):
    output, maximum, denominator = state
    block_output, block_maximum, block_denominator = block
    combined_maximum = torch.maximum(maximum, block_maximum)
    previous_mass = denominator * torch.exp(maximum - combined_maximum)
    block_mass = block_denominator * torch.exp(block_maximum - combined_maximum)
    combined_denominator = previous_mass + block_mass
    previous_weight = (previous_mass / combined_denominator)[..., :1]
    block_weight = (block_mass / combined_denominator)[..., :1]
    if input_layout == "BSND":
        previous_weight = previous_weight.transpose(1, 2)
        block_weight = block_weight.transpose(1, 2)
    return (
        output * previous_weight + block_output.float() * block_weight,
        combined_maximum,
        combined_denominator,
    )


def _ring_forward(query, key, value, ring, causal, scale, input_layout, overlap):
    exchange = _RingExchange((key, value), ring)
    current_key, current_value = key, value
    state = None
    for step in range(ring.size):
        if overlap and step + 1 < ring.size:
            exchange.launch((current_key, current_value))
        owner = (ring.rank - step) % ring.size
        if not causal or owner <= ring.rank:
            options = _kernel_options(query, input_layout, causal and owner == ring.rank)
            block = torch.ops.npu.npu_fusion_attention(
                query, current_key, current_value, scale=scale, **options,
            )[:3]
            if state is None:
                state = (block[0].float(), block[1], block[2])
            else:
                state = _merge_blocks(state, block, input_layout)
        if step + 1 < ring.size:
            current_key, current_value = (exchange.wait() if overlap
                                          else exchange.rotate((current_key, current_value)))
    return state[0].to(query.dtype), state[1], state[2]


def _ring_backward(saved, grad_output, ring, causal, scale, input_layout, overlap):
    query, key, value, output, maximum, denominator = saved
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    kv_exchange = _RingExchange((key, value), ring)
    grad_exchange = _RingExchange((grad_key, grad_value), ring)
    current_key, current_value = key, value
    for step in range(ring.size):
        if overlap and step + 1 < ring.size:
            kv_exchange.launch((current_key, current_value))
        owner = (ring.rank - step) % ring.size
        delta_key, delta_value = None, None
        if not causal or owner <= ring.rank:
            options = _kernel_options(query, input_layout, causal and owner == ring.rank)
            delta_query, delta_key, delta_value = torch.ops.npu.npu_fusion_attention_grad(
                query, current_key, current_value, grad_output,
                softmax_max=maximum, softmax_sum=denominator, attention_in=output,
                scale_value=scale, **options,
            )[:3]
            grad_query.add_(delta_query.float())
        # This block's derivatives do not depend on incoming dKV. Defer the wait
        # until accumulation so the preceding gradient rotation can overlap compute.
        if overlap and step > 0:
            grad_key, grad_value = grad_exchange.wait()
        if delta_key is not None:
            grad_key.add_(delta_key.float())
            grad_value.add_(delta_value.float())
        if not overlap and step + 1 < ring.size:
            current_key, current_value = kv_exchange.rotate((current_key, current_value))
        # All ranks communicate, including skipped future causal blocks. The last
        # gradient rotation completes the cycle and returns dKV to its original owner.
        if overlap:
            grad_exchange.launch((grad_key, grad_value))
            if step + 1 < ring.size:
                current_key, current_value = kv_exchange.wait()
            else:
                grad_key, grad_value = grad_exchange.wait()
        else:
            grad_key, grad_value = grad_exchange.rotate((grad_key, grad_value))
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def _token_axis(layout):
    return {"BNSD": 2, "BSND": 1, "TND": 0}[layout]


def _block_view(tensor, start, end, axis):
    return tensor.narrow(axis, start, end - start)


def _planned_options(query, key, layout, causal):
    options = _kernel_options(query, layout, causal)
    if causal:
        options["sparse_mode"] = 3
    if layout == "TND":
        options.update(head_num=query.shape[1], actual_seq_qlen=(query.shape[0],),
                       actual_seq_kvlen=(key.shape[0],))
    return options


def _empty_state(query, layout):
    shape = (*query.shape[:2], 8) if layout == "TND" else (
        query.shape[0], query.shape[1 if layout == "BNSD" else 2], query.shape[_token_axis(layout)], 8,
    )
    return (torch.zeros_like(query, dtype=torch.float32),
            query.new_full(shape, -float("inf"), dtype=torch.float32),
            query.new_zeros(shape, dtype=torch.float32))


def _row_span(bounds):
    if all(left[1] == right[0] for left, right in zip(bounds, bounds[1:])):
        return bounds[0][0], bounds[-1][1] - bounds[0][0]
    return None


def _take_rows(tensor, indices, span):
    return tensor.narrow(0, *span) if span is not None else tensor.index_select(0, indices)


def _update_rows(target, source, indices, span, *, add=False):
    if span is not None:
        view = target.narrow(0, *span)
        if add:
            view.add_(source)
        else:
            view.copy_(source)
    elif add:
        target.index_add_(0, indices, source)
    else:
        # Query rows are disjoint within a batch; indexed assignment avoids the
        # NPU scatter-copy kernel's expensive row-wise updates for wide heads.
        target.index_put_((indices,), source)


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _packed_indices(blocks, heads, device):
    queries = tuple(row for part in blocks for row in range(part.query_start, part.query_end))
    keys = tuple(row for part in blocks for row in range(part.key_start, part.key_end))
    qlengths = tuple(part.query_end - part.query_start for part in blocks)
    klengths = tuple(part.key_end - part.key_start for part in blocks)
    decode, encode, offset = [], [], 0
    # Native varlen statistics use N,T order separately inside each segment.
    for length in qlengths:
        decode.extend(offset + head * length + row for row in range(length) for head in range(heads))
        encode.extend(offset + row * heads + head for head in range(heads) for row in range(length))
        offset += length * heads
    tensors = tuple(torch.tensor(rows, device=device, dtype=torch.long) for rows in (queries, keys, decode, encode))
    qspan = _row_span(tuple((part.query_start, part.query_end) for part in blocks))
    kspan = _row_span(tuple((part.key_start, part.key_end) for part in blocks))
    return (*tensors, tuple(accumulate(qlengths)), tuple(accumulate(klengths)), qspan, kspan)


def _packed_options(query, key, blocks, ends):
    options = _planned_options(query, key, "TND", blocks[0].causal)
    options.update(actual_seq_qlen=ends[0], actual_seq_kvlen=ends[1])
    return options


def _forward_packed(query, key, value, state, blocks, scale):
    rows, keys, decode, _, qends, kends, qspan, kspan = _packed_indices(blocks, query.shape[1], query.device)
    q_block = _take_rows(query, rows, qspan)
    k_block, v_block = (_take_rows(tensor, keys, kspan) for tensor in (key, value))
    output, maximum, denominator = torch.ops.npu.npu_fusion_attention(
        q_block, k_block, v_block, scale=scale, **_packed_options(q_block, k_block, blocks, (qends, kends)),
    )[:3]
    maximum, denominator = (stat.reshape(-1, 8).index_select(0, decode).view(-1, query.shape[1], 8)
                            for stat in (maximum, denominator))
    previous = tuple(_take_rows(tensor, rows, qspan) for tensor in state)
    merged = _merge_blocks(previous, (output, maximum, denominator), "TND")
    for target, updated in zip(state, merged):
        _update_rows(target, updated, rows, qspan)


def _backward_packed(saved, grad_output, key, value, gradients, blocks, scale):
    query, _, _, output, maximum, denominator = saved
    rows, keys, _, encode, qends, kends, qspan, kspan = _packed_indices(blocks, query.shape[1], query.device)
    q_block, out, dout = (_take_rows(tensor, rows, qspan) for tensor in (query, output, grad_output))
    max_block, sum_block = (_take_rows(stat, rows, qspan).reshape(-1, 8).index_select(0, encode)
                            .view(-1, query.shape[1], 8) for stat in (maximum, denominator))
    k_block, v_block = (_take_rows(tensor, keys, kspan) for tensor in (key, value))
    values = torch.ops.npu.npu_fusion_attention_grad(
        q_block, k_block, v_block, dout, softmax_max=max_block, softmax_sum=sum_block, attention_in=out,
        scale_value=scale, **_packed_options(q_block, k_block, blocks, (qends, kends)),
    )[:3]
    for target, delta, indices, span in zip(gradients, values, (rows, keys, keys), (qspan, kspan, kspan)):
        _update_rows(target, delta.float(), indices, span, add=True)


def _forward_blocks(query, key, value, state, blocks, scale, layout, batches):
    if layout == "TND" and len(blocks) > 1:
        for batch in batches:
            _forward_packed(query, key, value, state, batch, scale)
        return
    axis, stats_axis = _token_axis(layout), 0 if layout == "TND" else 2
    for part in blocks:
        q_block = _block_view(query, part.query_start, part.query_end, axis).contiguous()
        k_block = _block_view(key, part.key_start, part.key_end, axis).contiguous()
        v_block = _block_view(value, part.key_start, part.key_end, axis).contiguous()
        block = torch.ops.npu.npu_fusion_attention(
            q_block, k_block, v_block, scale=scale, **_planned_options(q_block, k_block, layout, part.causal),
        )[:3]
        if layout == "TND":
            # NPU exposes [T,N,8] metadata but stores each single-segment statistic
            # in N,T order. Decode before row-wise merging; backward re-encodes it.
            block = (block[0], *(stat.reshape(q_block.shape[1], q_block.shape[0], 8).transpose(0, 1)
                                 for stat in block[1:]))
        previous = tuple(_block_view(tensor, part.query_start, part.query_end, dim)
                         for tensor, dim in zip(state, (axis, stats_axis, stats_axis)))
        merged = _merge_blocks(previous, block, layout)
        for target, source in zip(previous, merged):
            target.copy_(source)


def _planned_forward(query, key, value, ring, scale, layout, overlap, plan):
    exchange = _RingExchange((key, value), ring)
    state = _empty_state(query, layout)
    for step in range(ring.size):
        if overlap and step + 1 < ring.size:
            exchange.launch((key, value))
        owner = (ring.rank - step) % ring.size
        _forward_blocks(query, key, value, state, plan.blocks[owner], scale, layout, plan.batches[owner])
        if step + 1 < ring.size:
            key, value = exchange.wait() if overlap else exchange.rotate((key, value))
    return state[0].to(query.dtype), state[1], state[2]


def _backward_blocks(saved, grad_output, key, value, grad_query, blocks, scale, layout, batches):
    query, _, _, output, maximum, denominator = saved
    axis, stats_axis = _token_axis(layout), 0 if layout == "TND" else 2
    delta_key, delta_value = torch.zeros_like(key, dtype=torch.float32), torch.zeros_like(value, dtype=torch.float32)
    if layout == "TND" and len(blocks) > 1:
        for batch in batches:
            _backward_packed(saved, grad_output, key, value, (grad_query, delta_key, delta_value), batch, scale)
        return delta_key, delta_value
    for part in blocks:
        q_block, out, dout = (_block_view(tensor, part.query_start, part.query_end, axis).contiguous()
                             for tensor in (query, output, grad_output))
        max_block, sum_block = (_block_view(tensor, part.query_start, part.query_end, stats_axis).contiguous()
                               for tensor in (maximum, denominator))
        if layout == "TND":
            max_block, sum_block = (stat.transpose(0, 1).contiguous().reshape(q_block.shape[0], q_block.shape[1], 8)
                                    for stat in (max_block, sum_block))
        k_block, v_block = (_block_view(tensor, part.key_start, part.key_end, axis).contiguous()
                            for tensor in (key, value))
        dq, dk, dv = torch.ops.npu.npu_fusion_attention_grad(
            q_block, k_block, v_block, dout, softmax_max=max_block, softmax_sum=sum_block,
            attention_in=out, scale_value=scale, **_planned_options(q_block, k_block, layout, part.causal),
        )[:3]
        _block_view(grad_query, part.query_start, part.query_end, axis).add_(dq.float())
        _block_view(delta_key, part.key_start, part.key_end, axis).add_(dk.float())
        _block_view(delta_value, part.key_start, part.key_end, axis).add_(dv.float())
    return delta_key, delta_value


def _planned_backward(saved, grad_output, ring, scale, layout, overlap, plan):
    query, key, value = saved[:3]
    grad_query, grad_key, grad_value = (torch.zeros_like(tensor, dtype=torch.float32) for tensor in saved[:3])
    exchange = _RingExchange((key, value), ring)
    grad_exchange = _RingExchange((grad_key, grad_value), ring)
    for step in range(ring.size):
        if overlap and step + 1 < ring.size:
            exchange.launch((key, value))
        owner = (ring.rank - step) % ring.size
        delta_key, delta_value = _backward_blocks(saved, grad_output, key, value, grad_query,
                                                 plan.blocks[owner], scale, layout, plan.batches[owner])
        if overlap and step:
            grad_key, grad_value = grad_exchange.wait()
        grad_key.add_(delta_key)
        grad_value.add_(delta_value)
        if not overlap and step + 1 < ring.size:
            key, value = exchange.rotate((key, value))
        if overlap:
            grad_exchange.launch((grad_key, grad_value))
            if step + 1 < ring.size:
                key, value = exchange.wait()
            else:
                grad_key, grad_value = grad_exchange.wait()
        else:
            grad_key, grad_value = grad_exchange.rotate((grad_key, grad_value))
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def _launch_pruned(exchange, packet, overlap, step):
    if not overlap:
        exchange.stream.wait_stream(torch.npu.current_stream())
    received = exchange.launch(packet, step)
    if not overlap:
        received.ready()
    return received


def _pruned_forward(query, key, value, ring, scale, overlap, plan):
    initial = _pack_ring_kv((key, value), plan.source_spans, ring, plan.counts)
    exchange = _PackedRingExchange(key, plan, ring, _communication_stream(query.device, False))
    state, packet = _empty_state(query, "TND"), initial
    for step in range(max(plan.hops) + 1):
        following = exchange.launch(packet, step) if overlap and step < max(plan.hops) else None
        blocks, batches = (plan.local_blocks, plan.local_batches) if step == 0 else (
            plan.blocks[packet.owner], plan.batches[packet.owner])
        if blocks:
            packet.ready()
            keys, values = (key, value) if step == 0 else packet.tensors()
            _forward_blocks(query, keys, values, state, blocks, scale, "TND", batches)
            packet.consumed_now()
        if not overlap and step < max(plan.hops):
            following = _launch_pruned(exchange, packet, False, step)
        packet = following
        if step == max(plan.hops):
            exchange.finish(initial)
    return state[0].to(query.dtype), state[1], state[2]


def _pruned_delta(saved, grad_output, grad_query, packet, plan, scale, step):
    blocks, batches = (plan.local_blocks, plan.local_batches) if step == 0 else (
        plan.blocks[packet.owner], plan.batches[packet.owner])
    if not blocks:
        return None
    packet.ready()
    keys, values = saved[1:3] if step == 0 else packet.tensors()
    delta = _backward_blocks(saved, grad_output, keys, values, grad_query, blocks, scale, "TND", batches)
    packet.consumed_now()
    return delta


def _pruned_accumulate(gradient, delta, local_gradients, ring, plan, step):
    if delta is None:
        return
    gradient.ready()
    if step == 0:
        for target, source in zip(local_gradients, delta):
            target.add_(source)
        _pack_ring_kv(local_gradients, plan.source_spans, ring, plan.counts, out=gradient)
        return
    for target, source in zip(gradient.tensors(), delta):
        target.add_(source)
    gradient.produced_now()


def _restore_pruned_gradient(local_gradients, gradient, ring, plan):
    gradient.ready()
    for target, source in zip(local_gradients, gradient.tensors()):
        offset = 0
        for start, end in plan.source_spans[ring.rank]:
            length = end - start
            target[start:end].copy_(source[offset:offset + length])
            offset += length
    gradient.consumed_now()


def _pruned_backward(saved, grad_output, ring, scale, overlap, plan):
    query, key, value = saved[:3]
    grad_query, grad_key, grad_value = (torch.zeros_like(tensor, dtype=torch.float32) for tensor in saved[:3])
    initial = _pack_ring_kv((key, value), plan.source_spans, ring, plan.counts)
    grad_initial = _pack_ring_kv((grad_key, grad_value), plan.source_spans, ring, plan.counts)
    exchange = _PackedRingExchange(key, plan, ring, _communication_stream(query.device, False))
    grad_exchange = _PackedRingExchange(grad_key, plan, ring, _communication_stream(query.device, True))
    packet, gradient = initial, grad_initial
    for step in range(max(plan.hops) + 1):
        following = exchange.launch(packet, step) if overlap and step < max(plan.hops) else None
        delta = _pruned_delta(saved, grad_output, grad_query, packet, plan, scale, step)
        _pruned_accumulate(gradient, delta, (grad_key, grad_value), ring, plan, step)
        if not overlap and step < max(plan.hops):
            following = _launch_pruned(exchange, packet, False, step)
        next_gradient = _launch_pruned(grad_exchange, gradient, overlap, step) if step < max(plan.hops) else None
        if not overlap:
            grad_exchange.stream.wait_stream(torch.npu.current_stream())
        _return_ring_gradient(gradient, grad_initial, plan, ring, grad_exchange.stream, step)
        packet, gradient = following, next_gradient
        if step == max(plan.hops):
            _restore_pruned_gradient((grad_key, grad_value), grad_initial, ring, plan)
            exchange.finish(initial)
            grad_exchange.finish(grad_initial)
    return tuple(grad.to(tensor.dtype) for grad, tensor in zip((grad_query, grad_key, grad_value), saved[:3]))


class _RingAttention(torch.autograd.Function):
    """Pair raw P2P forward transport with an explicit gradient-return ring."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        ring: _RingContext, causal: bool, scale: float, input_layout: str, overlap: bool,
        plan: Any = None,
    ) -> torch.Tensor:
        """Save only local inputs, the final output and global softmax statistics."""
        query, key, value = (tensor.contiguous() for tensor in (query, key, value))
        if isinstance(plan, PrunedRingPlan):
            output, maximum, denominator = _pruned_forward(query, key, value, ring, scale, overlap, plan)
        elif plan is None:
            output, maximum, denominator = _ring_forward(
                query, key, value, ring, causal, scale, input_layout, overlap,
            )
        else:
            output, maximum, denominator = _planned_forward(
                query, key, value, ring, scale, input_layout, overlap, plan,
            )
        ctx.save_for_backward(query, key, value, output, maximum, denominator)
        ctx.ring = ring
        ctx.causal = causal
        ctx.scale = scale
        ctx.input_layout = input_layout
        ctx.overlap = overlap
        ctx.plan = plan
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Sum local dQ and circulate dKV without a CP-size averaging factor."""
        if isinstance(ctx.plan, PrunedRingPlan):
            gradients = _pruned_backward(ctx.saved_tensors, grad_output.contiguous(), ctx.ring, ctx.scale,
                                         ctx.overlap, ctx.plan)
        elif ctx.plan is None:
            gradients = _ring_backward(
                ctx.saved_tensors, grad_output.contiguous(), ctx.ring, ctx.causal, ctx.scale, ctx.input_layout,
                ctx.overlap,
            )
        else:
            gradients = _planned_backward(ctx.saved_tensors, grad_output.contiguous(), ctx.ring, ctx.scale,
                                          ctx.input_layout, ctx.overlap, ctx.plan)
        return (*gradients, *(None for _ in range(len(ctx.needs_input_grad) - 3)))


def _validate_inputs(query, key, value, causal, input_layout, overlap):
    if not isinstance(overlap, bool):
        raise ValueError("overlap must be bool")
    if input_layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("ring_attention supports only BNSD, BSND and TND layouts")
    if not isinstance(causal, bool):
        raise ValueError("causal must be bool")
    tensors = (query, key, value)
    dimensions = 3 if input_layout == "TND" else 4
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != dimensions for tensor in tensors):
        raise ValueError("query, key and value must be three-dimensional TND or four-dimensional BNSD/BSND tensors")
    if any(hasattr(tensor, "to_local") for tensor in tensors):
        raise ValueError("ring_attention requires local tensors; convert DTensors before calling")
    if any(dimension < 1 for tensor in tensors for dimension in tensor.shape):
        raise ValueError("query, key and value dimensions must be nonempty")
    head_dim = 2 if input_layout == "BSND" else 1
    sequence_dim = _token_axis(input_layout)
    if key.shape != value.shape:
        raise ValueError("key and value must have equal shapes in the initial MHA/GQA implementation")
    equal_axes = (0, 2) if input_layout == "TND" else (0, sequence_dim, 3)
    if any(query.shape[axis] != key.shape[axis] for axis in equal_axes):
        raise ValueError("query and key must have equal batch, local sequence and head dimensions")
    if query.shape[head_dim] % key.shape[head_dim]:
        raise ValueError("the query head count must be divisible by the KV head count")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(tensor.dtype != query.dtype for tensor in tensors):
        raise ValueError("query, key and value must share FP16 or BF16 dtype")
    if query.device.type != "npu" or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("ring_attention currently requires all inputs on the same NPU")


def _prepare_task_plan(query, layout, size, rank, causal, load_balance, lengths,
                       document_subrings=False):
    tokens = query.shape[_token_axis(layout)] * size
    if layout == "TND":
        if (not isinstance(lengths, tuple) or not lengths
                or any(not isinstance(length, int) or isinstance(length, bool) or length < 1 for length in lengths)
                or sum(lengths) != tokens):
            raise ValueError("TND requires global document_lengths as a host tuple summing to global tokens")
    elif lengths is not None:
        raise ValueError("document_lengths requires TND layout")
    return ring_task_plan((tokens,) if lengths is None else lengths, size, rank, load_balance, causal,
                          document_subrings=document_subrings)


def _validate_pruning(prune_documents):
    if not isinstance(prune_documents, bool):
        raise ValueError("prune_documents must be bool")


def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_mesh: Any = None,
    *,
    causal: bool = False,
    input_layout: str = "BNSD",
    overlap: bool = False,
    load_balance: str = "none",
    prune_documents: bool = True,
    document_lengths: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute NPU MHA/GQA attention and restore original contiguous sequence ownership.

    Args:
        query: Local Q in BNSD/BSND or packed TND, with tokens equal to global tokens / CP.
        key: Compact local K; KV heads need not be divisible by CP degree.
        value: Local V with the same shape as K, including head dimension.
        cp_mesh: One-dimensional framework CP mesh, or None for single-rank execution.
        causal: Apply global causal order, with token shards ordered by CP group rank.
        input_layout: BNSD, BSND, or packed TND; shared by Q, K, V and output.
        overlap: Prefetch KV and pipeline dKV communication with block computation.
        load_balance: Query layout, either contiguous ``none`` or ``zigzag``.
        prune_documents: Build per-document Ring subplans; can be combined with load_balance.
        document_lengths: Global positive document lengths, required for TND only.
            Supply identical host tuples on all ranks; total equals local tokens times CP.

    Returns:
        Attention output for this rank's Q shard, with Q's shape and dtype.

    Raises:
        ValueError: If the tensor, mesh, dtype or attention options are unsupported.

    Note:
        All ranks must use identical shapes, dtypes and options and enter forward
        and backward in the same order, including ranks with zero upstream gradient.
        Gradients represent the sum of the group ranks' local losses; no CP-size
        division is applied. TND documents are isolated; padding must be removed
        before calling. Explicit masks, higher-order gradients, and unequal source
        shards are unsupported. Internal task sizes may differ; communication pads
        to a common capacity and kernels operate only on real rows. Packed attention
        batches disjoint query intersections in native TND calls with bounded KV packing. Native backward
        may have atomic reduction roundoff; exact repeatability requires
        torch.use_deterministic_algorithms(True).
    """
    _validate_pruning(prune_documents)
    if load_balance not in ("none", "zigzag"):
        raise ValueError("load_balance must be none or zigzag")
    _validate_inputs(query, key, value, causal, input_layout, overlap)
    ring = _ring_context(cp_mesh)
    # Register optional NPU operators only when the NPU implementation is invoked.
    import torch_npu  # pylint: disable=C0415,unused-import

    scale = query.shape[-1] ** -0.5
    prune_documents = prune_documents and input_layout == "TND"
    plan = _prepare_task_plan(query, input_layout, ring.size, ring.rank, causal, load_balance,
                              document_lengths, document_subrings=prune_documents)
    if prune_documents:
        if load_balance != "none":
            query, key, value = redistribute_ring_qkv(
                query, key, value, plan, ring.group, _token_axis(input_layout),
                2 if input_layout == "BSND" else 1,
            )
        pruning_plan = pruned_ring_plan(
            document_lengths, ring.size, ring.rank, causal, plan.assignments,
        )
        output = _RingAttention.apply(
            query, key, value, ring, causal, scale, "TND", overlap, pruning_plan,
        )
        if load_balance != "none":
            output = redistribute_ring_rows(output, plan, ring.group, _token_axis(input_layout), inverse=True)
        return output
    if load_balance == "none" and input_layout != "TND":
        return _RingAttention.apply(query, key, value, ring, causal, scale, input_layout, overlap)
    axis = _token_axis(input_layout)
    if load_balance != "none":
        head_dim = 2 if input_layout == "BSND" else 1
        query, key, value = redistribute_ring_qkv(query, key, value, plan, ring.group, axis, head_dim)
    if input_layout != "TND" and not plan.fast_zigzag:
        batch, heads = query.shape[0], query.shape[head_dim]
        # Dense samples share the same mask. Folding independent batches into
        # heads preserves GQA groups and avoids duplicating the document plan.
        query, key, value = (tensor.permute(axis, 0, head_dim, 3).flatten(1, 2)
                             for tensor in (query, key, value))
        output = _RingAttention.apply(query, key, value, ring, causal, scale, "TND", overlap, plan)
        output = output.view(-1, batch, heads, output.shape[-1])
        output = output.permute(1, 2, 0, 3) if input_layout == "BNSD" else output.permute(1, 0, 2, 3)
    else:
        output = _RingAttention.apply(query, key, value, ring, causal, scale, input_layout, overlap, plan)
    if load_balance != "none":
        output = redistribute_ring_rows(output, plan, ring.group, axis, inverse=True)
    return output

