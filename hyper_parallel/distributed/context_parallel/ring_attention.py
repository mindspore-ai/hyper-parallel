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
Ulysses + Ring composes the framework's differentiable layout exchanges with
this kernel; the two-dimensional decomposition follows https://arxiv.org/abs/2405.07719.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
import math
from typing import Any, Optional

# Native Torch is required; the legacy lint rule predates the Torch-only migration.
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch.autograd.function import once_differentiable  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.context_parallel.collectives import (
    _build_hybrid_cp_submeshes, ulysses_head_to_seq, ulysses_seq_to_head,
)
from hyper_parallel.distributed.context_parallel.ring_layout import (
    RingAttentionTrace, redistribute_ring_qkv, redistribute_ring_rows, ring_task_plan,
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


def _ring_forward(query, key, value, ring, causal, scale, input_layout, overlap, trace=None):
    exchange = _RingExchange((key, value), ring)
    current_key, current_value = key, value
    state = None
    for step in range(ring.size):
        begin = trace.begin() if trace is not None else None
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
        if trace is not None:
            trace.end("forward", step, begin)
    return state[0].to(query.dtype), state[1], state[2]


def _ring_backward(saved, grad_output, ring, causal, scale, input_layout, overlap, trace=None):
    query, key, value, output, maximum, denominator = saved
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    kv_exchange = _RingExchange((key, value), ring)
    grad_exchange = _RingExchange((grad_key, grad_value), ring)
    current_key, current_value = key, value
    for step in range(ring.size):
        begin = trace.begin() if trace is not None else None
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
        if trace is not None:
            trace.end("backward", step, begin)
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


def _planned_forward(query, key, value, ring, scale, layout, overlap, plan, trace):
    exchange = _RingExchange((key, value), ring)
    state = _empty_state(query, layout)
    for step in range(ring.size):
        begin = trace.begin() if trace is not None else None
        if overlap and step + 1 < ring.size:
            exchange.launch((key, value))
        owner = (ring.rank - step) % ring.size
        _forward_blocks(query, key, value, state, plan.blocks[owner], scale, layout, plan.batches[owner])
        if step + 1 < ring.size:
            key, value = exchange.wait() if overlap else exchange.rotate((key, value))
        if trace is not None:
            trace.end("forward", step, begin)
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


def _planned_backward(saved, grad_output, ring, scale, layout, overlap, plan, trace):
    query, key, value = saved[:3]
    grad_query, grad_key, grad_value = (torch.zeros_like(tensor, dtype=torch.float32) for tensor in saved[:3])
    exchange = _RingExchange((key, value), ring)
    grad_exchange = _RingExchange((grad_key, grad_value), ring)
    for step in range(ring.size):
        begin = trace.begin() if trace is not None else None
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
        if trace is not None:
            trace.end("backward", step, begin)
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


class _RingAttention(torch.autograd.Function):
    """Pair raw P2P forward transport with an explicit gradient-return ring."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        ring: _RingContext, causal: bool, scale: float, input_layout: str, overlap: bool,
        plan: Any = None, trace: RingAttentionTrace | None = None,
    ) -> torch.Tensor:
        """Save only local inputs, the final output and global softmax statistics."""
        query, key, value = (tensor.contiguous() for tensor in (query, key, value))
        if plan is None:
            output, maximum, denominator = _ring_forward(
                query, key, value, ring, causal, scale, input_layout, overlap, trace,
            )
        else:
            output, maximum, denominator = _planned_forward(
                query, key, value, ring, scale, input_layout, overlap, plan, trace,
            )
        ctx.save_for_backward(query, key, value, output, maximum, denominator)
        ctx.ring = ring
        ctx.causal = causal
        ctx.scale = scale
        ctx.input_layout = input_layout
        ctx.overlap = overlap
        ctx.plan = plan
        ctx.trace = trace
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Sum local dQ and circulate dKV without a CP-size averaging factor."""
        if ctx.plan is None:
            gradients = _ring_backward(
                ctx.saved_tensors, grad_output.contiguous(), ctx.ring, ctx.causal, ctx.scale, ctx.input_layout,
                ctx.overlap, ctx.trace,
            )
        else:
            gradients = _planned_backward(ctx.saved_tensors, grad_output.contiguous(), ctx.ring, ctx.scale,
                                          ctx.input_layout, ctx.overlap, ctx.plan, ctx.trace)
        return (*gradients, *(None for _ in range(len(ctx.needs_input_grad) - 3)))


def _validate_inputs(query, key, value, causal, scale, input_layout, dropout_p, overlap):
    if not isinstance(overlap, bool):
        raise ValueError("overlap must be bool")
    if input_layout not in ("BNSD", "BSND", "TND"):
        raise ValueError("ring_attention supports only BNSD, BSND and TND layouts")
    if not isinstance(causal, bool):
        raise ValueError("causal must be bool")
    if dropout_p != 0:
        raise ValueError("ring_attention currently requires dropout_p=0")
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
    if scale is not None and (
        not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0
    ):
        raise ValueError("scale must be finite and positive")
    if query.device.type != "npu" or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("ring_attention currently requires all inputs on the same NPU")


def _prepare_task_plan(query, layout, size, rank, causal, mode, lengths, block_size, threshold, workers=None):
    tokens = query.shape[_token_axis(layout)] * size
    if layout == "TND":
        if (not isinstance(lengths, tuple) or not lengths
                or any(not isinstance(length, int) or isinstance(length, bool) or length < 1 for length in lengths)
                or sum(lengths) != tokens):
            raise ValueError("TND requires global document_lengths as a host tuple summing to global tokens")
    elif lengths is not None:
        raise ValueError("document_lengths requires TND layout")
    return ring_task_plan((tokens,) if lengths is None else lengths, size if workers is None else workers,
                          rank, mode, causal,
                          block_size, threshold)


def ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_mesh: Any = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    input_layout: str = "BNSD",
    dropout_p: float = 0.0,
    overlap: bool = False,
    load_balance: str = "none",
    document_lengths: tuple[int, ...] | None = None,
    balance_block_size: int | None = None,
    zigzag_min_length: int | None = None,
    trace: RingAttentionTrace | None = None,
) -> torch.Tensor:
    """Compute NPU MHA/GQA attention and restore original contiguous sequence ownership.

    Args:
        query: Local Q in BNSD/BSND or packed TND, with tokens equal to global tokens / CP.
        key: Compact local K; KV heads need not be divisible by CP degree.
        value: Local V with the same shape as K, including head dimension.
        cp_mesh: One-dimensional framework CP mesh, or None for single-rank execution.
        causal: Apply global causal order, with token shards ordered by CP group rank.
        scale: Positive score scale; defaults to the inverse square root of head dimension.
        input_layout: BNSD, BSND, or packed TND; shared by Q, K, V and output.
        dropout_p: Must be zero in this implementation.
        overlap: Prefetch KV and pipeline dKV communication with block computation.
        load_balance: Internal query/KV layout: none, zigzag, greedy or hybrid.
        document_lengths: Global positive document lengths, required for TND only.
            Supply identical host tuples on all ranks; total equals local tokens times CP.
        balance_block_size: Greedy block size, at most local tokens; default min(2048, local).
        zigzag_min_length: Explicit original-token threshold for hybrid.
        trace: Optional per-round NPU events; no barriers are inserted.

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
    _validate_inputs(query, key, value, causal, scale, input_layout, dropout_p, overlap)
    ring = _ring_context(cp_mesh)
    # Register optional NPU operators only when the NPU implementation is invoked.
    import torch_npu  # pylint: disable=C0415,unused-import

    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    plan = _prepare_task_plan(query, input_layout, ring.size, ring.rank, causal, load_balance,
                              document_lengths, balance_block_size, zigzag_min_length)
    if load_balance == "none" and input_layout != "TND":
        if trace is None:
            return _RingAttention.apply(query, key, value, ring, causal, scale, input_layout, overlap)
        return _RingAttention.apply(query, key, value, ring, causal, scale, input_layout, overlap, None, trace)
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
        output = _RingAttention.apply(query, key, value, ring, causal, scale, "TND", overlap, plan, trace)
        output = output.view(-1, batch, heads, output.shape[-1])
        output = output.permute(1, 2, 0, 3) if input_layout == "BNSD" else output.permute(1, 0, 2, 3)
    else:
        output = _RingAttention.apply(query, key, value, ring, causal, scale, input_layout, overlap, plan, trace)
    if load_balance != "none":
        output = redistribute_ring_rows(output, plan, ring.group, axis, inverse=True)
    return output


def _validate_ulysses_degree(cp_mesh, ulysses_degree):
    if isinstance(ulysses_degree, bool) or not isinstance(ulysses_degree, int):
        raise ValueError("ulysses_degree must be an integer")
    if cp_mesh is not None and cp_mesh.ndim != 1:
        raise ValueError("ulysses_ring_attention requires a one-dimensional CP mesh")
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    if not 1 <= ulysses_degree <= cp_size or cp_size % ulysses_degree:
        raise ValueError(
            f"ulysses_degree ({ulysses_degree}) must be a positive divisor of CP size ({cp_size})"
        )
    return cp_size


def ulysses_ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_mesh: Any = None,
    *,
    ulysses_degree: int = 1,
    causal: bool = False,
    scale: Optional[float] = None,
    input_layout: str = "BNSD",
    dropout_p: float = 0.0,
    overlap: bool = False,
    load_balance: str = "none",
    document_lengths: tuple[int, ...] | None = None,
    balance_block_size: int | None = None,
    zigzag_min_length: int | None = None,
    trace: RingAttentionTrace | None = None,
) -> torch.Tensor:
    """Compose Ulysses head sharding with NPU Ring Attention over sequence blocks.

    Args:
        query: Local Q in BNSD/BSND or TND, with sequence length global tokens / CP.
        key: Compact local K; both Q and KV head counts must be divisible by Ulysses degree.
        value: Local V with K's shape and dtype.
        cp_mesh: One-dimensional framework CP mesh in contiguous sequence-shard order.
            None selects single-rank execution and requires ulysses_degree=1.
        ulysses_degree: Positive divisor U of CP size C. Ring degree is R=C/U.
            U=1 selects pure Ring; U=C selects pure Ulysses.
        causal: Apply the causal mask in global sequence order.
        scale: Positive score scale; defaults to the inverse square root of head dimension.
        input_layout: BNSD, BSND, or packed TND, shared by inputs and output.
        dropout_p: Must be zero, as in ring_attention.
        overlap: Enable Ring KV/dKV overlap; Ulysses exchanges remain synchronous.
        load_balance: Ring subgroup task layout: none, zigzag, greedy or hybrid.
        document_lengths: Global document lengths for TND, as in ring_attention.
        balance_block_size: Greedy query block size, as in ring_attention.
        zigzag_min_length: Explicit document threshold for hybrid.
        trace: Optional Ring round device events, as in ring_attention.

    Returns:
        Output with the original local Q shape, dtype and sequence order.

    Raises:
        ValueError: If mesh, degree, local head counts or Ring input contracts are unsupported.

    Note:
        CP ranks are arranged as [R, U], with U varying fastest. At C=4, U=2,
        Ulysses groups are [0, 1] and [2, 3]; Ring groups are [0, 2] and [1, 3].
        All-to-all converts [B, H, S/C, D] to [B, H/U, S/R, D] before Ring,
        and the inverse exchange restores the output. KV heads are never repeated.
        Head counts are local counts after any TP sharding. All ranks must use
        matching shapes, options and requires_grad flags, and participate in backward
        even when their upstream gradient is zero. Gradients sum all local losses
        without CP-size averaging. Other limitations are shared with ring_attention.
    """
    cp_size = _validate_ulysses_degree(cp_mesh, ulysses_degree)
    _validate_inputs(query, key, value, causal, scale, input_layout, dropout_p, overlap)
    sequence_dim, head_dim = _token_axis(input_layout), 2 if input_layout == "BSND" else 1
    if any(tensor.shape[head_dim] % ulysses_degree for tensor in (query, key, value)):
        raise ValueError(
            f"local Q and KV head counts must be divisible by ulysses_degree ({ulysses_degree})"
        )
    # Validate metadata before Ulysses creates subgroups or exchanges tensors.
    _prepare_task_plan(query, input_layout, cp_size, 0, causal, load_balance,
                       document_lengths, balance_block_size, zigzag_min_length, workers=cp_size // ulysses_degree)
    options = {"load_balance": load_balance, "document_lengths": document_lengths,
               "balance_block_size": balance_block_size, "zigzag_min_length": zigzag_min_length, "trace": trace}
    # Validate parent rank order before submesh creation or any tensor exchange.
    _ring_context(cp_mesh)
    if ulysses_degree == 1:
        return ring_attention(
            query, key, value, cp_mesh, causal=causal, scale=scale, input_layout=input_layout,
            overlap=overlap, **options,
        )
    if ulysses_degree == cp_size:
        ulysses_mesh, ring_mesh = cp_mesh, None
    else:
        # The existing helper names the complementary axis "colossal"; here it carries Ring.
        ulysses_mesh, ring_mesh = _build_hybrid_cp_submeshes(cp_mesh, ulysses_degree)
    _ring_context(ulysses_mesh)
    _ring_context(ring_mesh)
    heads = tuple(tensor.shape[head_dim] // ulysses_degree for tensor in (query, key, value))
    # Group Q/K/V by destination head shard before the common sequence exchange.
    packed = torch.cat(tuple(tensor.unflatten(head_dim, (ulysses_degree, count))
                             for tensor, count in zip((query, key, value), heads)), dim=head_dim + 1)
    packed = packed.flatten(head_dim, head_dim + 1)
    packed = ulysses_seq_to_head(packed, sequence_dim, head_dim, ulysses_mesh)
    query, key, value = packed.split(heads, dim=head_dim)
    output = ring_attention(
        query, key, value, ring_mesh, causal=causal, scale=scale, input_layout=input_layout,
        overlap=overlap, **options,
    )
    return ulysses_head_to_seq(output, sequence_dim, head_dim, ulysses_mesh)
