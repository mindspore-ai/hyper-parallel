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
"""Serial query stages sharing one physical KV projection and transport.

Only integer row/column plans are cached. Autograd saves the original inputs
and parameters, and native FA statistics; backward replays projections only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from hyper_parallel.components.modules._head_chunk_ops import (
    _FA_SAVED_TENSOR_COUNT,
    _apply_rope,
    _fusion_attention_backward_saved,
    _fusion_attention_forward_saved,
    _head_to_seq,
    _rms_norm,
    _seq_to_head,
)

from hyper_parallel.components.modules._head_chunk_saved_stats import SavedStatsOffloadStore, _OffloadStatistics
from hyper_parallel.components.modules._head_chunk_transport import HeadChunkTransport
from hyper_parallel.core.activation_memory import PinnedMemoryPool


def shared_kv_head_schedule(
    query_heads: int, kv_heads: int, stage_heads: int, cp_size: int,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    """Plan destination-major KV groups and their disjoint query stages.

    Args:
        query_heads: Total query-head count.
        kv_heads: Physical key/value-head count, without replication.
        stage_heads: Maximum global query heads processed by one stage.
        cp_size: Number of ranks in the one-dimensional Ulysses mesh.

    Returns:
        Tuples of physical KV indices and the query-head tuples using them.
        Tail stages are allowed; every stage remains equally shardable by CP.

    Raises:
        ValueError: If counts are not positive integers or cannot support
            native GQA with equal-split Ulysses.
    """
    counts = (query_heads, kv_heads, stage_heads, cp_size)
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts):
        raise ValueError("head counts and CP size must be positive integers")
    if query_heads % kv_heads or kv_heads % cp_size or stage_heads % cp_size:
        raise ValueError("shared KV requires Q % KV = KV % CP = stage % CP = 0")
    if stage_heads > query_heads:
        raise ValueError("head_chunk_size cannot exceed the total query-head count")
    ratio = query_heads // kv_heads
    capacity = stage_heads // cp_size
    block_count = kv_heads // cp_size
    blocks_per_group = max(1, capacity // ratio)
    groups = []
    for begin in range(0, block_count, blocks_per_group):
        blocks = range(begin, min(begin + blocks_per_group, block_count))
        keys = tuple(block * cp_size + rank for rank in range(cp_size) for block in blocks)
        stages = []
        for start in range(0, ratio, min(capacity, ratio)):
            offsets = range(start, min(start + capacity, ratio))
            stages.append(tuple(
                (block * cp_size + rank) * ratio + offset
                for rank in range(cp_size) for block in blocks for offset in offsets
            ))
        groups.append((keys, tuple(stages)))
    return tuple(groups)


@dataclass(frozen=True)
class _QuerySpec:
    """One query stage in the master's interleaved packed-weight layout."""

    index: int
    count: int
    rows: torch.Tensor
    columns: torch.Tensor
    column_spans: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _KVSpec:
    """Physical key/value heads reused by the query stages in this group."""

    count: int
    rows: torch.Tensor
    stages: tuple[_QuerySpec, ...]


def _device_plan(owner: Any, device: torch.device) -> tuple[_KVSpec, ...]:
    """Cache only immutable indices; never cache parameter slices or activations."""
    cp_size = 1 if owner.head_chunk_cp_mesh is None else owner.head_chunk_cp_mesh.size()
    cache_key = (cp_size, str(device))
    cached = owner._shared_kv_plans.get(cache_key)
    if cached is not None:
        return cached
    dim, ratio, width = owner.qk_head_dim, owner.num_key_value_groups, owner.qkv_group_width
    result = []
    stage_index = 0
    for keys, queries in shared_kv_head_schedule(
        owner.num_heads, owner.num_key_value_heads, owner.head_chunk_size, cp_size,
    ):
        kv_rows = [
            key * width + (ratio + component) * dim + offset
            for component in range(2) for key in keys for offset in range(dim)
        ]
        stages = []
        for heads in queries:
            rows = [(head // ratio) * width + (head % ratio) * dim + offset
                    for head in heads for offset in range(dim)]
            columns = [head * dim + offset for head in heads for offset in range(dim)]
            spans = []
            for source_head, target_head in enumerate(heads):
                if spans and spans[-1][0] + spans[-1][2] == target_head * dim:
                    begin, source, span_width = spans[-1]
                    spans[-1] = (begin, source, span_width + dim)
                else:
                    spans.append((target_head * dim, source_head * dim, dim))
            stages.append(_QuerySpec(
                stage_index, len(heads), torch.tensor(rows, dtype=torch.int64, device=device),
                torch.tensor(columns, dtype=torch.int64, device=device), tuple(spans),
            ))
            stage_index += 1
        result.append(_KVSpec(len(keys), torch.tensor(kv_rows, dtype=torch.int64, device=device), tuple(stages)))
    owner._shared_kv_plans[cache_key] = tuple(result)
    return tuple(result)


@dataclass(frozen=True)
class _Execution:
    """Forward-local metadata, without references to modules or model parameters."""

    plan: tuple[_KVSpec, ...]
    head_dim: int
    query_norm_eps: float
    key_norm_eps: float
    query_norm_kind: str
    key_norm_kind: str
    scaling: float
    cp_mesh: Any | None
    retain_for_backward: bool = True
    offload_pool: PinnedMemoryPool | None = None
    prefetch_slots: int = 2
    offload_statistics: _OffloadStatistics | None = None
    pipeline_output_a2a: bool = False
    pipeline_dq_a2a: bool = False
    early_saved_stats: bool = False
    trim_a2a_stream_records: bool = False
    retire_d2h_sources: bool = False
    release_a2a_source: bool = False


@dataclass
class _Projection:
    """Stage-local norm/RoPE graph rooted at detached raw projection and scale."""

    raw: torch.Tensor
    norm: torch.Tensor
    outputs: tuple[torch.Tensor, ...]


def _project(
    execution: _Execution, spec: _QuerySpec | _KVSpec, hidden: torch.Tensor,
    weight: torch.Tensor, norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    *, kv: bool, replay: bool,
) -> _Projection:
    raw = F.linear(hidden, weight.index_select(0, spec.rows)).detach().requires_grad_(replay)
    norm_leaf = norm.detach().requires_grad_(replay)
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[0] == 1 and hidden.shape[0] > 1:
        cos, sin = cos.expand(hidden.shape[0], -1, -1), sin.expand(hidden.shape[0], -1, -1)
    with torch.set_grad_enabled(replay):
        if kv:
            key, value = raw.view(*hidden.shape[:2], 2, spec.count, execution.head_dim).unbind(2)
            normalized = _rms_norm(key, norm_leaf, execution.key_norm_eps, execution.key_norm_kind)
        else:
            query = raw.view(*hidden.shape[:2], spec.count, execution.head_dim)
            normalized = _rms_norm(query, norm_leaf, execution.query_norm_eps, execution.query_norm_kind)
        rotated = _apply_rope(normalized, cos.unsqueeze(2), sin.unsqueeze(2)).transpose(1, 2).contiguous()
        outputs = (rotated, value.transpose(1, 2).contiguous()) if kv else (rotated,)
    return _Projection(raw, norm_leaf, outputs)


def _projection_vjp(
    state: _Projection, gradients: tuple[torch.Tensor, ...], hidden: torch.Tensor,
    weight: torch.Tensor, rows: torch.Tensor, grad_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay only the local norm/RoPE VJP; form projection gradients explicitly."""
    grad_raw, grad_norm = torch.autograd.grad(state.outputs, (state.raw, state.norm), gradients)
    flat_raw = grad_raw.reshape(-1, grad_raw.shape[-1])
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    grad_weight = flat_raw.transpose(0, 1).mm(flat_hidden)
    # Retain the original low-precision GEMM and sequential dHidden addition.
    grad_hidden.view_as(flat_hidden).add_(flat_raw.mm(weight.index_select(0, rows)))
    return grad_norm, grad_weight


def _assemble_packed_gradient(weight: torch.Tensor, blocks: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Materialize the full return tensor only after all FA workspaces are gone."""
    result = torch.empty_like(weight)
    while blocks:
        rows, block = blocks.pop()
        result.index_copy_(0, rows, block)
    return result


def _write_output_gradient(destination: torch.Tensor, source: torch.Tensor, spec: _QuerySpec) -> None:
    """Write disjoint spans without the NPU's column-scatter slow path."""
    for target_start, source_start, width in spec.column_spans:
        destination.narrow(1, target_start, width).copy_(source.narrow(1, source_start, width))


def _retain_stage(
    execution: _Execution, spec: _QuerySpec, saved: tuple[torch.Tensor, ...],
    retained: list[torch.Tensor], store: SavedStatsOffloadStore | None,
) -> None:
    """Save one stage in order, at the explicitly selected offload boundary."""
    if not execution.retain_for_backward:
        return
    if store is None:
        retained.extend(saved)
    else:
        store.offload_stage(spec.index, saved[:3])
        retained.extend(saved[3:])


def _next_saved_ticket(store: SavedStatsOffloadStore | None, index: int, count: int) -> Any:
    """Look ahead one stage only when the other physical slot is available."""
    if store is not None and store.slots == 2 and index + 1 < count:
        return store.prefetch_stage(index + 1)
    return None


@dataclass
class _ForwardPipeline:
    """Keep at most one next-Q ticket and one previous-O ticket in flight."""

    execution: _Execution
    hidden: torch.Tensor
    weight: torch.Tensor
    output_weight: torch.Tensor
    query_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    output_flat: torch.Tensor
    retained: list[torch.Tensor]
    store: SavedStatsOffloadStore | None
    transport: HeadChunkTransport

    def _launch_query(self, spec: _QuerySpec) -> Any:
        state = _project(
            self.execution, spec, self.hidden, self.weight, self.query_norm, self.cos, self.sin,
            kv=False, replay=False,
        )
        return self.transport.launch(state.outputs[0])

    def _retire_output(self, pending: tuple) -> None:
        ticket, spec, saved = pending
        local = self.transport.wait(ticket).transpose(1, 2).contiguous()
        self.output_flat.addmm_(
            local.reshape(self.output_flat.shape[0], -1),
            self.output_weight.index_select(1, spec.columns).transpose(0, 1),
        )
        if saved is not None:
            _retain_stage(self.execution, spec, saved, self.retained, self.store)

    def run(self, group: _KVSpec, key: torch.Tensor, value: torch.Tensor) -> None:
        """Submit next Q before current O, then retire O after the next FA."""
        pending_query = self._launch_query(group.stages[0])
        pending_output = None
        for position, spec in enumerate(group.stages):
            query = self.transport.wait(pending_query)
            pending_query = None
            if position + 1 < len(group.stages):
                pending_query = self._launch_query(group.stages[position + 1])
            attention, saved = _fusion_attention_forward_saved(query, key, value, self.execution.scaling)
            if self.execution.early_saved_stats:
                _retain_stage(self.execution, spec, saved, self.retained, self.store)
            if pending_output is not None:
                self._retire_output(pending_output)
                pending_output = None
            late_saved = saved if self.execution.retain_for_backward and not self.execution.early_saved_stats else None
            pending_output = (self.transport.launch(attention, inverse=True), spec, late_saved)
            del query, attention, saved, late_saved
        self._retire_output(pending_output)


@dataclass
class _ReadyBackwardStage:
    """Current FA-grad inputs, independent of the previous stage's local VJP."""

    spec: _QuerySpec
    state: _Projection
    query: torch.Tensor
    saved: tuple[torch.Tensor, ...]
    grad_attention: torch.Tensor
    saved_ticket: Any


@dataclass
class _BackwardPipeline:
    """Delay dQ's wait past the next FA-grad without reordering any gradient add."""

    execution: _Execution
    hidden: torch.Tensor
    weight: torch.Tensor
    output_weight: torch.Tensor
    query_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    grad_flat: torch.Tensor
    grad_hidden: torch.Tensor
    grad_output_weight: torch.Tensor
    grad_query_norm: torch.Tensor
    grad_key: torch.Tensor
    grad_value: torch.Tensor
    weight_blocks: list
    retained: list[torch.Tensor]
    store: SavedStatsOffloadStore | None
    transport: HeadChunkTransport
    stage_count: int

    def _prepare(self, spec: _QuerySpec, ticket: Any) -> tuple[_ReadyBackwardStage, Any]:
        if self.store is not None and ticket is None:
            ticket = self.store.prefetch_stage(spec.index)
        next_ticket = None
        if self.execution.early_saved_stats:
            next_ticket = _next_saved_ticket(self.store, spec.index, self.stage_count)
        state = _project(
            self.execution, spec, self.hidden, self.weight, self.query_norm, self.cos, self.sin,
            kv=False, replay=True,
        )
        query = _seq_to_head(state.outputs[0].detach(), self.execution.cp_mesh)
        if self.store is None:
            start = _FA_SAVED_TENSOR_COUNT * spec.index
            saved = tuple(self.retained[start:start + _FA_SAVED_TENSOR_COUNT])
        else:
            saved = self.store.consume(ticket) + tuple(self.retained[2 * spec.index:2 * spec.index + 2])
            if not self.execution.early_saved_stats:
                next_ticket = _next_saved_ticket(self.store, spec.index, self.stage_count)
        local = _head_to_seq(saved[0], self.execution.cp_mesh).transpose(1, 2).contiguous()
        _write_output_gradient(
            self.grad_output_weight,
            self.grad_flat.transpose(0, 1).mm(local.reshape(self.grad_flat.shape[0], -1)), spec,
        )
        grad_attention = self.grad_flat.mm(self.output_weight.index_select(1, spec.columns))
        grad_attention = grad_attention.view(*self.hidden.shape[:2], spec.count, self.execution.head_dim)
        grad_attention = _seq_to_head(grad_attention.transpose(1, 2).contiguous(), self.execution.cp_mesh)
        return _ReadyBackwardStage(spec, state, query, saved, grad_attention, ticket), next_ticket

    def _retire_query(self, pending: tuple) -> None:
        ticket, state, spec = pending
        local_dq = self.transport.wait(ticket)
        norm_gradient, weight_gradient = _projection_vjp(
            state, (local_dq,), self.hidden, self.weight, spec.rows, self.grad_hidden,
        )
        self.grad_query_norm.add_(norm_gradient.float())
        self.weight_blocks.append((spec.rows, weight_gradient))

    def run(self, group: _KVSpec, ticket: Any) -> Any:
        """Prepare next Q/O/dO before submitting current inverse-dQ on the same PG."""
        ready, next_ticket = self._prepare(group.stages[0], ticket)
        pending_query = None
        for position, spec in enumerate(group.stages):
            dq, dk, dv = _fusion_attention_backward_saved(
                ready.query, self.key, self.value, ready.grad_attention, self.execution.scaling, ready.saved,
            )
            self.grad_key.add_(dk.float())
            self.grad_value.add_(dv.float())
            del dk, dv
            if pending_query is not None:
                self._retire_query(pending_query)
                pending_query = None
            state = ready.state
            if self.store is not None:
                self.store.release_slot(ready.saved_ticket)
                if next_ticket is None and spec.index + 1 < self.stage_count:
                    next_ticket = self.store.prefetch_stage(spec.index + 1)
            ready = None
            if position + 1 < len(group.stages):
                ready, next_ticket = self._prepare(group.stages[position + 1], next_ticket)
            # Next Q/O/dO have already passed HCCL. Their consumers must not
            # wait behind dQ_i; the next FA-grad runs before retiring this VJP.
            pending_query = (self.transport.launch(dq, inverse=True), state, spec)
            del dq, state
        self._retire_query(pending_query)
        return next_ticket


class SharedKVHeadChunkFunction(torch.autograd.Function):
    """Saved-FA-stat backward with bounded query-stage communication pipelines."""

    @staticmethod
    def forward(
        ctx: Any, hidden: torch.Tensor, weight: torch.Tensor, output_weight: torch.Tensor,
        query_norm: torch.Tensor, key_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        execution: _Execution,
    ) -> torch.Tensor:
        """Preserve output add order with optional one-stage communication look-ahead."""
        store = None
        if execution.retain_for_backward and execution.offload_pool is not None:
            store = SavedStatsOffloadStore(
                hidden.device, execution.offload_pool, execution.prefetch_slots,
                statistics=execution.offload_statistics,
                retire_d2h_sources=execution.retire_d2h_sources,
            )
        ctx.offload_store = store
        ctx.offload_used = False
        try:
            output = hidden.new_zeros((*hidden.shape[:2], output_weight.shape[0]))
            output_flat = output.view(-1, output.shape[-1])
            retained = []
            with HeadChunkTransport(
                execution.cp_mesh, trim_stream_records=execution.trim_a2a_stream_records,
                release_source=execution.release_a2a_source,
            ) as transport:
                distributed = execution.cp_mesh is not None and execution.cp_mesh.size() > 1
                for group in execution.plan:
                    kv = _project(execution, group, hidden, weight, key_norm, cos, sin, kv=True, replay=False)
                    key, value = (_seq_to_head(tensor, execution.cp_mesh) for tensor in kv.outputs)
                    del kv
                    if execution.pipeline_output_a2a and distributed and len(group.stages) > 1:
                        pipeline = _ForwardPipeline(
                            execution, hidden, weight, output_weight, query_norm, cos, sin,
                            output_flat, retained, store, transport,
                        )
                        pipeline.run(group, key, value)
                        del pipeline, key, value
                        continue
                    for spec in group.stages:
                        state = _project(execution, spec, hidden, weight, query_norm, cos, sin, kv=False, replay=False)
                        query = _seq_to_head(state.outputs[0], execution.cp_mesh)
                        del state
                        attention, saved = _fusion_attention_forward_saved(query, key, value, execution.scaling)
                        if execution.early_saved_stats:
                            _retain_stage(execution, spec, saved, retained, store)
                        local = _head_to_seq(attention, execution.cp_mesh).transpose(1, 2).contiguous()
                        output_flat.addmm_(local.reshape(output_flat.shape[0], -1),
                                           output_weight.index_select(1, spec.columns).transpose(0, 1))
                        if not execution.early_saved_stats:
                            _retain_stage(execution, spec, saved, retained, store)
                        del query, attention, saved, local
                    del key, value
            if store is not None and execution.retire_d2h_sources:
                # No D2H source ticket may survive into another layer's forward.
                store.finish_forward()
            if execution.retain_for_backward:
                ctx.save_for_backward(hidden, weight, output_weight, query_norm, key_norm, cos, sin, *retained)
                ctx.execution = execution
            return output
        except Exception:
            if store is not None:
                store.close()
                ctx.offload_store = None
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Return full gradients after every pipeline tail and collective is drained."""
        if ctx.offload_used:
            raise RuntimeError(
                "Head Chunk saved-state offload supports one backward per forward; "
                "retain_graph reuse requires offload_saved_stats=False"
            )
        store = ctx.offload_store
        if store is not None:
            ctx.offload_used = True
        try:
            hidden, weight, output_weight, query_norm, key_norm, cos, sin, *retained = ctx.saved_tensors
            execution = ctx.execution
            grad_hidden = torch.zeros_like(hidden, memory_format=torch.contiguous_format)
            grad_output_weight = torch.empty_like(output_weight)
            grad_weight_blocks = []
            grad_query_norm = torch.zeros_like(query_norm, dtype=torch.float32)
            grad_key_norm = torch.zeros_like(key_norm, dtype=torch.float32)
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            stage_count = sum(len(group.stages) for group in execution.plan)
            ticket = store.prefetch_stage(0) if store is not None else None
            with HeadChunkTransport(
                execution.cp_mesh, trim_stream_records=execution.trim_a2a_stream_records,
                release_source=execution.release_a2a_source,
            ) as transport:
                distributed = execution.cp_mesh is not None and execution.cp_mesh.size() > 1
                for group in execution.plan:
                    kv = _project(execution, group, hidden, weight, key_norm, cos, sin, kv=True, replay=True)
                    key, value = (_seq_to_head(tensor.detach(), execution.cp_mesh) for tensor in kv.outputs)
                    grad_key = torch.zeros_like(key, dtype=torch.float32)
                    grad_value = torch.zeros_like(value, dtype=torch.float32)
                    if execution.pipeline_dq_a2a and distributed and len(group.stages) > 1:
                        pipeline = _BackwardPipeline(
                            execution, hidden, weight, output_weight, query_norm, cos, sin, key, value,
                            grad_flat, grad_hidden, grad_output_weight, grad_query_norm, grad_key, grad_value,
                            grad_weight_blocks, retained, store, transport, stage_count,
                        )
                        ticket = pipeline.run(group, ticket)
                        del pipeline
                    else:
                        for spec in group.stages:
                            next_ticket = None
                            if execution.early_saved_stats:
                                next_ticket = _next_saved_ticket(store, spec.index, stage_count)
                            state = _project(
                                execution, spec, hidden, weight, query_norm, cos, sin, kv=False, replay=True,
                            )
                            query = _seq_to_head(state.outputs[0].detach(), execution.cp_mesh)
                            if store is None:
                                start = _FA_SAVED_TENSOR_COUNT * spec.index
                                saved = tuple(retained[start:start + _FA_SAVED_TENSOR_COUNT])
                            else:
                                saved = store.consume(ticket) + tuple(retained[2 * spec.index:2 * spec.index + 2])
                                if not execution.early_saved_stats:
                                    next_ticket = _next_saved_ticket(store, spec.index, stage_count)
                            local = _head_to_seq(saved[0], execution.cp_mesh).transpose(1, 2).contiguous()
                            _write_output_gradient(
                                grad_output_weight,
                                grad_flat.transpose(0, 1).mm(local.reshape(grad_flat.shape[0], -1)), spec,
                            )
                            grad_attention = grad_flat.mm(output_weight.index_select(1, spec.columns))
                            grad_attention = grad_attention.view(*hidden.shape[:2], spec.count, execution.head_dim)
                            grad_attention = _seq_to_head(
                                grad_attention.transpose(1, 2).contiguous(), execution.cp_mesh,
                            )
                            del local
                            dq, dk, dv = _fusion_attention_backward_saved(
                                query, key, value, grad_attention, execution.scaling, saved,
                            )
                            grad_key.add_(dk.float())
                            grad_value.add_(dv.float())
                            local_dq = _head_to_seq(dq, execution.cp_mesh)
                            norm_gradient, weight_gradient = _projection_vjp(
                                state, (local_dq,), hidden, weight, spec.rows, grad_hidden,
                            )
                            grad_query_norm.add_(norm_gradient.float())
                            grad_weight_blocks.append((spec.rows, weight_gradient))
                            del state, query, saved, grad_attention, dq, dk, dv
                            del local_dq, norm_gradient, weight_gradient
                            if store is not None:
                                store.release_slot(ticket)
                                if next_ticket is None and spec.index + 1 < stage_count:
                                    next_ticket = store.prefetch_stage(spec.index + 1)
                                ticket = next_ticket
                    local_dkv = tuple(_head_to_seq(tensor, execution.cp_mesh)
                                      for tensor in (grad_key.to(key.dtype), grad_value.to(value.dtype)))
                    norm_gradient, weight_gradient = _projection_vjp(
                        kv, local_dkv, hidden, weight, group.rows, grad_hidden,
                    )
                    grad_key_norm.add_(norm_gradient.float())
                    grad_weight_blocks.append((group.rows, weight_gradient))
                    del kv, key, value, grad_key, grad_value, local_dkv, norm_gradient, weight_gradient
            grad_weight = _assemble_packed_gradient(weight, grad_weight_blocks)
            gradients = (grad_hidden, grad_weight, grad_output_weight,
                         grad_query_norm.to(query_norm.dtype), grad_key_norm.to(key_norm.dtype))
            return tuple(gradient if ctx.needs_input_grad[index] else None
                         for index, gradient in enumerate(gradients)) + (None, None, None)
        finally:
            if store is not None:
                try:
                    store.close()
                finally:
                    ctx.offload_store = None
