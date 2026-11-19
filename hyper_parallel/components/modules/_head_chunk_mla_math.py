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
"""Private low-rank MLA projections and explicit first-order VJPs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.func import functional_call
from torch.nn import functional as F

from hyper_parallel.core.activation_memory import PinnedMemoryPool
from hyper_parallel.core.utils import communication
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head
from ._head_chunk_mla_ops import _rotate
from ._head_chunk_mla_offload import _OffloadStatistics

_SAVED_TENSOR_COUNT = 3
_COMPRESSED_COUNTERS = (
    "forward_allgather_calls", "backward_allgather_calls", "backward_reduce_scatter_calls",
    "allgather_input_bytes", "allgather_output_bytes", "allgather_remote_receive_bytes",
    "reduce_scatter_input_bytes", "reduce_scatter_output_bytes", "reduce_scatter_remote_send_bytes",
    "completed_forward", "completed_backward",
)


def head_stage_schedule(num_heads: int, head_chunk_size: int, cp_size: int = 1) -> tuple[tuple[int, int], ...]:
    """Return contiguous, CP-divisible ``(first_head, head_count)`` stages."""
    counts = (num_heads, head_chunk_size, cp_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise ValueError("head counts and CP size must be positive integers")
    if num_heads % cp_size or head_chunk_size % cp_size:
        raise ValueError("num_heads and head_chunk_size must both be divisible by CP size")
    if head_chunk_size > num_heads:
        raise ValueError("head_chunk_size cannot exceed num_heads")
    return tuple((start, min(head_chunk_size, num_heads - start))
                 for start in range(0, num_heads, head_chunk_size))


def _seq_to_head(tensor: torch.Tensor, mesh: Any) -> torch.Tensor:
    if mesh is None or mesh.size() == 1:
        return tensor
    return ulysses_seq_to_head(tensor, 2, 1, mesh)


def _head_to_seq(tensor: torch.Tensor, mesh: Any) -> torch.Tensor:
    if mesh is None or mesh.size() == 1:
        return tensor
    return ulysses_head_to_seq(tensor, 2, 1, mesh)


@dataclass(frozen=True)
class _Execution:
    stages: tuple[tuple[int, int], ...]
    query_rank: int
    kv_rank: int
    nope_dim: int
    rope_dim: int
    value_dim: int
    scale: float
    query_norm: nn.Module
    kv_norm: nn.Module
    backend_name: str
    backend: Any
    cp_mesh: Any
    retain: bool
    pool: PinnedMemoryPool | None
    prefetch_slots: int
    statistics: _OffloadStatistics
    retire_d2h_sources: bool = True
    release_local_output_early: bool = True
    use_output_gradient_edges: bool = True

    @property
    def query_dim(self) -> int:
        """Return the combined NoPE and RoPE width of one query head."""
        return self.nope_dim + self.rope_dim


@dataclass
class _SharedProjection:
    raw: torch.Tensor
    query_norm_weight: torch.Tensor
    kv_norm_weight: torch.Tensor
    query_latent: torch.Tensor
    kv_latent: torch.Tensor
    key_rope: torch.Tensor


def _project_shared(execution, hidden, weight, query_norm_weight, kv_norm_weight, cos, sin, *, replay):
    raw = F.linear(hidden, weight).detach().requires_grad_(replay)
    query_norm_weight = query_norm_weight.detach().requires_grad_(replay)
    kv_norm_weight = kv_norm_weight.detach().requires_grad_(replay)
    with torch.set_grad_enabled(replay):
        query, kv, rope = raw.split((execution.query_rank, execution.kv_rank, execution.rope_dim), dim=-1)
        query_latent = functional_call(execution.query_norm, {"weight": query_norm_weight}, (query,), strict=True)
        kv_latent = functional_call(execution.kv_norm, {"weight": kv_norm_weight}, (kv,), strict=True)
        key_rope = _rotate(rope.unsqueeze(1), cos, sin, execution.backend_name)
    return _SharedProjection(raw, query_norm_weight, kv_norm_weight, query_latent, kv_latent, key_rope)


def _release_stage_outputs(state: Any) -> None:
    """Retain VJP entry nodes without retaining the final projection tensors."""
    # FA-layout consumers already exist. Do not force storage release: CP1
    # aliases and autograd-saved inputs may still legitimately own the values.
    state.outputs = tuple(get_gradient_edge(tensor) for tensor in state.outputs)


def _accumulator(tensor):
    dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    return torch.zeros_like(tensor, dtype=dtype)


@dataclass(frozen=True)
class _LatentExecution(_Execution):
    compressed_statistics: dict[str, int] = field(default_factory=dict)
    release_global_buffers: bool = True
    overlap_backward: bool = False


@dataclass
class _QueryProjection:
    raw_query: torch.Tensor
    outputs: tuple[torch.Tensor | GradientEdge, ...]


def _project_query_stage(execution, shared, query_weight, stage, cos, sin, *, replay):
    """Keep the original local Q math without computing local expanded K/V."""
    start, count = stage
    query_rows = query_weight.narrow(0, start * execution.query_dim, count * execution.query_dim)
    raw_query = F.linear(shared.query_latent.detach(), query_rows).detach().requires_grad_(replay)
    with torch.set_grad_enabled(replay):
        shape = (*raw_query.shape[:2], count)
        query = raw_query.view(*shape, execution.query_dim).transpose(1, 2)
        query_nope, query_rope = query.split((execution.nope_dim, execution.rope_dim), dim=-1)
        query = torch.cat((query_nope, _rotate(query_rope, cos, sin, execution.backend_name)), dim=-1)
        outputs = (query.contiguous(),)
    return _QueryProjection(raw_query, outputs)


def _publish_compressed_kv(execution, shared, phase):
    """Gather normalized latent and already-positioned one-head RoPE once."""
    mesh = execution.cp_mesh
    if mesh is None or mesh.size() == 1:
        return shared.kv_latent.detach(), shared.key_rope.detach()
    packed = torch.cat((shared.kv_latent.detach(), shared.key_rope.detach().squeeze(1)), dim=-1)
    published = communication.differentiable_all_gather_concat(
        packed, mesh.get_group(), mesh.size(), 1,
    )
    stats = execution.compressed_statistics
    stats[f"{phase}_allgather_calls"] += 1
    input_bytes = packed.numel() * packed.element_size()
    output_bytes = published.numel() * published.element_size()
    stats["allgather_input_bytes"] += input_bytes
    stats["allgather_output_bytes"] += output_bytes
    stats["allgather_remote_receive_bytes"] += output_bytes - input_bytes
    latent, rope = published.split((execution.kv_rank, execution.rope_dim), dim=-1)
    return latent, rope.unsqueeze(1)


def _owner_stage(execution, stage):
    start, count = stage
    mesh = execution.cp_mesh
    cp_size = 1 if mesh is None else mesh.size()
    cp_rank = 0 if mesh is None else mesh.get_local_rank()
    owner_count = count // cp_size
    return start + cp_rank * owner_count, owner_count


def _project_owner_kv_stage(execution, latent, rope, kv_weight, stage):
    """Expand only this Ulysses owner's heads over the full published sequence."""
    start, count = _owner_stage(execution, stage)
    kv_width = execution.nope_dim + execution.value_dim
    rows = kv_weight.narrow(0, start * kv_width, count * kv_width)
    raw = F.linear(latent, rows)
    expanded = raw.view(*raw.shape[:2], count, kv_width).transpose(1, 2)
    key_nope, value = expanded.split((execution.nope_dim, execution.value_dim), dim=-1)
    key = torch.cat((key_nope, rope.expand(-1, count, -1, -1)), dim=-1)
    return key.contiguous(), value.contiguous()


def _query_stage_vjp(execution, state, grad_query, shared, query_weight, stage, accumulator):
    """Apply the original local query VJP and FP32 latent accumulation."""
    grad_raw, = torch.autograd.grad(state.outputs, (state.raw_query,), (grad_query,))
    query_flat = grad_raw.flatten(0, 1)
    query_latent = shared.query_latent.detach().flatten(0, 1)
    start, count = stage
    rows = query_weight.narrow(0, start * execution.query_dim, count * execution.query_dim)
    accumulator.add_(query_flat.mm(rows).view_as(shared.query_latent))
    return query_flat.transpose(0, 1).mm(query_latent)


def _owner_kv_stage_vjp(execution, grad_key, grad_value, latent, kv_weight, stage, accumulators):
    """Accumulate head-owner activation gradients; return this stage's weight rows."""
    start, count = _owner_stage(execution, stage)
    kv_width = execution.nope_dim + execution.value_dim
    rows = kv_weight.narrow(0, start * kv_width, count * kv_width)
    grad_nope, grad_rope = grad_key.split((execution.nope_dim, execution.rope_dim), dim=-1)
    grad_raw = torch.cat((grad_nope, grad_value), dim=-1).transpose(1, 2).contiguous()
    grad_flat = grad_raw.view(*latent.shape[:2], count * kv_width).flatten(0, 1)
    accumulators[0].add_(grad_flat.mm(rows).view_as(latent))
    # Match the original expand-backward output dtype before FP32 accumulation;
    # head-owner grouping itself still changes the BF16 reduction order.
    accumulators[1].add_(grad_rope.sum(dim=1, keepdim=True))
    return grad_flat.transpose(0, 1).mm(latent.flatten(0, 1))


def _reduce_owner_kv_gradients(execution, accumulators):
    """Return SUM-reduced token-owner gradients, never parameter gradients."""
    mesh = execution.cp_mesh
    if mesh is None or mesh.size() == 1:
        return accumulators
    reduced = []
    for index, axis in enumerate((1, 2)):
        tensor = accumulators[index]
        local = communication.differentiable_reduce_scatter(
            tensor, mesh.size(), axis, "sum", mesh.get_group(),
        )
        stats = execution.compressed_statistics
        stats["backward_reduce_scatter_calls"] += 1
        input_bytes = tensor.numel() * tensor.element_size()
        output_bytes = local.numel() * local.element_size()
        stats["reduce_scatter_input_bytes"] += input_bytes
        stats["reduce_scatter_output_bytes"] += output_bytes
        stats["reduce_scatter_remote_send_bytes"] += input_bytes - output_bytes
        reduced.append(local)
        if execution.release_global_buffers:
            accumulators[index] = None
            del tensor
    return tuple(reduced)
