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

"""MoonEP token/rank deduplication and pre-reduced combine over collectives.

The mapping corresponds to native dispatch primary/duplicate slots: send a
hidden row once per destination, expand directly into expert-major order,
then accumulate weighted duplicates before the inverse token exchange.
"""

import math
from typing import Any

import torch
import torch.distributed as dist

# These helpers share the planner, expert semantics and weight-gradient path
# with the assignment transport used as the no-dedup numerical reference.
from hyper_parallel.distributed.expert_parallel.moonep_collective import (
    _compute_experts,
    _Exchange,
    _exchange_weights,
    _plan_assignments,
)
from hyper_parallel.distributed.expert_parallel.moonep_collective_packet import (
    PackedDispatch,
    packet_layout,
)


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate duplicate low-precision rows in FP32."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _unique_sends(order: torch.Tensor, send: list[int], top_k: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Deduplicate within each destination; inverse IDs remain peer-local."""
    unique_ids, inverse, counts = [], [], []
    offset = 0
    for count in send:
        tokens = order[offset : offset + count] // top_k
        unique, mapping = torch.unique(tokens, sorted=True, return_inverse=True)
        unique_ids.append(unique)
        inverse.append(mapping)
        counts.append(unique.numel())
        offset += count
    return torch.cat(unique_ids), torch.cat(inverse), counts


def collective_token_forward(
    experts: Any,
    states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    *,
    group: Any,
    diagnostics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Dispatch one hidden row per token/destination and pre-reduce outputs.

    Args:
        experts: Local packed SwiGLU parameter holder entered via module hooks.
        states: Original unexpanded token rows [T, H].
        indices: Global expert IDs [T, K].
        weights: Route weights [T, K], preserving their own floating dtype.
        group: EP process group, using group-local rank and expert numbering.
        diagnostics: Optional per-call local transport counters and global loads.
            Recomputed calls are separate observations; no state is cached.

    Returns:
        Weighted token outputs [T, H]. Each expert choice remains distinct even
        when hidden payloads are deduplicated; all three gradients are preserved.
    """
    if (
        states.ndim != 2
        or indices.ndim != 2
        or weights.shape != indices.shape
        or indices.shape[0] != states.shape[0]
        or indices.shape[1] < 1
    ):
        raise ValueError("expected states [T,H], matching indices/weights [T,K] and K >= 1")
    if indices.dtype != torch.int64 or indices.device != states.device or weights.device != states.device:
        raise ValueError("indices must be int64 and all token fields must share a device")
    if not states.is_floating_point() or not weights.is_floating_point():
        raise ValueError("hidden states and route weights must be floating point")
    dtype_codes = {
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float32: 3,
        torch.float64: 4,
    }
    if states.dtype not in dtype_codes or weights.dtype not in dtype_codes:
        raise ValueError("collective token payloads support float16, bfloat16, float32 and float64")
    specification = (
        states.shape[1],
        dtype_codes[states.dtype],
        dtype_codes[weights.dtype],
    )
    plan, order, send, receive, sorting, counts, rank = _plan_assignments(
        experts, indices.flatten(), group, specification
    )
    source_tokens, inverse, send_unique = _unique_sends(order, send, indices.shape[1])
    count_tensor = torch.tensor(send_unique, device=states.device, dtype=torch.int64)
    receive_tensor = torch.empty_like(count_tensor)
    dist.all_to_all_single(receive_tensor, count_tensor, group=group)
    receive_unique = receive_tensor.cpu().tolist()
    # Casting before indexing makes duplicate hidden gradients sum in FP32.
    send_states = states.to(_accumulation_dtype(states.dtype))[source_tokens].to(states.dtype)
    received, route, expansion = PackedDispatch.apply(
        send_states,
        weights.flatten()[order],
        inverse,
        send_unique,
        receive_unique,
        send,
        receive,
        group,
    )
    gate, down, _ = _exchange_weights(experts, plan, rank, group, received)
    expert_expansion = expansion[sorting]
    sorted_states = received.to(_accumulation_dtype(received.dtype))[expert_expansion].to(states.dtype)
    output = _compute_experts(experts, sorted_states, gate, down, counts)
    output = output + gate[:0].sum() + down[:0].sum() + received[:0].sum()
    output_dtype = torch.promote_types(output.dtype, weights.dtype)
    accumulator_dtype = _accumulation_dtype(output_dtype)
    weighted = output.to(accumulator_dtype) * route[sorting, None].to(accumulator_dtype)
    combined = torch.zeros(
        (sum(receive_unique), states.shape[1]),
        device=states.device,
        dtype=accumulator_dtype,
    )
    combined.index_add_(0, expert_expansion, weighted)
    # Low-precision partial sums round before the final source-rank reduction.
    combined = combined.to(output_dtype)
    returned = _Exchange.apply(combined, receive_unique, send_unique, group)
    result = torch.zeros(states.shape, device=states.device, dtype=accumulator_dtype)
    result.index_add_(0, source_tokens, returned.to(accumulator_dtype))
    result = result.to(output_dtype)
    if diagnostics is not None:
        packet_bytes = [
            packet_layout(u, a, states.shape[1], states.dtype, weights.dtype).total_bytes
            for u, a in zip(send_unique, send)
        ]
        remote = [peer for peer in range(len(send)) if peer != rank]
        remote_copies = sum(
            expert // plan.experts_per_rank != peer for peer in range(len(send)) for expert in plan.experts_on(peer)
        )
        diagnostics.update(
            scope="local_source_rank",
            assignment_rows=sum(send),
            unique_token_rank_rows=sum(send_unique),
            remote_assignment_rows=sum(send[peer] for peer in remote),
            remote_unique_rows=sum(send_unique[peer] for peer in remote),
            remote_hidden_bytes_before=sum(send[peer] for peer in remote) * states.shape[1] * states.element_size(),
            remote_hidden_bytes_after=(
                sum(send_unique[peer] for peer in remote) * states.shape[1] * states.element_size()
            ),
            remote_dispatch_packet_bytes=sum(packet_bytes[peer] for peer in remote),
            remote_combine_bytes=sum(send_unique[peer] for peer in remote) * states.shape[1] * result.element_size(),
            remote_combine_bytes_before=sum(send[peer] for peer in remote) * states.shape[1] * result.element_size(),
            global_execution_assignments=list(plan.execution_counts),
            global_remote_expert_copies=remote_copies,
            global_weight_forward_bytes=(
                remote_copies * (math.prod(gate.shape[1:]) + math.prod(down.shape[1:])) * gate.element_size()
            ),
        )
    return result
