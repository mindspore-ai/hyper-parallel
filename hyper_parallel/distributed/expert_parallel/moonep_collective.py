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

"""Synchronous collective transport for load-balanced redundant SwiGLU experts.

Counts deliberately cross to the host once per call for deterministic planning.
No weights, plans or scratch buffers are cached across forwards or recomputes.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.experts import resolve_swiglu_weights
from hyper_parallel.distributed.expert_parallel.moonep_collective_plan import (
    CollectivePlan,
    build_collective_plan,
)


class _Exchange(torch.autograd.Function):
    """Uneven synchronous all-to-all whose backward is the inverse exchange."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, send: list[int], receive: list[int], group: Any) -> torch.Tensor:
        """Exchange rows in process-group rank order."""
        ctx.send, ctx.receive, ctx.group = send, receive, group
        output = tensor.new_empty((sum(receive), *tensor.shape[1:]))
        dist.all_to_all_single(output, tensor.contiguous(), receive, send, group=group)
        # Non-reentrant checkpoint must replay collectives even on empty ranks.
        # A saved empty sentinel triggers replay without retaining the payload.
        ctx.save_for_backward(tensor.new_empty(0))
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Reverse the exchange, replaying checkpointed collectives."""
        (_sentinel,) = ctx.saved_tensors
        output = gradient.new_empty((sum(ctx.send), *gradient.shape[1:]))
        dist.all_to_all_single(output, gradient.contiguous(), ctx.send, ctx.receive, group=ctx.group)
        return output, None, None, None


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Read a local expert shard while retaining its autograd edge."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _exchange_weights(
    experts: Any, plan: CollectivePlan, rank: int, group: Any, dependency: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Copy requested experts and return remote gradients to their owners."""
    gate, up, down = resolve_swiglu_weights(experts)
    if up is not None:
        raise ValueError("MoonEP collective currently requires packed gate_up_proj")
    gate, down = _local(gate), _local(down)
    # GroupedExperts stores GEMM-ready [E,H,2I]/[E,I,H] parameters. Normalize
    # views before packing so the inverse exchange still reaches those owners.
    if getattr(experts, "is_transposed", False):
        gate, down = gate.transpose(-1, -2), down.transpose(-1, -2)
    if (
        gate.ndim != 3
        or down.ndim != 3
        or gate.shape[0] != plan.experts_per_rank
        or down.shape[0] != plan.experts_per_rank
    ):
        raise ValueError("MoonEP collective requires local complete stacked expert weights")
    if gate.shape[1] != 2 * down.shape[2] or gate.shape[2] != down.shape[1]:
        raise ValueError("MoonEP collective requires [E, 2I, H] and [E, H, I] weights")
    local_count = plan.experts_per_rank
    size = len(plan.counts)
    requests = plan.experts_by_rank()
    outgoing = [
        [expert % local_count for expert in request if expert // local_count == rank] if peer != rank else []
        for peer, request in enumerate(requests)
    ]
    incoming = [
        [expert for expert in requests[rank] if expert // local_count == owner] if owner != rank else []
        for owner in range(size)
    ]
    indices = torch.tensor(
        [expert for peers in outgoing for expert in peers],
        device=gate.device,
        dtype=torch.int64,
    )
    # Pack only requested remote copies, rather than all local expert weights.
    packed = torch.cat((gate[indices].flatten(1), down[indices].flatten(1)), dim=1)
    # Order weight-exchange backward before token-exchange backward on every rank.
    packed = packed + dependency[:0].sum()
    exchanged = _Exchange.apply(packed, list(map(len, outgoing)), list(map(len, incoming)), group)
    remote_position = {
        expert: position for position, expert in enumerate(expert for peers in incoming for expert in peers)
    }
    local_experts = [expert for expert in requests[rank] if expert // local_count == rank]
    local_indices = torch.tensor(
        [expert % local_count for expert in local_experts], device=gate.device, dtype=torch.int64
    )
    local_weights = torch.cat((gate[local_indices].flatten(1), down[local_indices].flatten(1)), dim=1)
    positions = {expert: position for position, expert in enumerate(local_experts)}
    positions.update({expert: len(local_experts) + position for expert, position in remote_position.items()})
    active_indices = torch.tensor(
        [positions[expert] for expert in requests[rank]], device=gate.device, dtype=torch.int64
    )
    active_weights = torch.cat((local_weights, exchanged), dim=0)[active_indices]
    # Even ranks needing only local weights participate in the reverse exchange.
    active_weights = active_weights + exchanged[:0].sum()
    gate_elements = gate[0].numel()
    received_gate = active_weights[:, :gate_elements].reshape(-1, *gate.shape[1:])
    received_down = active_weights[:, gate_elements:].reshape(-1, *down.shape[1:])
    return received_gate, received_down, requests[rank]


def _segment_ranges(starts: Sequence[int], counts: Sequence[int]) -> np.ndarray:
    """Expand contiguous segment ranges without per-assignment Python objects."""
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    return np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(
        np.asarray(starts, dtype=np.int64) - offsets, counts
    )


def _assignment_layout(
    plan: CollectivePlan, rank: int, expert_order: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build peer-major send and stable expert-major receive permutations.

    Only segment metadata is traversed in Python. Receive expert IDs remain
    implicit in segment counts, avoiding a full ID tensor and device argsort.
    """
    sending = sorted(
        (segment for segment in plan.segments if segment.source == rank),
        key=lambda segment: segment.destination,
    )
    receiving = sorted(
        (segment for segment in plan.segments if segment.destination == rank),
        key=lambda segment: segment.source,
    )
    expert_offsets = np.cumsum((0, *plan.counts[rank]), dtype=np.int64)
    send_positions = _segment_ranges(
        [int(expert_offsets[segment.expert]) + segment.offset for segment in sending],
        [segment.count for segment in sending],
    )
    receive_offsets = np.cumsum([0] + [segment.count for segment in receiving], dtype=np.int64)
    expert_segments = sorted(range(len(receiving)), key=lambda index: receiving[index].expert)
    sorting = _segment_ranges(
        [int(receive_offsets[index]) for index in expert_segments],
        [receiving[index].count for index in expert_segments],
    )
    counts = {}
    for segment in receiving:
        counts[segment.expert] = counts.get(segment.expert, 0) + segment.count
    order = expert_order[torch.from_numpy(send_positions).to(expert_order.device)]
    return (
        order,
        torch.from_numpy(sorting).to(expert_order.device),
        [counts[expert] for expert in sorted(counts)],
    )


def _inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    """Invert a bijective permutation in linear time."""
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))
    return inverse


def _plan_assignments(
    experts: Any, indices: torch.Tensor, group: Any, payload_spec: tuple[int, ...] = ()
) -> tuple[CollectivePlan, torch.Tensor, list[int], list[int], torch.Tensor, list[int], int]:
    """Resolve a common assignment schedule for both transport layouts."""
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    expert_count = experts.local_expert_count * size
    if expert_count < 1:
        raise ValueError("local_expert_count must be positive")
    invalid = ((indices < 0) | (indices >= expert_count)).sum().reshape(1)
    # Fixed-size scatter avoids bincount's device min/max readback on NPU.
    local_counts = torch.zeros(expert_count, device=indices.device, dtype=torch.int64)
    local_counts.scatter_add_(0, indices.clamp(0, expert_count - 1), torch.ones_like(indices))
    specification = torch.tensor(payload_spec, device=indices.device, dtype=torch.int64)
    metadata = torch.cat((local_counts, invalid, specification))
    gathered = [torch.empty_like(metadata) for _ in range(size)]
    dist.all_gather(gathered, metadata, group=group)
    # Validate IDs on every rank before entering payload exchanges.
    host_counts = torch.stack(gathered).cpu().tolist()
    if any(row[expert_count] for row in host_counts):
        raise ValueError("router expert IDs must be within the global expert range")
    if payload_spec and len({tuple(row[expert_count + 1 :]) for row in host_counts}) != 1:
        raise ValueError("all EP ranks must use matching hidden sizes and payload dtypes")
    plan = build_collective_plan([row[:expert_count] for row in host_counts])
    expert_order = torch.argsort(indices, stable=True)
    order, sorting, counts = _assignment_layout(plan, rank, expert_order)
    send, receive = plan.splits(rank)
    return plan, order, send, receive, sorting, counts, rank


def _compute_experts(
    experts: Any,
    sorted_states: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    counts: list[int],
) -> torch.Tensor:
    """Evaluate expert-major rows using the existing packed SwiGLU contract."""
    activation = getattr(experts, "_ep_act_fn", F.silu)
    apply_gate = getattr(experts, "_ep_apply_gate", None)
    # The fused kernel implements SiLU(gate) * up, not arbitrary model gates.
    is_silu = activation is F.silu or isinstance(activation, torch.nn.SiLU)
    if getattr(experts, "_ep_use_grouped_gemm", False) and counts and is_silu and apply_gate is None:
        output = npu_grouped_swiglu(
            sorted_states,
            gate,
            down,
            torch.tensor(counts, device=sorted_states.device, dtype=torch.int64),
        )
    else:
        outputs, start = [], 0
        for position, count in enumerate(counts):
            projected = F.linear(sorted_states[start : start + count], gate[position])
            if apply_gate is None:
                gate_states, up_states = projected.chunk(2, dim=-1)
                activated = activation(gate_states) * up_states
            else:
                activated = apply_gate(projected)
            outputs.append(F.linear(activated, down[position]))
            start += count
        output = torch.cat(outputs) if outputs else sorted_states * 0
    return output


def collective_expert_forward(
    experts: Any,
    states: torch.Tensor,
    indices: torch.Tensor,
    *,
    group: Any,
    diagnostics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Redistribute assignments and evaluate complete packed SwiGLU experts.

    Called through the expert module so parameter hooks own the full lifetime.
    All ranks must participate, including ranks with no input assignments.

    Args:
        experts: Holder with local packed gate/up and down parameters.
        states: Local token rows expanded to one row per top-k assignment.
        indices: Global expert ID for every assignment row.
        group: EP process group; experts are numbered in group rank order.
        diagnostics: Optional caller-owned dictionary receiving per-call global
            counts and logical payload bytes. Collect outside timing windows;
            checkpoint recomputation invokes this function again.

    Returns:
        Expert outputs in the original local assignment order.
    """
    if states.ndim != 2 or indices.ndim != 1 or states.shape[0] != indices.numel():
        raise ValueError("expected states [assignments, hidden] and indices [assignments]")
    if indices.dtype != torch.int64 or indices.device != states.device:
        raise ValueError("expert indices must be int64 on the states device")
    plan, order, send, receive, sorting, counts, rank = _plan_assignments(experts, indices, group)
    size, expert_count = len(plan.counts), len(plan.counts[0])
    received = _Exchange.apply(states[order], send, receive, group)
    gate, down, _ = _exchange_weights(experts, plan, rank, group, received)
    sorted_states = received[sorting]
    output = _compute_experts(experts, sorted_states, gate, down, counts)
    # Empty ranks must retain both exchanges in backward, in the same order.
    output = output + gate[:0].sum() + down[:0].sum() + received[:0].sum()
    inverse = _inverse_permutation(sorting)
    returned = _Exchange.apply(output[inverse], receive, send, group)
    restored = returned[_inverse_permutation(order)]
    if diagnostics is not None:
        owner_load = [0] * size
        for expert in range(expert_count):
            owner_load[expert // plan.experts_per_rank] += sum(row[expert] for row in plan.counts)
        remote_copies = sum(
            expert // plan.experts_per_rank != peer for peer in range(size) for expert in plan.experts_on(peer)
        )
        weight_elements = math.prod(gate.shape[1:]) + math.prod(down.shape[1:])
        diagnostics.update(
            owner_assignments=owner_load,
            execution_assignments=list(plan.execution_counts),
            migrated_assignments=sum(
                segment.count
                for segment in plan.segments
                if segment.destination != segment.expert // plan.experts_per_rank
            ),
            remote_expert_copies=remote_copies,
            weight_forward_bytes=remote_copies * weight_elements * gate.element_size(),
            token_dispatch_bytes=(
                sum(segment.count for segment in plan.segments if segment.source != segment.destination)
                * states.shape[1]
                * states.element_size()
            ),
        )
    return restored


def collective_routed_forward(
    module: Any, hidden_states: torch.Tensor, *, router_fn: Callable, ep_group: Any
) -> torch.Tensor:
    """Route once, execute balanced assignments, and combine on token owners.

    Args:
        module: MoE block whose experts use collective_token_forward.
        hidden_states: Local hidden states with hidden dimension last.
        router_fn: Existing adapter returning global indices and route weights.
        ep_group: EP process group.

    Returns:
        Routed branch output with the same shape as hidden_states.
    """
    indices, weights = router_fn(module, hidden_states)
    states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if indices.ndim != 2 or weights.shape != indices.shape or indices.shape[0] != states.shape[0]:
        raise ValueError("router must return matching [tokens, top_k] indices and weights")
    return module.experts(states, indices, weights, group=ep_group).reshape(hidden_states.shape)
