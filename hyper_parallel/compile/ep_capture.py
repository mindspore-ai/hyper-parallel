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
"""Static-graph capture adapter for an AutoModels dynamic-EP model.

The AutoModels planner remains the owner of expert sharding, router semantics,
and the local expert contract.  This module only supplies a shape-static
transport during FX capture: every source reserves a fixed token capacity for
every EP destination, so routing counts never become Python values.
"""
# This compile subsystem is intentionally Torch-specific.
# pylint: disable=forbidden-backend-import

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from hyper_parallel.distributed.expert_parallel.experts import (
    resolve_swiglu_weights,
)


_EP_COLLECTIVES_PER_ROUTED_CALL = 5


@dataclass
class EPCaptureMetadata:
    """Collective provenance recorded while tracing routed EP."""

    ep_degree: int = 1
    group_names: set[str] = field(default_factory=set)
    routed_call_count: int = 0
    collective_counts_by_group: dict[str, int] = field(default_factory=dict)

    @property
    def expected_collective_count(self) -> int:
        """Return the expected joint-graph All-to-All count."""
        return self.routed_call_count * _EP_COLLECTIVES_PER_ROUTED_CALL

    def record_routed_call(self, ep_group: Any) -> None:
        """Record one routed branch and its concrete process-group name."""
        group_name = getattr(ep_group, "group_name", None)
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(
                "EP graph capture requires a process group with a stable group_name"
            )
        self.group_names.add(group_name)
        self.routed_call_count += 1
        self.collective_counts_by_group[group_name] = (
            self.collective_counts_by_group.get(group_name, 0)
            + _EP_COLLECTIVES_PER_ROUTED_CALL
        )


def _exchange_equal_capacity(
    tensor: torch.Tensor,
    rows_per_peer: int,
    ep_size: int,
    ep_group: Any,
) -> torch.Tensor:
    """Validate the static capacity and run the differentiable exchange."""
    if tensor.shape[0] != rows_per_peer * ep_size:
        raise ValueError("EP equal-capacity buffer has an invalid leading dimension")
    splits = [rows_per_peer] * ep_size
    return all_to_all_single(tensor, splits, splits, ep_group)


def _compute_swiglu_expert(
    states: torch.Tensor,
    weights: tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    expert_index: int,
    activation: Callable,
) -> torch.Tensor:
    """Evaluate one stacked SwiGLU expert independently of token transport.

    Args:
        states: Token activations for this expert, including any masked rows.
        weights: Gate, optional up, and down weights from ``resolve_swiglu_weights``.
        expert_index: Index into the local expert dimension.
        activation: Activation chosen by the model's expert binding.

    Returns:
        Expert outputs with the same leading dimensions as ``states``.
    """
    gate_weight, up_weight, down_weight = weights
    gate_states = states @ gate_weight[expert_index].transpose(0, 1)
    if up_weight is None:
        gate_states, up_states = gate_states.chunk(2, dim=-1)
    else:
        up_states = states @ up_weight[expert_index].transpose(0, 1)
    return (activation(gate_states) * up_states) @ down_weight[expert_index].transpose(0, 1)


def _masked_local_swiglu(
    experts: Any,
    received_states: torch.Tensor,
    local_indices: torch.Tensor,
    local_expert_count: int,
) -> torch.Tensor:
    """Evaluate sharded dynamic-EP expert weights without data-dependent slices."""
    weights = resolve_swiglu_weights(experts)
    activation = experts._ep_act_fn
    output = torch.zeros_like(received_states)
    for expert_index in range(local_expert_count):
        expert_mask = local_indices.eq(expert_index).unsqueeze(-1)
        expert_states = received_states * expert_mask
        expert_output = _compute_swiglu_expert(
            expert_states, weights, expert_index, activation
        )
        output = output + expert_output * expert_mask
    return output


def static_ep_routed_forward(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    router_fn: Callable,
    ep_group: Any,
) -> torch.Tensor:
    """Graph-capturable equivalent of AutoModels ``ep_routed_forward``.

    Router selection and expert parameters come from the already-applied
    dynamic EP factory.  Only the ragged transport is represented differently:
    each peer receives ``local_tokens * top_k`` reserved rows from every source.
    Invalid rows are masked before combine, preserving dynamic-EP numerics.

    Args:
        module: EP-sharded MoE module prepared by the dynamic EP factory.
        hidden_states: Local token activations shaped ``[batch, sequence, hidden]``.
        router_fn: Dynamic model-family router adapter.
        ep_group: Expert-parallel process group.

    Returns:
        Routed expert output in the input activation shape.
    """
    ep_size = ep_group.size()
    metadata = getattr(module.experts, "_ep_capture_metadata", None)
    if metadata is not None:
        if ep_size != metadata.ep_degree:
            raise ValueError("Routed EP process group size does not match ep_degree")
        metadata.record_routed_call(ep_group)
    ep_rank = dist.get_rank(group=ep_group)
    local_expert_count = module.experts.local_expert_count
    expert_offset = ep_rank * local_expert_count

    batch_size, sequence_length, hidden_size = hidden_states.shape
    topk_indices, topk_weights = router_fn(module, hidden_states)
    flat_states = hidden_states.reshape(-1, hidden_size)
    token_count = flat_states.shape[0]
    experts_per_token = topk_indices.shape[1]
    expert_indices = topk_indices.reshape(-1)
    expert_weights = topk_weights.reshape(-1).to(flat_states.dtype)
    source_indices = torch.arange(
        token_count, device=hidden_states.device
    ).repeat_interleave(experts_per_token)
    destination_ranks = torch.div(
        expert_indices, local_expert_count, rounding_mode="floor"
    )

    routed_count = expert_indices.numel()
    # Every destination reserves all routed rows, so source row indices are
    # already unique slots; packing by destination would add a redundant scan.
    positions = torch.arange(routed_count, device=hidden_states.device)
    slots = destination_ranks * routed_count + positions

    send_states = flat_states.new_zeros((ep_size * routed_count, hidden_size))
    send_states = send_states.index_copy(0, slots, flat_states[source_indices])
    send_experts = expert_indices.new_full((ep_size * routed_count,), -1)
    send_experts = send_experts.index_copy(0, slots, expert_indices)

    received_states = _exchange_equal_capacity(
        send_states, routed_count, ep_size, ep_group
    )
    received_experts = _exchange_equal_capacity(
        send_experts, routed_count, ep_size, ep_group
    )
    valid = received_experts.ge(0)
    local_indices = (received_experts - expert_offset).clamp(
        min=0, max=local_expert_count - 1
    )
    local_outputs = _masked_local_swiglu(
        module.experts,
        received_states,
        local_indices,
        local_expert_count,
    )
    local_outputs = local_outputs * valid.unsqueeze(-1)
    combined_outputs = _exchange_equal_capacity(
        local_outputs, routed_count, ep_size, ep_group
    )[slots]

    weighted_outputs = combined_outputs * expert_weights.unsqueeze(-1)
    output = flat_states.new_zeros((token_count, hidden_size))
    output.index_add_(0, source_indices, weighted_outputs)
    return output.view(batch_size, sequence_length, hidden_size)


def _find_dynamic_ep_modules(
    model: torch.nn.Module,
    ep_degree: int,
) -> list[torch.nn.Module]:
    if isinstance(ep_degree, bool) or not isinstance(ep_degree, int) or ep_degree < 1:
        raise ValueError("ep_degree must be a positive integer")
    modules = []
    seen_experts = set()
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or not hasattr(experts, "local_expert_count"):
            continue
        if id(experts) in seen_experts:
            continue
        global_count = getattr(experts, "num_experts", None)
        if global_count is not None:
            is_divisible = global_count % ep_degree == 0
            local_count_matches = (
                is_divisible
                and global_count // ep_degree == experts.local_expert_count
            )
            if not local_count_matches:
                raise ValueError(
                    "AutoModels EP expert sharding does not match ep_degree"
                )
        modules.append(module)
        seen_experts.add(id(experts))
    return modules


@contextmanager
def capture_dynamic_ep(
    model: torch.nn.Module, ep_degree: int
) -> Generator[EPCaptureMetadata, None, None]:
    """Mark a dynamic-EP model for static execution while tracing it.

    Args:
        model: AutoModels module after its dynamic EP sharding plan is applied.
        ep_degree: Expected number of ranks in the EP group.

    Yields:
        Metadata populated while static EP execution is active.

    Raises:
        ValueError: If the model does not contain matching dynamic EP metadata.
    """
    ep_modules = _find_dynamic_ep_modules(model, ep_degree)
    if not ep_modules:
        raise ValueError(
            "EP graph capture requires an AutoModels model with the dynamic EP "
            "sharding plan already applied"
        )
    metadata = EPCaptureMetadata(ep_degree=ep_degree)
    previous_metadata = []
    try:
        for module in ep_modules:
            previous_metadata.append(getattr(module.experts, "_ep_capture_metadata", None))
            module.experts._ep_capture_metadata = metadata
        yield metadata
    finally:
        for module, previous in reversed(list(zip(ep_modules, previous_metadata))):
            if previous is None:
                del module.experts._ep_capture_metadata
            else:
                module.experts._ep_capture_metadata = previous


__all__ = ["EPCaptureMetadata", "capture_dynamic_ep", "static_ep_routed_forward"]
