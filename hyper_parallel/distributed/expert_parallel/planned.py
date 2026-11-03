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
"""Lossless EP with one immutable route plan and optional routed-branch replay."""

import importlib
from dataclasses import dataclass
from typing import Any, Optional

# Native Torch communication and autograd are required by this execution boundary.
# pylint: disable=forbidden-backend-import
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys, _expert_token_counts


@dataclass(frozen=True)
class EPTokenPlan:
    """Per-forward integer metadata, retained unchanged across backward replay.

    Tensor storage belongs to this forward. Plans are neither cached across
    batches nor mutated by communication, so simultaneous autograd graphs
    retain independent routing decisions.
    """

    source_tokens: torch.Tensor
    slot_order: torch.Tensor
    local_expert_indices: torch.Tensor
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    token_count: int
    inverse_slot_order: Optional[torch.Tensor]


def build_ep_token_plan(topk_indices: torch.Tensor, local_expert_count: int, ep_group: Any,
                        *, with_inverse: bool = False) -> EPTokenPlan:
    """Plan lossless dispatch while overlapping count exchange with local sorting.

    Args:
        topk_indices: Current learned global expert indices, shaped [tokens, top_k].
        local_expert_count: Complete experts owned by each EP rank.
        ep_group: Process group with contiguous expert ownership, or None for EP1.
        with_inverse: Retain the inverse slot permutation required by native combine.

    Returns:
        Immutable route metadata. Count tensors use one host transfer, and expert
        indices are exchanged once even when the token payload is later replayed.
    """
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    global_experts = local_expert_count * ep_size
    experts = topk_indices.reshape(-1)
    destinations = experts.div(local_expert_count, rounding_mode="floor")
    send_counts = _expert_token_counts(destinations, ep_size)
    receive_counts = torch.empty_like(send_counts)
    counts_work = (dist.all_to_all_single(receive_counts, send_counts, group=ep_group, async_op=True)
                   if ep_group is not None else None)
    order = _argsort_keys(destinations * global_experts + experts, bound=ep_size * global_experts)
    source_tokens = order.div(topk_indices.shape[1], rounding_mode="floor")
    inverse_order = None
    if with_inverse:
        inverse_order = torch.empty_like(order, dtype=torch.int32)
        inverse_order.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
    ordered_experts = experts[order].contiguous()
    if counts_work is None:
        receive_counts.copy_(send_counts)
    else:
        counts_work.wait()
    # A single device-to-host transfer supplies both lists required by c10d.
    counts = torch.stack((send_counts, receive_counts)).cpu().tolist()
    if ep_group is None:
        received_experts = ordered_experts
    else:
        received_experts = torch.empty(sum(counts[1]), dtype=experts.dtype, device=experts.device)
        dist.all_to_all_single(received_experts, ordered_experts, counts[1], counts[0], group=ep_group)
        received_experts = received_experts - dist.get_rank(ep_group) * local_expert_count
    return EPTokenPlan(source_tokens, order, received_experts, tuple(counts[0]), tuple(counts[1]),
                       topk_indices.shape[0], inverse_order)


class _DispatchTokenGather(torch.autograd.Function):
    """Transport narrow payloads while accumulating duplicated token gradients in FP32."""

    @staticmethod
    def forward(ctx: Any, states: torch.Tensor, source_tokens: torch.Tensor,
                payload_dtype: torch.dtype) -> torch.Tensor:
        """Expand tokens in the exact dispatch order and retain only integer metadata.

        Args:
            ctx: Autograd context retaining the per-invocation state.
            states: Token states before expert dispatch.
            source_tokens: Source token indices in dispatch order.
            payload_dtype: Dtype used for dispatched token payloads.
        """
        ctx.save_for_backward(source_tokens)
        ctx.input_shape = states.shape
        ctx.input_dtype = states.dtype
        return states[source_tokens].to(payload_dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Sum each token's expert contributions before casting to its input dtype.

        Args:
            ctx: Autograd context retaining the per-invocation state.
            gradient: Upstream gradient for the returned output.
        """
        (source_tokens,) = ctx.saved_tensors
        grad_states = torch.zeros(ctx.input_shape, dtype=torch.float32, device=gradient.device)
        grad_states.index_add_(0, source_tokens, gradient.float())
        return grad_states.to(ctx.input_dtype), None, None


def _execute_plan(module: Any, states: torch.Tensor, weights: torch.Tensor, plan: EPTokenPlan,
                  ep_group: Any, payload_dtype: torch.dtype, combine_backend: str) -> torch.Tensor:
    """Run token communication and expert compute without executing the router."""
    dispatch = _DispatchTokenGather.apply(states, plan.source_tokens, payload_dtype)
    received = (ep_all_to_all(dispatch, list(plan.send_counts), list(plan.receive_counts), ep_group)
                if ep_group is not None else dispatch)
    # Enter through __call__ so nested FSDP parameter hooks retain ownership.
    expert_output = module.experts(received, plan.local_expert_indices)
    combined = expert_output.to(payload_dtype).contiguous()
    if ep_group is not None:
        combined = ep_all_to_all(combined, list(plan.receive_counts), list(plan.send_counts), ep_group)
    if combine_backend == "npu" and plan.token_count:
        # The optional native binding must not make generic EP imports require torch-npu.
        operation = importlib.import_module("hyper_parallel.components.functional.moe_token_unpermute")
        return operation.moe_token_unpermute(combined.float(), plan.inverse_slot_order, weights)
    weighted = combined.float() * weights.reshape(-1)[plan.slot_order, None]
    output = torch.zeros((plan.token_count, states.shape[-1]), dtype=torch.float32, device=states.device)
    return output.index_add(0, plan.source_tokens, weighted)


def ep_planned_forward(module: Any, hidden_states: torch.Tensor, topk_indices: torch.Tensor,
                       topk_weights: torch.Tensor, ep_group: Any, *, payload_dtype: torch.dtype = torch.float32,
                       replay_routed: bool = False, combine_backend: str = "index_add") -> torch.Tensor:
    """Execute learned EP routing with FP32 combination and optional activation replay.

    Args:
        module: Routed MoE exposing callable experts and local_expert_count.
        hidden_states: Local activations [batch, sequence, width].
        topk_indices: Global expert indices [batch * sequence, top_k].
        topk_weights: FP32 differentiable weights with the same shape as indices.
        ep_group: EP process group, or None for local EP1. Distributed ranks call
            forward and backward together.
        payload_dtype: float32 or bfloat16 communication payload. BF16 intentionally
            quantizes FP32 inputs; use it with BF16 expert computation.
        replay_routed: Save pre-dispatch input and weights, then replay only token
            exchanges and expert computation. The router and its statistics remain
            outside this boundary. All ranks must use the same setting.
        combine_backend: FP32 index-add reference or native NPU unpermutation.
            Native top-k accumulation can change floating-point addition order.

    Returns:
        FP32 routed output with the input shape. Shared experts and latent
        projections remain the caller's responsibility.

    Raises:
        ValueError: If shapes or payload/weight dtypes violate the execution contract.
    """
    if payload_dtype not in (torch.float32, torch.bfloat16) or topk_weights.dtype != torch.float32:
        raise ValueError("Planned EP requires FP32 routing weights and FP32 or BF16 payloads")
    if (hidden_states.ndim != 3 or topk_indices.ndim != 2 or topk_indices.shape != topk_weights.shape
            or topk_indices.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]
            or topk_indices.shape[1] == 0):
        raise ValueError("Planned EP requires [batch, sequence, width] states and matching [tokens, top_k] routes")
    if combine_backend not in ("index_add", "npu"):
        raise ValueError("Planned EP combine must be index_add or npu")
    plan = build_ep_token_plan(topk_indices, module.experts.local_expert_count, ep_group,
                               with_inverse=combine_backend == "npu")
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])

    def routed(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Replay the same immutable plan without revisiting learned selection.

        Args:
            states: Token states before expert dispatch.
            weights: Differentiable selected routing weights.
        """
        return _execute_plan(module, states, weights, plan, ep_group, payload_dtype, combine_backend)

    if replay_routed and torch.is_grad_enabled():
        # Early stopping can truncate different collectives on empty/nonempty ranks.
        with set_checkpoint_early_stop(False):
            output = checkpoint(routed, flattened, topk_weights, use_reentrant=False)
    else:
        output = routed(flattened, topk_weights)
    return output.reshape(hidden_states.shape)
