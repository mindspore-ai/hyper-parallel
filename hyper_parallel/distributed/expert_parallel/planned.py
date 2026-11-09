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
from functools import lru_cache, partial
from typing import Any, Optional

# Native Torch communication and autograd are required by this execution boundary.
# pylint: disable=forbidden-backend-import
import torch
import torch.distributed as dist
from torch.utils.checkpoint import (
    CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts, set_checkpoint_early_stop,
)

from hyper_parallel.core.utils.communication import get_device_handle
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


@lru_cache(maxsize=None)
def _counts_copy_stream(device: torch.device) -> Any:
    """Reuse a device-specific transfer stream without caching invocation buffers."""
    return get_device_handle(device.type).Stream(device=device)


def _start_counts_copy(send_counts: torch.Tensor, receive_counts: torch.Tensor, work: Any) -> tuple:
    """Enqueue count D2H independently of sorting; retain both source and destination."""
    handle = get_device_handle(send_counts.device.type)
    stream = _counts_copy_stream(send_counts.device)
    stream.wait_stream(handle.current_stream())
    host = torch.empty((2, send_counts.numel()), dtype=send_counts.dtype, device="cpu", pin_memory=True)
    with handle.stream(stream):
        work.wait()
        device_counts = torch.stack((send_counts, receive_counts))
        host.copy_(device_counts, non_blocking=True)
        ready = handle.Event()
        ready.record(stream)
    # Keep original count buffers alive through the asynchronous stack as well.
    return host, ready, device_counts, send_counts, receive_counts


def build_ep_token_plan(topk_indices: torch.Tensor, local_expert_count: int, ep_group: Any,
                        *, with_inverse: bool = False, async_counts: bool = False) -> EPTokenPlan:
    """Plan lossless dispatch while overlapping count exchange with local sorting.

    Args:
        topk_indices: Current learned global expert indices, shaped [tokens, top_k].
        local_expert_count: Complete experts owned by each EP rank.
        ep_group: Process group with contiguous expert ownership, or None for EP1.
        with_inverse: Retain the inverse slot permutation required by native combine.
        async_counts: Enqueue pinned D2H on a side stream before independent sorting.
            A host wait remains at the first consumer of c10d split lists.

    Returns:
        Immutable route metadata. Count tensors use one host transfer, and expert
        indices are exchanged once even when the token payload is later replayed.
    """
    ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
    global_experts = local_expert_count * ep_size
    experts = topk_indices.reshape(-1)
    destinations = experts.div(local_expert_count, rounding_mode="floor")
    pending_counts = None
    if ep_group is not None:
        send_counts = _expert_token_counts(destinations, ep_size)
        receive_counts = torch.empty_like(send_counts)
        counts_work = dist.all_to_all_single(receive_counts, send_counts, group=ep_group, async_op=True)
        if async_counts and send_counts.device.type in ("cuda", "npu"):
            pending_counts = _start_counts_copy(send_counts, receive_counts, counts_work)
    order = _argsort_keys(destinations * global_experts + experts, bound=ep_size * global_experts)
    source_tokens = order.div(topk_indices.shape[1], rounding_mode="floor")
    inverse_order = None
    if with_inverse:
        inverse_order = torch.empty_like(order, dtype=torch.int32)
        inverse_order.scatter_(0, order, torch.arange(order.numel(), dtype=torch.int32, device=order.device))
    ordered_experts = experts[order].contiguous()
    if ep_group is None:
        # EP1 owns every slot; shape metadata supplies both counts without D2H.
        counts = ((experts.numel(),), (experts.numel(),))
    elif pending_counts is not None:
        # Never read pinned host values before the event; buffers live until here.
        pending_counts[1].synchronize()
        counts = pending_counts[0].tolist()
    else:
        counts_work.wait()
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


class _CombineTokens(torch.autograd.Function):
    """Retain payload precision while evaluating combination and probability VJP in FP32."""

    @staticmethod
    def forward(ctx: Any, payload: torch.Tensor, weights: torch.Tensor,
                source_tokens: torch.Tensor, slot_order: torch.Tensor) -> torch.Tensor:
        """Combine routed slots without retaining an expanded FP32 payload copy.

        Args:
            ctx: Per-invocation gradient dependencies.
            payload: Unweighted expert outputs in dispatch order.
            weights: FP32 routing probabilities in original token/slot order.
            source_tokens: Source token for each dispatched row.
            slot_order: Unique permutation from dispatch rows to original slots.
        """
        ctx.payload_dtype = payload.dtype
        ctx.weight_shape = weights.shape
        ctx.save_for_backward(payload if ctx.needs_input_grad[1] else None,
                              weights if ctx.needs_input_grad[0] else None,
                              source_tokens, slot_order)
        weighted = payload.float() * weights.reshape(-1)[slot_order, None]
        output = torch.zeros((weights.shape[0], payload.shape[-1]), dtype=torch.float32, device=payload.device)
        return output.index_add(0, source_tokens, weighted)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Compute only requested gradients, preserving FP32 probability reductions.

        Args:
            ctx: Per-invocation gradient dependencies.
            gradient: FP32 token-output gradient.
        """
        payload, weights, source_tokens, slot_order = ctx.saved_tensors
        slot_gradient = gradient[source_tokens].float()
        payload_gradient = None
        weight_gradient = None
        if ctx.needs_input_grad[0]:
            payload_gradient = (slot_gradient * weights.reshape(-1)[slot_order, None]).to(ctx.payload_dtype)
        if ctx.needs_input_grad[1]:
            slot_weights = (slot_gradient * payload.float()).sum(dim=-1)
            weight_gradient = torch.zeros(ctx.weight_shape, dtype=torch.float32, device=gradient.device)
            weight_gradient = weight_gradient.reshape(-1).index_copy(0, slot_order, slot_weights)
            weight_gradient = weight_gradient.reshape(ctx.weight_shape)
        return payload_gradient, weight_gradient, None, None


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
    return _CombineTokens.apply(combined, weights, plan.source_tokens, plan.slot_order)


def _retain_gemm_outputs(context: Any, operation: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    """Replay dispatch/elementwise work while reusing expensive matrix outputs."""
    del context, args, kwargs
    matrix_operations = ("npu.npu_grouped_matmul.default", "aten.mm.default", "aten.bmm.default", "aten.addmm.default")
    return CheckpointPolicy.MUST_SAVE if str(operation) in matrix_operations else CheckpointPolicy.PREFER_RECOMPUTE


def ep_planned_forward(module: Any, hidden_states: torch.Tensor, topk_indices: torch.Tensor,
                       topk_weights: torch.Tensor, ep_group: Any, *, payload_dtype: torch.dtype = torch.float32,
                       replay_routed: bool = False, combine_backend: str = "index_add",
                       async_counts: bool = False, replay_policy: str = "full") -> torch.Tensor:
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
        async_counts: Prepare c10d host splits on a side stream during local sorting.
        replay_policy: full replays all routed work. gemm_outputs retains matrix
            outputs and replays transport, permutations and elementwise operations.
            It saves more activations than full replay but avoids forward GEMM replay;
            it does not implement communication overlap with GEMM backward.

    Returns:
        FP32 routed output with the input shape. Shared experts and latent
        projections remain the caller's responsibility.

    Raises:
        ValueError: If shapes or payload/weight dtypes violate the execution contract.
    """
    if replay_policy not in ("full", "gemm_outputs"):
        raise ValueError("Planned EP replay_policy must be full or gemm_outputs")
    if payload_dtype not in (torch.float32, torch.bfloat16) or topk_weights.dtype != torch.float32:
        raise ValueError("Planned EP requires FP32 routing weights and FP32 or BF16 payloads")
    if (hidden_states.ndim != 3 or topk_indices.ndim != 2 or topk_indices.shape != topk_weights.shape
            or topk_indices.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]
            or topk_indices.shape[1] == 0):
        raise ValueError("Planned EP requires [batch, sequence, width] states and matching [tokens, top_k] routes")
    if combine_backend not in ("index_add", "npu"):
        raise ValueError("Planned EP combine must be index_add or npu")
    plan = build_ep_token_plan(topk_indices, module.experts.local_expert_count, ep_group,
                               with_inverse=combine_backend == "npu", async_counts=async_counts)
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
        options = ({"context_fn": partial(create_selective_checkpoint_contexts, _retain_gemm_outputs)}
                   if replay_policy == "gemm_outputs" else {})
        with set_checkpoint_early_stop(False):
            output = checkpoint(routed, flattened, topk_weights, use_reentrant=False, **options)
    else:
        output = routed(flattened, topk_weights)
    return output.reshape(hidden_states.shape)
