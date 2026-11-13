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
"""Prepared PIK execution with fixed forward arithmetic and first-order gradients."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from rl.unified_execution.batch_invariant import initialize_batch_invariant, normalize_policy
from .gemm import local_gemm
from .plan import PikPlan
from .collectives import combine_partials, gather_sequence_gradient


@dataclass(frozen=True)
class PikContext:
    """Caller-owned static metadata; never caches weights, activations or process groups globally.

    All participating ranks prepare together, then call with compatible shapes and ordering.
    None group means single-rank execution even inside a larger distributed job.
    """

    plan: PikPlan
    global_k: int
    output_features: int
    group: Any
    world_size: int
    rank: int
    device_index: int
    sequence_parallel: bool = False
    sequence_dim: int = 0
    transport: str = "all_gather"


def _preparation_contract(plan, global_k, world, output_features, sequence_parallel, sequence_dim, transport,
                          policy, group) -> dict:
    """Validate local preparation before every rank joins contract agreement."""
    initialize_batch_invariant(normalize_policy(policy or {}))
    if not isinstance(plan, PikPlan):
        raise ValueError("plan must be PikPlan")
    plan.validate(global_k, world)
    if isinstance(output_features, bool) or not isinstance(output_features, int) or output_features < 1:
        raise ValueError("output_features must be a positive integer")
    if (not isinstance(sequence_parallel, bool) or isinstance(sequence_dim, bool)
            or not isinstance(sequence_dim, int)):
        raise ValueError("sequence_parallel must be bool and sequence_dim must be int")
    if transport not in ("all_gather", "all_to_all"):
        raise ValueError("transport must be all_gather or all_to_all")
    if group is not None:
        backends = {entry.strip() for entry in dist.get_backend_config(group).lower().split(",")}
        if dist.get_backend(group) != "hccl" and "npu:hccl" not in backends:
            raise ValueError("PIK requires an HCCL backend for NPU tensors")
    return {**plan.contract(), "global_k": global_k, "n": output_features,
                "sp": sequence_parallel, "axis": sequence_dim, "transport": transport,
                "policy": normalize_policy(policy or {})}


def prepare_pik(global_k: int, output_features: int, *, plan: PikPlan | None = None,
                group: Any = None, sequence_parallel: bool = False, sequence_dim: int = 0,
                transport: str = "all_gather", policy: dict | None = None) -> PikContext:
    """Initialize Level 3 and agree on the static contract outside compilation/capture.

    Args:
        global_k: Full K before TP sharding; ranks own consecutive K intervals.
        output_features: Complete output feature count N.
        plan: Fixed global leaves and BMM physical layout.
        group: Explicit caller-owned HCCL TP group; None means TP1.
        sequence_parallel: Return the rank's contiguous sequence-axis slice.
        sequence_dim: Input token axis to scatter, before flattening.
        transport: all_gather reference or all_to_all; no shape-dependent auto selection.
        policy: Existing shared Level 3 policy, including memory-fill choice.

    Returns:
        Immutable context passed to pik_linear. No collective occurs in its constructor.

    Raises:
        ValueError: Any rank rejects the contract, or ranks disagree.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("prepare_pik must run before compilation")
    if group is not None and not dist.is_initialized():
        raise ValueError("explicit group requires initialized distributed runtime")
    world = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    if world < 1 or rank < 0:
        raise ValueError("caller must belong to the TP group")
    device = torch.device("npu", torch.npu.current_device())
    plan = PikPlan() if plan is None else plan
    error, contract = None, {}
    try:
        contract = _preparation_contract(plan, global_k, world, output_features, sequence_parallel,
                                         sequence_dim, transport, policy, group)
    except (ValueError, RuntimeError) as exception:
        error = exception
    if world > 1:
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).digest()
        control = torch.tensor([int(error is not None), *digest], device=device, dtype=torch.int32)
        received = torch.empty(world * control.numel(), device=device, dtype=torch.int32)
        dist.all_gather_into_tensor(received, control, group=group)
        records = received.reshape(world, -1).cpu().tolist()
        if any(row[0] for row in records):
            detail = str(error) if error is not None else "peer validation failed"
            raise ValueError(f"PIK preparation rejected by a TP rank: {detail}") from error
        if any(row != records[0] for row in records[1:]):
            raise ValueError("PIK static contracts differ across TP ranks")
    if error is not None:
        raise error
    return PikContext(plan, global_k, output_features, group, world, rank, device.index,
                      sequence_parallel, sequence_dim, transport)


def _validate_input_tensors(x, weight, context) -> None:
    """Check local tensor rank, device and dtype before validating sharded geometry."""
    if x.ndim < 2 or weight.ndim != 2 or any(size < 1 for size in x.shape):
        raise ValueError("PIK expects nonempty [...,K_local] input and [N,K_local] weight")
    if x.device.type != "npu" or x.device.index != context.device_index or weight.device != x.device:
        raise ValueError("PIK tensors must reside on the prepared NPU")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("PIK inputs and weights must be BF16")


def _validate(x, weight, bias, context):
    """Check prepared local geometry and return the token axis."""
    _validate_input_tensors(x, weight, context)
    local_k = context.global_k // context.world_size
    if x.shape[-1] != local_k or tuple(weight.shape) != (context.output_features, local_k):
        raise ValueError("PIK local shapes disagree with the prepared global shape")
    axis = context.sequence_dim % x.ndim
    if not -x.ndim <= context.sequence_dim < x.ndim or axis == x.ndim - 1:
        raise ValueError("sequence_dim must select a token axis")
    if context.sequence_parallel and x.shape[axis] % context.world_size:
        raise ValueError("SP sequence length must be divisible by TP; pad before PIK")
    if context.transport == "all_to_all" and math.prod(x.shape[:-1]) % context.world_size:
        raise ValueError("AllToAll requires flattened token rows divisible by TP")
    if bias is not None and (bias.device != x.device or tuple(bias.shape) != (context.output_features,)
                             or bias.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("bias must be [N] BF16/FP32 on the input NPU")
    return axis


def _forward(x, weight, bias, context, axis):
    """Compute local leaves, combine rank roots, then cast and add bias."""
    ordered = x.movedim(axis, 0)
    shape = list(ordered.shape[:-1])
    flat = ordered.reshape(-1, ordered.shape[-1])
    partial = local_gemm(flat, weight, plan=context.plan,
                         local_leaves=context.plan.num_leaves // context.world_size)
    root = combine_partials(partial, context.group, context.world_size, context.rank,
                            scatter=context.sequence_parallel, transport=context.transport)
    output = root.to(torch.bfloat16)
    if bias is not None:
        output = output + bias.to(torch.bfloat16)
    if context.sequence_parallel:
        shape[0] //= context.world_size
    return output.reshape(*shape, weight.shape[0]).movedim(0, axis).contiguous()


class _PikLinear(torch.autograd.Function):
    """Standard first-order linear gradients through a fixed, rounded forward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                context: PikContext, axis: int) -> torch.Tensor:
        """Save original local tensors so callers retain parameter/optimizer ownership."""
        ctx.save_for_backward(x, weight)
        ctx.context, ctx.axis = context, axis
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return _forward(x, weight, bias, context, axis)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Gather disjoint SP dY, then evaluate conventional local dX/dW/dB."""
        x, weight = ctx.saved_tensors
        context = ctx.context
        if context.sequence_parallel and context.world_size > 1:
            gradient = gather_sequence_gradient(gradient, context.group, context.world_size, ctx.axis)
        dy = gradient.reshape(-1, weight.shape[0]).contiguous()
        flat_x = x.reshape(-1, x.shape[-1])
        dx = (dy @ weight).reshape(x.shape) if ctx.needs_input_grad[0] else None
        dw = dy.T @ flat_x if ctx.needs_input_grad[1] else None
        db = dy.sum(0).to(ctx.bias_dtype) if ctx.needs_input_grad[2] else None
        return dx, dw, db, None, None


def pik_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
               *, context: PikContext) -> torch.Tensor:
    """Compute fixed-tree X@W.T using local K shards and a prepared static context.

    SP returns this rank's slice along context.sequence_dim. Its backward gathers dY;
    replicated output backward expects the same full dY on TP ranks. Bias gradient is
    already complete and must not be summed across TP again. Gradients follow usual
    first-order rounded-linear semantics; no cross-TP gradient-bit guarantee is made.
    All ranks must call with compatible shapes and ordering; no CPU agreement occurs
    here. Prepare, policy checks, kernel warmup and process-group lifecycle are outside
    capture. The operator does not own or cache caller weights.
    """
    axis = _validate(x, weight, bias, context)
    return _PikLinear.apply(x, weight, bias, context, axis)
