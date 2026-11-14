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
"""Call-local source-token chunks for the common MoE training module.

The executor owns routing plans and explicit first-order backward, but owns no
parameters. EP uses the existing expert shards and differentiable collectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.dtensor._utils import differentiable_all_to_all_single
from hyper_parallel.core.expert_parallel.expert_parallel import AllToAllTokenDispatcher
from hyper_parallel.core.expert_parallel.row_permutation import permute_rows
from hyper_parallel.core.expert_parallel.static_splits import static_splits_enabled


@dataclass
class _Plan:
    indices: torch.Tensor
    order: torch.Tensor
    source_rows: torch.Tensor
    counts: torch.Tensor
    context: Any = None


def _local(parameter):
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _expert_mesh(module):
    weights = (module.experts.w1, module.experts.w2, module.experts.w3)
    if not any(isinstance(weight, DTensor) for weight in weights):
        return None
    if not all(isinstance(weight, DTensor) for weight in weights):
        raise ValueError("Chunk MoE requires all three expert weights to use the same EP layout.")
    mesh = weights[0].device_mesh
    for weight in weights:
        if (mesh.ndim != 1 or tuple(weight.placements) != (Shard(0),)
                or weight.device_mesh.rank_list != mesh.rank_list):
            raise ValueError("Chunk MoE requires one-dimensional expert Shard(0) execution parameters.")
    if list(mesh.rank_list) != dist.get_process_group_ranks(mesh.get_group()):
        raise ValueError("Chunk MoE requires mesh order to match EP process-group order.")
    return mesh


def _validate_input(module, hidden):
    if not isinstance(module.chunk_size, int) or isinstance(module.chunk_size, bool) or module.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if isinstance(hidden, DTensor) or hidden.ndim != 3 or not hidden.numel():
        raise ValueError("Chunk MoE requires a nonempty local [batch, sequence, hidden] tensor.")
    if hidden.device.type not in ("cpu", "npu") or torch.is_autocast_enabled(hidden.device.type):
        raise ValueError("Chunk MoE supports CPU reference or BF16 NPU training without autocast.")
    if not torch.is_grad_enabled():
        raise ValueError("Chunk MoE training requires grad mode; outer reentrant checkpoint is unsupported.")
    if module.score_before_experts:
        raise ValueError("Chunk MoE requires score_before_experts=False.")
    if hidden.shape[-1] != module.router.gate.in_features:
        raise ValueError("Chunk MoE hidden width must match the router.")
    if hidden.device.type == "npu" and (hidden.dtype != torch.bfloat16 or not module.experts.use_grouped_mm):
        raise ValueError("Chunk MoE on NPU requires BF16 and use_grouped_mm=True.")


def _validate_parameters(module, hidden, names, parameters, mesh):
    for name, parameter in zip(names, parameters):
        if parameter.dtype != hidden.dtype or parameter.device != hidden.device:
            raise ValueError("Chunk MoE parameters must match the input dtype and device.")
        original = module.get_parameter(name)
        if isinstance(original, DTensor) and name not in ("experts.w1", "experts.w2", "experts.w3"):
            raise ValueError("Chunk MoE requires unsharded local router/shared execution parameters.")
        if getattr(original, "main_grad", None) is not None:
            raise ValueError("Chunk MoE returns gradients through autograd; external main_grad is unsupported.")
    if any(getattr(child, "hsdp_scheduler", None) is not None
           for child in module.modules() if child is not module):
        raise ValueError("Chunk MoE requires FSDP on the whole MoE or its parent, not its bypassed children.")
    degree = mesh.size() if mesh is not None else 1
    local_counts = [_local(getattr(module.experts, name)).shape[0] for name in ("w1", "w2", "w3")]
    if module.num_experts % degree or any(count != module.num_experts // degree for count in local_counts):
        raise ValueError("Chunk MoE expert shards must evenly cover the configured global expert count.")


def _validate_groups(module, hidden, mesh):
    if mesh is not None and static_splits_enabled():
        raise ValueError("Chunk MoE cannot use HP_EP_STATIC_SPLITS: each chunk has a distinct routing plan.")
    signature = hidden.new_tensor(
        [hidden.numel() // hidden.shape[-1], module.chunk_size, module.num_experts, module.top_k], dtype=torch.int64,
    )
    groups = [mesh.get_group()] if mesh is not None else []
    if module.sequence_partition_group is not None and module.sequence_partition_group not in groups:
        groups.append(module.sequence_partition_group)
    for group in groups:
        copies = [torch.empty_like(signature) for _ in range(dist.get_world_size(group))]
        work = dist.all_gather(copies, signature, group=group, async_op=True)
        work.wait()
        if any(not torch.equal(signature, other) for other in copies):
            raise ValueError("Chunk MoE requires matching token counts, chunk sizes and routing across its groups.")


def _dispatch(values, plan, mesh):
    if plan.context is None:
        return values
    context = plan.context
    values = differentiable_all_to_all_single(
        values.contiguous(), context.input_splits, context.output_splits, group=mesh.get_group(),
    )
    return values.index_select(0, context.permuted_indices)


def _combine(values, plan, mesh):
    if plan.context is None:
        return values
    context = plan.context
    # Feature dimensions differ for dX and dp; only the row permutation is shared.
    rank_order = permute_rows(values, context.inverse_indices, context.permuted_indices)
    return differentiable_all_to_all_single(
        rank_order, context.output_splits, context.input_splits, group=mesh.get_group(),
    )


class _Experts:
    """Per-pass packed compute views and gradient sinks, never new Parameters."""

    def __init__(self, weights, reference, npu, backward=False):
        self.weights, self.reference, self.npu = weights, reference, npu
        self.packed = torch.cat((weights[0], weights[2]), dim=1) if npu else None
        self.gradients = [None] * 3
        self.packed_gradient = None
        if backward:
            for index, weight in enumerate(weights):
                if weight.requires_grad and (not npu or index == 1):
                    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
                    self.gradients[index] = torch.zeros_like(weight, dtype=dtype)
            if npu and (weights[0].requires_grad or weights[2].requires_grad):
                self.packed_gradient = torch.zeros_like(self.packed, dtype=torch.float32)

    def forward(self, values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Evaluate expert rows with the original grouped SwiGLU backend.

        Args:
            values: Expert-major received rows.
            counts: Number of rows assigned to each local expert.

        Returns:
            Expert outputs in the same row order."""
        if self.npu:
            if not values.shape[0]:
                return torch.empty_like(values)
            return npu_grouped_swiglu(values, self.packed, self.weights[1], counts)
        return self.reference(*self.weights, values, counts)

    def backward(self, values: torch.Tensor, scores: torch.Tensor, cotangent: torch.Tensor,
                 counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one block and accumulate expert gradients in private sinks.

        Args:
            values: Expert-major received input rows.
            scores: Selected routing probabilities for these rows.
            cotangent: Output gradient in expert-major order.
            counts: Number of rows assigned to each local expert.

        Returns:
            Input-row and selected-probability gradients."""
        if self.npu:
            # NPU/Triton are optional when importing the CPU reference module.
            from hyper_parallel.components.functional.npu_grouped_swiglu_vjp import (  # pylint: disable=C0415
                grouped_swiglu_vjp,
            )
            return grouped_swiglu_vjp(
                values, scores, cotangent, self.packed, self.weights[1], counts,
                self.packed_gradient, self.gradients[1],
            )
        selected = [index for index, weight in enumerate(self.weights) if weight.requires_grad]
        with torch.enable_grad():
            values = values.detach().requires_grad_()
            scores = scores.detach().requires_grad_()
            output = self.forward(values, counts) * scores[:, None]
            gradients = torch.autograd.grad(
                output, (values, scores, *(self.weights[index] for index in selected)), cotangent,
                allow_unused=True,
            )
        for index, gradient in zip(selected, gradients[2:]):
            if gradient is not None:
                self.gradients[index].add_(gradient)
        return gradients[:2]

    def finish(self) -> list[torch.Tensor | None]:
        """Return gradients in the original w1/w2/w3 parameter layout."""
        if self.packed_gradient is not None:
            for index, gradient in zip((0, 2), self.packed_gradient.chunk(2, dim=1)):
                if self.weights[index].requires_grad:
                    self.gradients[index] = gradient
        return self.gradients


def _forward_chunk(module, chunk, experts, mesh):
    scores, indices, counts = module.router(chunk, module.expert_bias)
    order = indices.reshape(-1).argsort(stable=True)
    plan = _Plan(indices, order, order // module.top_k, counts)
    received = chunk.index_select(0, plan.source_rows)
    if mesh is not None:
        dispatched = AllToAllTokenDispatcher.dispatch(module.experts, (received, counts), mesh)
        received, plan.counts = dispatched[:2]
        plan.context = dispatched[-1]
    values = _combine(experts.forward(received, plan.counts), plan, mesh)
    values = values * scores.to(values.dtype).reshape(-1)[order, None]
    output = torch.zeros_like(chunk).index_add_(0, plan.source_rows, values)
    if module.shared_expert is not None:
        output.add_(module.shared_expert(chunk))
    return output, scores, counts, plan


def _finish_statistics(module, counts, score_sums, tokens):
    module.tokens_per_expert.add_(counts)
    module.last_aux_loss = None
    if module.load_balance_coeff is None:
        return None
    global_counts = counts.clone()
    group = module.sequence_partition_group
    if group is not None:
        work = dist.all_reduce(global_counts, group=group, async_op=True)
        work.wait()
        tokens *= dist.get_world_size(group)
    weights = module.load_balance_coeff * module.num_experts * global_counts / (tokens * tokens * module.top_k)
    module.last_aux_loss = (weights * score_sums).sum().detach()
    return weights


def _backward_chunk(ctx, chunk, cotangent, plan, experts, parameters, selected):
    module = ctx.module
    with torch.enable_grad():
        scores = module.router.scores_for_selected(chunk, plan.indices)
        shared = module.shared_expert(chunk) if module.shared_expert is not None else None
    probabilities = scores.detach().to(chunk.dtype).reshape(-1)[plan.order]
    received = _dispatch(chunk.detach().index_select(0, plan.source_rows), plan, ctx.mesh)
    probabilities = _dispatch(probabilities, plan, ctx.mesh)
    received_gradient = _dispatch(cotangent.index_select(0, plan.source_rows), plan, ctx.mesh)
    grad_received, grad_probability = experts.backward(received, probabilities, received_gradient, plan.counts)
    del received, received_gradient, probabilities
    grad_rows = _combine(grad_received, plan, ctx.mesh)
    grad_input = torch.zeros_like(chunk).index_add_(0, plan.source_rows, grad_rows)
    del grad_received, grad_rows
    grad_probability = _combine(grad_probability, plan, ctx.mesh)
    grad_scores = torch.empty_like(scores).reshape(-1)
    grad_scores.index_copy_(0, plan.order, grad_probability.to(scores.dtype))
    grad_scores = grad_scores.view_as(scores)
    if ctx.aux_weights is not None:
        scale = ctx.aux_scaler.main_loss_backward_scale
        grad_scores = grad_scores + ctx.aux_weights[plan.indices] * (1.0 if scale is None else scale)
    outputs, gradients = [scores], [grad_scores]
    if shared is not None:
        outputs.append(shared)
        gradients.append(cotangent)
    values = torch.autograd.grad(
        outputs, (chunk, *(parameters[index] for index in selected)), gradients, allow_unused=True,
    )
    if values[0] is not None:
        grad_input.add_(values[0])
    return grad_input, values[1:]


def _accumulate_parameters(buffers, selected, gradients):
    for index, value in zip(selected, gradients):
        if value is not None:
            if buffers[index] is None:
                buffers[index] = value.clone() if value.dtype == torch.float64 else value.float()
            else:
                buffers[index].add_(value)


class _ChunkedMoE(torch.autograd.Function):
    """Retain local input and discrete plans, then consume one expert block at a time."""

    @staticmethod
    def forward(ctx: Any, configuration: tuple, hidden: torch.Tensor,
                *parameters: torch.Tensor) -> torch.Tensor:
        """Save one input plus discrete plans, without retaining expert activations.

        Args:
            ctx: Autograd context owning this invocation's plans.
            configuration: Module, expert reference, auxiliary scaler, mesh and expert parameter indices.
            hidden: Local input with batch, sequence and hidden dimensions.
            parameters: Execution parameter references, without additional copies.

        Returns:
            MoE output with the same shape as hidden."""
        module, reference, aux_scaler, mesh, expert_indices = configuration
        ctx.module, ctx.reference, ctx.aux_scaler, ctx.mesh = module, reference, aux_scaler, mesh
        ctx.expert_indices = expert_indices
        ctx.plans, ctx.consumed = [], False
        flat = hidden.reshape(-1, hidden.shape[-1])
        output = torch.empty_like(flat)
        experts = _Experts(tuple(parameters[index] for index in expert_indices), reference, hidden.device.type == "npu")
        counts = torch.zeros_like(module.tokens_per_expert, dtype=torch.float32)
        score_sums = torch.zeros_like(counts)
        for start in range(0, flat.shape[0], module.chunk_size):
            end = min(start + module.chunk_size, flat.shape[0])
            values, scores, local_counts, plan = _forward_chunk(module, flat[start:end], experts, mesh)
            output[start:end].copy_(values)
            counts.add_(local_counts.float())
            score_sums.scatter_add_(0, plan.indices.reshape(-1), scores.reshape(-1))
            ctx.plans.append((start, end, plan))
            del values, scores, local_counts, plan
        ctx.aux_weights = _finish_statistics(module, counts, score_sums, flat.shape[0])
        ctx.save_for_backward(hidden, *parameters)
        return output.reshape_as(hidden)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, cotangent: torch.Tensor) -> tuple:
        """Reverse chunks once and return complete input and parameter gradients.

        Args:
            ctx: Autograd context containing the input and unconsumed plans.
            cotangent: Gradient of the complete local MoE output.

        Returns:
            No configuration gradient, followed by input and parameter gradients."""
        if ctx.consumed:
            raise RuntimeError("Chunk MoE plans support one backward; run a new forward for another backward.")
        ctx.consumed = True
        hidden, *parameters = ctx.saved_tensors
        selected = [index for index, parameter in enumerate(parameters)
                    if parameter.requires_grad and index not in ctx.expert_indices]
        buffers = [None] * len(parameters)
        flat, cotangent = hidden.reshape(-1, hidden.shape[-1]), cotangent.reshape(-1, hidden.shape[-1])
        grad_hidden = torch.empty_like(flat) if ctx.needs_input_grad[1] else None
        experts = _Experts(tuple(parameters[index] for index in ctx.expert_indices), ctx.reference,
                           hidden.device.type == "npu", backward=True)
        while ctx.plans:
            start, end, plan = ctx.plans.pop()
            chunk = flat[start:end].detach().requires_grad_()
            grad_input, gradients = _backward_chunk(
                ctx, chunk, cotangent[start:end], plan, experts, parameters, selected,
            )
            if grad_hidden is not None:
                grad_hidden[start:end].copy_(grad_input)
            _accumulate_parameters(buffers, selected, gradients)
            del plan, chunk, grad_input, gradients
        for index, value in zip(ctx.expert_indices, experts.finish()):
            buffers[index] = value
        result = [value.to(parameter.dtype) if value is not None else None
                  for value, parameter in zip(buffers, parameters)]
        return None, grad_hidden.reshape_as(hidden) if grad_hidden is not None else None, *result


def chunked_moe(module: Any, hidden: torch.Tensor, reference: Callable, aux_scaler: Any) -> torch.Tensor:
    """Execute the common MoE's opt-in chunked training without replacing parameters.

    Args:
        module: Validated common MoE with positive chunk_size and score-after experts.
        hidden: Local, nonempty [batch, sequence, hidden] training input.
        reference: Common MoE's CPU expert reference function.
        aux_scaler: Common MoE's auxiliary gradient scaling owner.

    Returns:
        Full local output with complete first-order gradients. Plans are call-local
        and consumed by one backward. FSDP must own the whole MoE or a parent;
        independently wrapped expert children and outer MoE checkpoint are unsupported.
    """
    names, original = zip(*module.named_parameters())
    parameters = tuple(_local(parameter) for parameter in original)
    mesh = _expert_mesh(module)
    _validate_input(module, hidden)
    _validate_parameters(module, hidden, names, parameters, mesh)
    _validate_groups(module, hidden, mesh)
    indices = tuple(names.index(name) for name in ("experts.w1", "experts.w2", "experts.w3"))
    output = _ChunkedMoE.apply((module, reference, aux_scaler, mesh, indices), hidden, *parameters)
    if module.last_aux_loss is not None:
        output._load_balance_loss = module.last_aux_loss  # pylint: disable=protected-access
    return output
