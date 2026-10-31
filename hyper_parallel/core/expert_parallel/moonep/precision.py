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
"""Per-invocation native expert transport with continuous FP32 gradient return.

Only gradients use FP32 scratch. Forward projections and transfers preserve the
original parameter dtype; the scratch is allocated in backward and consumed by
one multi-output transport node, including zero gradients for unused experts.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional
import torch.distributed as dist

from hyper_parallel.core.utils.communication import differentiable_all_to_all_single


class GradientTarget:
    """Share a lazy gradient accumulator between one GMM and its owner exchange."""

    zero_out_wgrad = True

    def __init__(self, shape: tuple[int, ...], device: torch.device, requires_grad: bool = True) -> None:
        """Keep metadata only until a backward kernel needs its accumulator."""
        self.shape = shape
        self.skip_weight_gradient = not requires_grad
        self.device = device
        self.transpose = False
        self.grad_added_to_main_grad = False
        self.buffer: torch.Tensor | None = None

    def configure(self, shape: tuple[int, ...], transpose: bool) -> None:
        """Describe the compute projection layout without allocating a tensor."""
        if self.buffer is not None:
            raise RuntimeError('Cannot change an in-flight native gradient layout')
        self.shape = shape
        self.transpose = transpose

    @property
    def main_grad(self) -> torch.Tensor:
        """Allocate FP32 gradient storage only when the GMM backward consumes it."""
        if self.buffer is None:
            self.buffer = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        return self.buffer

    def gradient(self, fallback: torch.Tensor) -> torch.Tensor:
        """Return the current gradient in parameter layout without casting it down."""
        if self.buffer is None:
            return fallback.float()
        return self.buffer.transpose(-1, -2) if self.transpose else self.buffer

    def gmm_arguments(self, weight: torch.Tensor, *, transpose: bool = False) -> dict[str, Any]:
        """Expose HP's existing fused accumulator without a feature-specific import."""
        self.configure(tuple(weight.shape), transpose)
        return {'gemm_fusion': True, 'original_weight': self}

    def add_bias(self, inputs: torch.Tensor, bias: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Add expert biases while retaining this invocation's FP32 gradient target."""
        return _ExpertBias.apply(inputs, bias, counts, self)

    def release(self) -> None:
        """Release this backward's scratch so a retained graph starts from zero."""
        self.buffer = None


class _BorrowParameters(torch.autograd.Function):
    """Exchange all current expert parameters as one ordered autograd operation."""

    @staticmethod
    def forward(ctx: Any, copies: tuple[tuple[int, ...], ...], group: Any,
                targets: tuple[GradientTarget | None, ...], return_mask: tuple[bool, ...],
                *weights: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Send the original dtypes and return contiguous home-plus-borrowed views."""
        rank, ranks = dist.get_rank(group), dist.get_world_size(group)
        local = weights[0].shape[0]
        if len(copies) != ranks or any(len(row) != local for row in copies):
            raise ValueError('Native MoonEP copy extent must match EP and local experts')
        send = [[expert % local for expert in row if expert >= 0 and expert // local == rank]
                for row in copies]
        receive = [[slot for slot, expert in enumerate(copies[rank])
                    if expert >= 0 and expert // local == owner] for owner in range(ranks)]
        send_indices = torch.tensor([expert for row in send for expert in row],
                                    dtype=torch.int64, device=weights[0].device)
        receive_slots = torch.tensor([slot for row in receive for slot in row],
                                     dtype=torch.int64, device=weights[0].device)
        send_counts, receive_counts = [len(row) for row in send], [len(row) for row in receive]
        dtype_groups = tuple(tuple(index for index, weight in enumerate(weights) if weight.dtype == dtype)
                             for dtype in dict.fromkeys(weight.dtype for weight in weights))
        outputs: list[torch.Tensor | None] = [None] * len(weights)
        for indices in dtype_groups:
            packed = torch.cat([weights[index].index_select(0, send_indices).flatten(1)
                                for index in indices], dim=1)
            incoming = differentiable_all_to_all_single(packed, send_counts, receive_counts, group)
            offset = 0
            for index in indices:
                weight = weights[index]
                width = weight[0].numel()
                received = incoming[:, offset:offset + width].contiguous().reshape(
                    incoming.shape[0], *weight.shape[1:])
                borrowed = weight.new_zeros(weight.shape).index_copy(0, receive_slots, received)
                outputs[index] = torch.cat((weight, borrowed), dim=0)
                offset += width
        ctx.save_for_backward(send_indices, receive_slots)
        ctx.group, ctx.targets, ctx.local = group, targets, local
        ctx.return_mask = return_mask
        ctx.parameter_requires_grad = tuple(weight.requires_grad for weight in weights)
        ctx.send_counts, ctx.receive_counts, ctx.dtype_groups = send_counts, receive_counts, dtype_groups
        ctx.parameter_metadata = tuple((tuple(weight.shape), weight.dtype, weight.device) for weight in weights)
        ctx.set_materialize_grads(False)
        frozen = [output for output, weight in zip(outputs, weights) if not weight.requires_grad]
        if frozen:
            ctx.mark_non_differentiable(*frozen)
        return tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[Any, ...]:
        """Accumulate remote partials in FP32 and cast once at each original owner."""
        try:
            return _BorrowParameters._return_gradients(ctx, gradients)
        finally:
            for target in ctx.targets:
                if target is not None:
                    target.release()

    @staticmethod
    def _return_gradients(ctx: Any, gradients: tuple[torch.Tensor | None, ...]) -> tuple[Any, ...]:
        """Complete the owner exchange while the invocation owns its scratch."""
        send_indices, receive_slots = ctx.saved_tensors
        precise = []
        for gradient, target, metadata, needed in zip(
                gradients, ctx.targets, ctx.parameter_metadata, ctx.parameter_requires_grad):
            shape, _, device = metadata
            if not needed:
                precise.append(None)
            elif gradient is None:
                precise.append(torch.zeros((2 * shape[0], *shape[1:]), dtype=torch.float32, device=device))
            else:
                precise.append(gradient.float() if target is None else target.gradient(gradient))
        results: list[torch.Tensor | None] = [None] * len(gradients)
        for dtype_indices in ctx.dtype_groups:
            indices = tuple(index for index in dtype_indices if ctx.parameter_requires_grad[index])
            if not indices:
                continue
            packed = torch.cat([precise[index][ctx.local:].index_select(0, receive_slots).flatten(1)
                                for index in indices], dim=1)
            incoming = differentiable_all_to_all_single(
                packed, ctx.receive_counts, ctx.send_counts, ctx.group)
            offset = 0
            for index in indices:
                shape, dtype, _ = ctx.parameter_metadata[index]
                width = precise[index][0].numel()
                returned = incoming[:, offset:offset + width].reshape(incoming.shape[0], *shape[1:])
                home = precise[index][:ctx.local].clone(memory_format=torch.contiguous_format)
                home.index_add_(0, send_indices, returned)
                results[index] = home.to(dtype=dtype) if ctx.return_mask[index] else None
                offset += width
        return None, None, None, None, *results


def borrow_parameters(weights: tuple[torch.Tensor, ...], copies: tuple[tuple[int, ...], ...],
                      group: Any, return_mask: tuple[bool, ...] | None = None
                      ) -> tuple[tuple[torch.Tensor, ...], tuple[GradientTarget | None, ...]]:
    """Create each forward's parameter views and non-Tensor gradient descriptors."""
    targets = tuple(GradientTarget((2 * weight.shape[0], *weight.shape[1:]), weight.device, weight.requires_grad)
                    if weight.dtype in (torch.bfloat16, torch.float16) or not weight.requires_grad
                    else None for weight in weights)
    if return_mask is None:
        return_mask = tuple(weight.requires_grad for weight in weights)
    return _BorrowParameters.apply(copies, group, targets, return_mask, *weights), targets


class _ExpertLinear(torch.autograd.Function):
    """Retain original linear arithmetic and accumulate its expert dW before quantization."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor,
                target: GradientTarget, expert: int) -> torch.Tensor:
        """Run the same per-expert linear as the ordinary native implementation."""
        ctx.save_for_backward(inputs, weight)
        ctx.target, ctx.expert = target, expert
        return functional.linear(inputs, weight)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        """Keep dX arithmetic and accumulate the unquantized matrix gradient."""
        import torch_npu  # pylint: disable=C0415

        inputs, weight = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = functional.linear(gradient, weight.transpose(0, 1)) if ctx.needs_input_grad[0] else None
        if ctx.target.skip_weight_gradient:
            return dx, None, None, None
        if inputs.shape[0]:
            bounds = torch.tensor([inputs.shape[0]], dtype=torch.int64, device=inputs.device)
            torch_npu.npu_grouped_matmul_add_(
                ctx.target.main_grad[ctx.expert], inputs, gradient, bounds,
                transpose_x=True, transpose_weight=False, group_type=2)
        return dx, torch.zeros_like(weight), None, None


def expert_linear(inputs: torch.Tensor, weights: torch.Tensor, expert: int,
                  target: GradientTarget | None) -> torch.Tensor:
    """Select an expert without replacing the registered parameter or its FSDP hooks."""
    if target is None:
        return functional.linear(inputs, weights[expert])
    target.configure((weights.shape[0], weights.shape[2], weights.shape[1]), True)
    return _ExpertLinear.apply(inputs, weights[expert], target, expert)


class _ExpertBias(torch.autograd.Function):
    """Accumulate native expert bias gradients before their owner dtype conversion."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, bias: torch.Tensor,
                counts: torch.Tensor, target: GradientTarget) -> torch.Tensor:
        """Preserve the existing repeat-interleave and addition."""
        ctx.save_for_backward(counts)
        ctx.target, ctx.bias_shape = target, tuple(bias.shape)
        return inputs + torch.repeat_interleave(bias, counts, dim=0, output_size=inputs.shape[0])

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        """Reduce rows in FP32 into this invocation's lazy expert buffer."""
        dx = gradient if ctx.needs_input_grad[0] else None
        if ctx.target.skip_weight_gradient:
            return dx, None, None, None
        counts, = ctx.saved_tensors
        experts = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts, output_size=gradient.shape[0])
        ctx.target.main_grad.index_add_(0, experts, gradient.float())
        return dx, gradient.new_zeros(ctx.bias_shape), None, None
