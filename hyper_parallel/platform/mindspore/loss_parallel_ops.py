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
"""MindSpore-specific loss_parallel operations.

Distributed cross-entropy kernel implementation using mindspore.common._grad_function._Function.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import mindspore as ms  # pylint: disable=C0415
from mindspore import mint  # pylint: disable=C0415
from mindspore.common._grad_function import _Function  # pylint: disable=C0415
from mindspore.common.tensor import Tensor  # pylint: disable=C0415

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.tensor_parallel.loss_parallel_ops_common import (
    _is_dtensor,
    _is_shard_on_last_dim,
    _get_mesh_and_dim,
    _get_local_tensor,
    _validate_cross_entropy_params,
    _check_context_and_layout,
    _validate_mesh_and_shard,
)
from hyper_parallel.core.tensor_parallel.loss_parallel import _get_loss_parallel_strict
from hyper_parallel.platform import get_platform

platform = get_platform()

__all__ = [
    "distributed_cross_entropy",
    "distributed_log_softmax",
    "distributed_nll_loss_forward",
    "DistributedCrossEntropyFunction",
]


def _is_floating_ms(tensor: Tensor) -> bool:
    """Check if MindSpore tensor is floating point."""
    return tensor.dtype in (ms.float16, ms.float32, ms.float64)


def _compute_vocab_start(vocab_size: int, tp_size: int, rank: int) -> int:
    """Compute the starting index for this rank's vocab shard.

    Args:
        vocab_size: Total vocabulary size.
        tp_size: Tensor parallel world size.
        rank: Current rank in TP mesh.

    Returns:
        Starting index of this rank's vocab shard.

    Note:
        This follows torch.chunk behavior: chunk_size = ceil(vocab_size/tp_size),
        and each rank's start = rank * chunk_size. The last rank may have fewer elements.
    """
    chunk_size = (vocab_size + tp_size - 1) // tp_size  # ceil division
    return rank * chunk_size


def distributed_log_softmax(
    logits_local: Tensor,
    dim: int,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
) -> Tensor:
    """K1: Stable log-softmax on class-sharded dimension.

    Args:
        logits_local: Local logits shard.
        dim: Class dimension.
        mesh: DeviceMesh.
        mesh_dim: Mesh dimension (default 0).

    Returns:
        Local log-softmax with unchanged layout.

    Communication:
        MAX + SUM all_reduce
    """
    max_local = logits_local.max(axis=dim, keepdims=True)

    group = mesh.get_group(mesh_dim)
    max_global = platform.differentiable_all_reduce(max_local, op="max", group=group)

    exp_local = (logits_local - max_global).exp()

    sum_local = exp_local.sum(axis=dim, keepdims=True)

    sum_global = platform.differentiable_all_reduce(sum_local, op="sum", group=group)

    log_softmax = logits_local - max_global - sum_global.log()

    return log_softmax


def distributed_nll_loss_forward(
    log_probs: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    ignore_index: int,
    reduction: str,
    vocab_start: int,
    vocab_end: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """K2: Index target + optional weight + reduction.

    Args:
        log_probs: Sharded log_probs.
        target: Target class indices.
        weight: Optional weights.
        ignore_index: Index to ignore.
        reduction: Reduction method.
        vocab_start: Start index of this vocab shard.
        vocab_end: End index of this vocab shard.

    Returns:
        Tuple of (loss, total_weight, target_mask, vocab_start_tensor).
    """
    batch_size = target.numel()

    target_flat = target.flatten()

    target_mask = (target_flat >= vocab_start) & (target_flat < vocab_end)

    ignore_mask = target_flat != ignore_index
    target_mask = target_mask & ignore_mask

    if reduction == "none":
        loss = mint.zeros((batch_size,), dtype=log_probs.dtype)
    else:
        loss = mint.zeros((1,), dtype=log_probs.dtype)

    total_weight = mint.zeros((1,), dtype=log_probs.dtype)

    if target_mask.any():
        local_target = target_flat[target_mask] - vocab_start

        log_probs_2d = log_probs.reshape(-1, log_probs.shape[-1])

        row_indices = mint.nonzero(target_mask).flatten()

        selected_log_probs = log_probs_2d[row_indices, local_target]

        if weight is not None:
            global_target = target_flat[target_mask]
            sample_weights = weight[global_target]
            selected_log_probs = selected_log_probs * sample_weights
            total_weight = sample_weights.sum().reshape(1)
        else:
            total_weight = ms.Tensor(
                target_mask.sum().asnumpy().item(), dtype=log_probs.dtype
            ).reshape(1)

        nll = -selected_log_probs

        if reduction == "none":
            loss_flat = mint.zeros((batch_size,), dtype=log_probs.dtype)
            loss_flat[target_mask] = nll
            loss = loss_flat.reshape(target.shape)
        elif reduction == "sum":
            loss = nll.sum().unsqueeze(0)
        else:
            loss = nll.sum().unsqueeze(0)
    else:
        if reduction == "none":
            loss = mint.zeros((batch_size,), dtype=log_probs.dtype).reshape(target.shape)
        total_weight = mint.zeros((1,), dtype=log_probs.dtype)

    vocab_start_tensor = ms.Tensor(vocab_start, dtype=ms.int64)
    return loss, total_weight, target_mask, vocab_start_tensor


class DistributedCrossEntropyFunction(_Function):
    """K3: Fused backward for distributed cross_entropy (MindSpore version)."""

    @staticmethod
    def forward(
        ctx: Any,
        input_local: Tensor,
        target: Tensor,
        weight: Optional[Tensor],
        ignore_index: int,
        reduction: str,
        vocab_size: int,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> Tensor:
        """Forward pass."""
        local_vocab_size = input_local.shape[-1]
        rank = mesh.get_local_rank(mesh_dim)
        tp_size = mesh.size(mesh_dim)
        vocab_start = _compute_vocab_start(vocab_size, tp_size, rank)
        vocab_end = vocab_start + local_vocab_size

        log_probs_local = distributed_log_softmax(
            input_local, dim=-1, mesh=mesh, mesh_dim=mesh_dim
        )

        loss, total_weight, target_mask, vocab_start_tensor = distributed_nll_loss_forward(
            log_probs_local,
            target,
            weight,
            ignore_index,
            reduction,
            vocab_start,
            vocab_end,
        )

        if reduction == "mean":
            group = mesh.get_group(mesh_dim)
            total_loss = platform.differentiable_all_reduce(loss, op="sum", group=group)
            total_weight_sum = platform.differentiable_all_reduce(
                total_weight, op="sum", group=group
            )

            ctx.save_for_backward(
                input_local,
                log_probs_local,
                target,
                weight,
                total_weight_sum,
                target_mask,
                vocab_start_tensor,
            )
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index
            ctx.vocab_size = vocab_size
            ctx.local_vocab_size = local_vocab_size
            ctx.mesh = mesh
            ctx.mesh_dim = mesh_dim
            ctx.vocab_start = vocab_start
            ctx.vocab_end = vocab_end

            return total_loss / total_weight_sum.clamp(min=1e-12)
        if reduction == "sum":
            group = mesh.get_group(mesh_dim)
            total_loss = platform.differentiable_all_reduce(loss, op="sum", group=group)

            ctx.save_for_backward(
                input_local,
                log_probs_local,
                target,
                weight,
                mint.zeros((1,), dtype=loss.dtype),
                target_mask,
                vocab_start_tensor,
            )
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index
            ctx.vocab_size = vocab_size
            ctx.local_vocab_size = local_vocab_size
            ctx.mesh = mesh
            ctx.mesh_dim = mesh_dim
            ctx.vocab_start = vocab_start
            ctx.vocab_end = vocab_end

            return total_loss
        ctx.save_for_backward(
            input_local,
            log_probs_local,
            target,
            weight,
            mint.zeros((1,), dtype=loss.dtype),
            target_mask,
            vocab_start_tensor,
        )
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.vocab_size = vocab_size
        ctx.local_vocab_size = local_vocab_size
        ctx.mesh = mesh
        ctx.mesh_dim = mesh_dim
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end

        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass (vectorized implementation)."""
        (
            _,
            log_probs_local,
            target,
            weight,
            total_weight,
            _,
            _,
        ) = ctx.saved_tensors

        reduction = ctx.reduction
        ignore_index = ctx.ignore_index
        _ = ctx.local_vocab_size
        vocab_start = ctx.vocab_start
        vocab_end = ctx.vocab_end
        _ = ctx.mesh
        _ = ctx.mesh_dim

        batch_size = target.numel()
        target_flat = target.flatten()

        softmax_local = log_probs_local.exp()

        ignore_mask = target_flat != ignore_index

        if weight is not None:
            sample_weights = weight[target_flat]
        else:
            sample_weights = None

        if reduction == "mean":
            grad_scale = grad_output / total_weight.clamp(min=1e-12)
        elif reduction == "sum":
            grad_scale = grad_output
        else:
            grad_scale = grad_output.flatten()

        in_vocab_mask = (target_flat >= vocab_start) & (target_flat < vocab_end) & ignore_mask

        if reduction == "none":
            grad_scale_expanded = grad_scale.unsqueeze(-1)
            if sample_weights is not None:
                grad_scale_expanded = grad_scale_expanded * sample_weights.unsqueeze(-1)
            grad_input = softmax_local * grad_scale_expanded
        else:
            if sample_weights is not None:
                grad_scale = grad_scale * sample_weights.unsqueeze(-1)
            grad_input = softmax_local * grad_scale.unsqueeze(-1)

        local_targets = mint.where(in_vocab_mask, target_flat - vocab_start, mint.zeros_like(target_flat))

        if in_vocab_mask.any():
            row_indices = mint.arange(batch_size, dtype=ms.int64)

            if reduction == "none":
                if sample_weights is not None:
                    grad_values = -grad_scale * sample_weights
                else:
                    grad_values = -grad_scale
            else:
                grad_values = -grad_scale.expand_as(target_flat)

            grad_input = grad_input.contiguous()
            grad_input[row_indices[in_vocab_mask], local_targets[in_vocab_mask]] += grad_values[in_vocab_mask]

        if not ignore_mask.all():
            ignore_indices = ~ignore_mask
            if reduction == "none":
                grad_input[ignore_indices] = 0.0
            else:
                ignore_indices_expanded = ignore_indices.unsqueeze(-1).expand_as(grad_input)
                grad_input = mint.where(ignore_indices_expanded, grad_input, mint.zeros_like(grad_input))

        return grad_input, None, None, None, None, None, None, None


def distributed_cross_entropy(
    input_tensor: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Distributed cross_entropy main entry (MindSpore version).

    Args:
        input_tensor: Must be DTensor with Shard(-1) on class dimension.
        target: Class indices with Replicate or consistent layout.
        weight: Optional weights; if DTensor, must be Replicate.
        size_average: Deprecated.
        ignore_index: Index to ignore (default -100).
        reduce: Deprecated.
        reduction: Reduction method: 'none', 'mean', or 'sum'.
        label_smoothing: Not supported (must be 0.0).

    Returns:
        Loss tensor.
    """
    input_dtensor = None
    mesh = None
    vocab_size = None

    if _is_dtensor(input_tensor):
        if not _is_shard_on_last_dim(input_tensor):
            raise ValueError(
                "input must be Shard(-1) on class dimension. "
                f"Got placements: {input_tensor.placements}"
            )
        input_dtensor = input_tensor
        mesh, _ = _get_mesh_and_dim(input_tensor)
        vocab_size = input_tensor.shape[-1]

    input_for_check = input_dtensor if input_dtensor is not None else input_tensor
    _check_context_and_layout(input_for_check)  # type: ignore

    _validate_cross_entropy_params(
        input_tensor,
        target,
        weight,
        size_average,
        ignore_index,
        reduce,
        reduction,
        label_smoothing,
        _is_floating_ms,
    )

    if input_dtensor is not None:
        input_local = _get_local_tensor(input_dtensor)
        local_vocab_size = input_local.shape[-1]

        if input_dtensor.ndim > 2:
            input_local = input_local.reshape(-1, local_vocab_size)
            target = target.reshape(-1)
    else:
        raise ValueError(
            "input must be a DTensor when using loss_parallel. "
            f"Got type: {type(input_tensor)}"
        )

    strict = _get_loss_parallel_strict()
    _validate_mesh_and_shard(input_dtensor, strict)  # type: ignore

    mesh_dim = 0

    loss = DistributedCrossEntropyFunction.apply(
        input_local,
        target,
        weight,
        ignore_index,
        reduction,
        vocab_size,
        mesh,
        mesh_dim,
    )

    return loss


def distributed_cross_entropy_from_op_call(
    op_call: Any,  # pylint: disable=W0613
    args: tuple,
    kwargs: dict,
) -> Tensor:
    """Parse arguments from op_call and invoke distributed cross_entropy.

    Used for OpDispatcher routing.

    Args:
        op_call: Op call object (reserved for future use).
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Loss tensor.
    """
    input_tensor = args[0] if len(args) > 0 else kwargs.get("input")
    target = args[1] if len(args) > 1 else kwargs.get("target")
    weight = args[2] if len(args) > 2 else kwargs.get("weight")
    size_average = args[3] if len(args) > 3 else kwargs.get("size_average")
    ignore_index = args[4] if len(args) > 4 else kwargs.get("ignore_index", -100)
    reduce = args[5] if len(args) > 5 else kwargs.get("reduce")
    reduction = args[6] if len(args) > 6 else kwargs.get("reduction", "mean")
    label_smoothing = args[7] if len(args) > 7 else kwargs.get("label_smoothing", 0.0)

    return distributed_cross_entropy(
        input_tensor=input_tensor,
        target=target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
