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
"""AutoModels-private vocab-parallel cross entropy (local-first).

Merges the former ``components/loss/loss_parallel_ops.py`` kernel with the
``loss_parallel_ops_common.py`` validation helpers into a single private
production implementation. The leading underscore marks this module as
internal: the generic DTensor loss API stays in ``core.tensor_parallel`` /
``platform`` (frozen), while AutoModels production loss uses this module.

The public-in-package entry is :func:`vocab_parallel_cross_entropy_local`:
it takes the *local* logits shard plus the global vocabulary size, the mesh
and the mesh axis that shards the class dimension — no temporary DTensor is
created per call. The 2D mesh-axis recognition and the "class dimension
sharded by exactly one mesh axis" validation from the former AutoModels
DTensor wrapper are absorbed by :func:`_resolve_class_mesh_dim`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from hyper_parallel.platform import get_platform

if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

platform = get_platform()

__all__ = [
    "vocab_parallel_cross_entropy_local",
    "distributed_log_softmax",
    "distributed_nll_loss_forward",
    "DistributedCrossEntropyFunction",
]


def _validate_target_type_base(is_floating: bool) -> None:
    """Validate target type (base check).

    Args:
        is_floating: Whether target tensor has floating dtype.

    Raises:
        ValueError: If target is floating point (probabilistic).
    """
    if is_floating:
        raise ValueError(
            "Probabilistic target (float) is not supported in loss_parallel. "
            "Target must be class indices (int64)."
        )


def _resolve_class_mesh_dim(mesh: "DeviceMesh", mesh_dim: Optional[int]) -> int:
    """Identify the single mesh axis that shards the class dimension.

    Absorbs the 2D mesh-axis recognition of the former DTensor-based
    implementation: an explicit ``mesh_dim`` is validated against the mesh,
    a 1D mesh resolves to axis 0, and a multi-dimensional mesh resolves the
    axis named ``"tp"``. The class dimension must be sharded by exactly one
    mesh axis.

    Args:
        mesh: Mesh holding the class-sharding axis.
        mesh_dim: Explicitly declared class-sharding axis, or ``None`` to
            resolve it from the mesh shape and dimension names.

    Returns:
        The resolved mesh axis index.

    Raises:
        ValueError: If no unique class-sharding axis can be identified.
    """
    if mesh_dim is not None:
        if not isinstance(mesh_dim, int) or not 0 <= mesh_dim < mesh.ndim:
            raise ValueError(
                "Expected the class dimension to be sharded on exactly one "
                f"mesh dimension. Got mesh_dim={mesh_dim} for a "
                f"{mesh.ndim}D mesh."
            )
        return mesh_dim
    if mesh.ndim == 1:
        return 0
    names = mesh.mesh_dim_names or ()
    matches = [axis for axis, name in enumerate(names) if name == "tp"]
    if len(matches) != 1:
        raise ValueError(
            "Expected the class dimension to be sharded on exactly one mesh "
            f"dimension. Got {mesh.ndim}D mesh with dimension names {names}; "
            "pass mesh_dim explicitly or name the sharding axis 'tp'."
        )
    return matches[0]


def _is_floating_torch(tensor: Tensor) -> bool:
    """Check if PyTorch tensor is floating point."""
    return tensor.is_floating_point()


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
    max_local = logits_local.max(dim=dim, keepdim=True).values

    group = mesh.get_group(mesh_dim)
    max_global = platform.differentiable_all_reduce(max_local, op="max", group=group)

    exp_local = (logits_local - max_global).exp()

    sum_local = exp_local.sum(dim=dim, keepdim=True)

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
        loss = torch.zeros(batch_size, dtype=log_probs.dtype, device=log_probs.device)
    else:
        loss = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)

    total_weight = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)

    if target_mask.any():
        local_target = target_flat[target_mask] - vocab_start

        log_probs_2d = log_probs.reshape(-1, log_probs.shape[-1])

        row_indices = torch.where(target_mask)[0]

        selected_log_probs = log_probs_2d[row_indices, local_target]

        if weight is not None:
            global_target = target_flat[target_mask]
            sample_weights = weight[global_target]
            selected_log_probs = selected_log_probs * sample_weights
            total_weight = sample_weights.sum().reshape(1)
        else:
            total_weight = torch.tensor(
                target_mask.sum().item(), dtype=log_probs.dtype, device=log_probs.device
            ).reshape(1)

        nll = -selected_log_probs

        if reduction == "none":
            loss_flat = torch.zeros(batch_size, dtype=log_probs.dtype, device=log_probs.device)
            loss_flat[target_mask] = nll
            loss = loss_flat.reshape(target.shape)
        elif reduction == "sum":
            loss = nll.sum().unsqueeze(0)
        else:
            loss = nll.sum().unsqueeze(0)
    else:
        if reduction == "none":
            loss = torch.zeros(
                batch_size, dtype=log_probs.dtype, device=log_probs.device
            ).reshape(target.shape)
        total_weight = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)

    return loss, total_weight, target_mask, torch.tensor(
        vocab_start, dtype=torch.long, device=log_probs.device
    )


class DistributedCrossEntropyFunction(torch.autograd.Function):
    """K3: Fused backward for distributed cross_entropy."""

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

            if total_weight_sum.item() == 0:
                return torch.tensor(float('nan'), dtype=total_loss.dtype, device=total_loss.device)
            return total_loss / total_weight_sum
        if reduction == "sum":
            group = mesh.get_group(mesh_dim)
            total_loss = platform.differentiable_all_reduce(loss, op="sum", group=group)

            ctx.save_for_backward(
                log_probs_local,
                target,
                weight,
                torch.zeros(1, dtype=loss.dtype, device=loss.device),
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
            log_probs_local,
            target,
            weight,
            torch.zeros(1, dtype=loss.dtype, device=loss.device),
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

        local_targets = torch.where(in_vocab_mask, target_flat - vocab_start, torch.zeros_like(target_flat))

        if in_vocab_mask.any():
            row_indices = torch.arange(batch_size, device=target.device, dtype=torch.long)

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
            if reduction == "none":
                grad_input[~ignore_mask] = 0.0
            else:
                ignore_indices_expanded = (~ignore_mask).unsqueeze(-1).expand_as(grad_input)
                grad_input[ignore_indices_expanded] = 0.0

        return grad_input, None, None, None, None, None, None, None


def vocab_parallel_cross_entropy_local(
    local_logits: Tensor,
    target: Tensor,
    *,
    vocab_size: int,
    mesh: "DeviceMesh",
    mesh_dim: Optional[int] = None,
    ignore_index: int = -100,
    reduction: str = "sum",
) -> Tensor:
    """Vocab-parallel cross entropy on a local logits shard (local-first).

    AutoModels production entry: the caller owns the sharding contract and
    passes the local shard directly, so no temporary DTensor is created per
    call. ``mesh``/``mesh_dim`` identify the single mesh axis that shards the
    class dimension; see :func:`_resolve_class_mesh_dim`.

    Args:
        local_logits: Local class-sharded logits, ``[..., local_vocab_size]``.
            Rank above 2 is flattened to ``[-1, local_vocab_size]`` together
            with ``target``.
        target: Class indices aligned with the leading dims of
            ``local_logits``.
        vocab_size: Global vocabulary size (across all shard ranks).
        mesh: Mesh holding the class-sharding axis.
        mesh_dim: Explicit class-sharding mesh axis; ``None`` resolves it
            from the mesh (1D mesh → axis 0; otherwise the axis named "tp").
        ignore_index: Target value excluded from the loss.
        reduction: Reduction method: 'none', 'mean', or 'sum'.

    Returns:
        Loss tensor.

    Raises:
        ValueError: If the class-sharding mesh axis cannot be uniquely
            identified, the target is probabilistic, or the reduction is
            invalid.
    """
    mesh_dim = _resolve_class_mesh_dim(mesh, mesh_dim)
    _validate_target_type_base(_is_floating_torch(target))
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'.")

    local_vocab_size = local_logits.shape[-1]
    if local_logits.ndim > 2:
        local_logits = local_logits.reshape(-1, local_vocab_size)
        target = target.reshape(-1)

    return DistributedCrossEntropyFunction.apply(
        local_logits,
        target,
        None,
        ignore_index,
        reduction,
        vocab_size,
        mesh,
        mesh_dim,
    )
