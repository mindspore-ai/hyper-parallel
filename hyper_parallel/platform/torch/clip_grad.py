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
"""Distributed-aware gradient clipping for parallel training.

Communication is driven by each parameter's DTensorSpec (device_mesh +
placements) rather than any specific parallelism strategy, so a single
implementation covers FSDP, HSDP, TP+FSDP, and other DTensor-expressed
parallelisms.

Design aligned with FSDP1 (``fully_sharded_data_parallel.py``):

* Gradient norms from sharded parameters are all-reduced across the
  corresponding shard process group.
* Non-sharded / replicated norms contribute locally without communication.
* **All ranks participate in the same collectives** regardless of local
  gradient availability, preventing collective-misalignment deadlocks.

Note: PP does not use DTensor layout for gradients today.  Cross-stage
norm aggregation will require an additional manual all-reduce and is
left for future work.
"""
import functools
import math
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.placement_types import Partial

try:
    from torch.utils._foreach_utils import (
        _device_has_foreach_support,
        _group_tensors_by_device_and_dtype,
        _has_foreach_support,
    )
except ImportError:
    _device_has_foreach_support = None  # type: ignore[assignment]
    _group_tensors_by_device_and_dtype = None  # type: ignore[assignment]
    _has_foreach_support = None  # type: ignore[assignment]

__all__: list[str] = ["clip_grad_norm_"]

# (id(mesh) or None, shard_dims) -> list of local grads for norm computation
_GradGroupKey = Tuple[Optional[int], Tuple[int, ...]]

# (mesh_dim_index, dist.ReduceOp)
_PartialReduceInfo = Tuple[int, "dist.ReduceOp"]


# ---------------------------------------------------------------------------
# Reduce-op mapping
# ---------------------------------------------------------------------------

_STR_TO_REDUCE_OP: Dict[str, "dist.ReduceOp"] = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
}


def _str_to_reduce_op(op_str: str) -> "dist.ReduceOp":
    """Map a ``Partial`` placement's *reduce_op* string to ``dist.ReduceOp``."""
    op = _STR_TO_REDUCE_OP.get(op_str.lower())
    if op is None:
        raise ValueError(
            f"Unsupported Partial reduce_op: {op_str!r}. "
            f"Supported: {list(_STR_TO_REDUCE_OP)}"
        )
    return op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_parameters(
    parameters: Union["torch.nn.Module", torch.Tensor, Iterable[torch.Tensor]],
) -> List[torch.Tensor]:
    """Normalize *parameters* to a flat list of tensors.

    * ``torch.nn.Module``  -> ``list(module.parameters())``
    * single ``torch.Tensor`` -> ``[tensor]``
    * iterable of tensors    -> ``list(iterable)``
    """
    if isinstance(parameters, torch.nn.Module):
        return list(parameters.parameters())
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


def _param_device(param: torch.Tensor) -> torch.device:
    """Return the local device of *param* (unwrap DTensor if needed)."""
    if isinstance(param, DTensor):
        return param._local_tensor.device  # pylint: disable=protected-access
    return param.device


def _get_local_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Return the local gradient tensor, or ``None`` if absent.

    If the gradient is a DTensor, returns its ``_local_tensor``.
    """
    if not param.requires_grad:
        return None
    grad = param.grad
    if grad is None:
        return None
    if isinstance(grad, DTensor):
        return grad._local_tensor  # pylint: disable=protected-access
    return grad


def _get_param_mesh_info(
    param: torch.nn.Parameter,
) -> Tuple[
    Optional[object],
    Tuple[int, ...],
    Tuple[_PartialReduceInfo, ...],
]:
    """Derive DeviceMesh, Shard dims and Partial info from DTensorSpec.

    Checks the *gradient's* spec first; falls back to the *parameter's*
    spec when the gradient is a plain tensor on a DTensor parameter
    (common after FSDP/HSDP backward where ``param.grad`` is stored as
    the local shard tensor).

    Returns ``(mesh, shard_dims, partial_info)`` where *partial_info*
    is a tuple of ``(mesh_dim, dist.ReduceOp)`` pairs that respect the
    ``Partial`` placement's ``reduce_op`` attribute.
    """
    grad = param.grad
    # Prefer grad's spec (most accurate); fall back to param's.
    spec_source = grad if isinstance(grad, DTensor) else param
    if not isinstance(spec_source, DTensor):
        return None, (), ()

    shard_dims = tuple(
        i for i, p in enumerate(spec_source.placements)
        if p.is_shard()
    )
    partial_info = tuple(
        (i, _str_to_reduce_op(p.reduce_op))
        for i, p in enumerate(spec_source.placements)
        if isinstance(p, Partial)
    )
    return spec_source.device_mesh, shard_dims, partial_info


def _compute_local_norm(
    grads: List[torch.Tensor],
    norm_type: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute the combined norm of *grads* locally in FP32.

    When *grads* is empty, returns the **identity element** for the
    subsequent all-reduce so that this rank contributes a neutral value
    (aligned with FSDP1's ``_zero_scalar`` approach):

    * ``inf``   -> 0    (neutral for MAX; norms are non-negative)
    * ``-inf``  -> +inf (neutral for MIN)
    * ``0``     -> 0    (neutral for SUM)
    * finite    -> 0    (neutral for SUM)
    """
    if not grads:
        if norm_type == -math.inf:
            return torch.tensor(
                float("inf"), device=device, dtype=torch.float32,
            )
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    if norm_type == math.inf:
        norms = [
            torch.linalg.vector_norm(
                g.detach(), math.inf, dtype=torch.float32,
            )
            for g in grads
        ]
        return torch.stack(norms).max().to(device)

    if norm_type == -math.inf:
        norms = [
            torch.linalg.vector_norm(
                g.detach(), -math.inf, dtype=torch.float32,
            )
            for g in grads
        ]
        return torch.stack(norms).min().to(device)

    if norm_type == 0:
        norms = [
            torch.linalg.vector_norm(
                g.detach(), 0, dtype=torch.float32,
            )
            for g in grads
        ]
        return torch.stack(norms).sum().to(device)

    # Finite p-norm: return sum of p-th powers.
    norms = [
        torch.linalg.vector_norm(
            g.detach(), norm_type, dtype=torch.float32,
        )
        for g in grads
    ]
    norm_powers = [n.to(device=device) ** norm_type for n in norms]
    return torch.stack(norm_powers).sum()


# ---------------------------------------------------------------------------
# Total norm aggregation with collectives
# ---------------------------------------------------------------------------

def _get_total_norm(
    grad_groups: Dict[_GradGroupKey, List[torch.Tensor]],
    norm_type: float,
    mesh_cache: Dict[int, object],
    device: torch.device,
) -> torch.Tensor:
    """Compute total gradient norm with per-group all-reduce.

    Each group shares the same ``(mesh, shard_dims)`` signature.  For
    every ``Shard`` dimension we issue one all-reduce on the
    corresponding mesh process group.  Groups with **no gradients**
    still participate in the collective (contributing an identity
    element) to prevent collective misalignment across ranks.

    * Pure FSDP  -- one group, one shard dim, one all-reduce.
    * HSDP       -- Replicate dim is ignored, Shard dim is reduced.
    * TP + FSDP  -- two Shard dims, two sequential all-reduces.
    * Replicated -- no all-reduce at all.
    """
    if norm_type == math.inf:
        return _total_norm_inf(
            grad_groups, norm_type, mesh_cache, device,
            dist.ReduceOp.MAX,
        )

    if norm_type == -math.inf:
        return _total_norm_inf(
            grad_groups, norm_type, mesh_cache, device,
            dist.ReduceOp.MIN,
        )

    if norm_type == 0:
        return _total_norm_sum(
            grad_groups, norm_type, mesh_cache, device,
        )

    # Finite p-norm.
    total_p = _total_norm_sum(
        grad_groups, norm_type, mesh_cache, device,
    )
    return total_p ** (1.0 / norm_type)


def _total_norm_inf(  # pylint: disable=R0913,R0917
    grad_groups, norm_type, mesh_cache, device, reduce_op,
):
    """Shared logic for inf / -inf norms."""
    group_norms: List[torch.Tensor] = []
    for (mesh_id, shard_dims), grads in grad_groups.items():
        local_norm = _compute_local_norm(grads, norm_type, device)
        if mesh_id is not None:
            mesh = mesh_cache[mesh_id]
            for dim in shard_dims:
                dist.all_reduce(
                    local_norm, op=reduce_op,
                    group=mesh.get_group(dim),
                )
        group_norms.append(local_norm)
    if not group_norms:
        if norm_type == -math.inf:
            return torch.tensor(float("inf"), device=device)
        return torch.tensor(0.0, device=device)
    stacked = torch.stack(group_norms)
    return stacked.max() if reduce_op == dist.ReduceOp.MAX else stacked.min()


def _total_norm_sum(grad_groups, norm_type, mesh_cache, device):
    """Shared logic for finite norms and L0 (all use SUM all-reduce)."""
    total = torch.tensor(0.0, device=device)
    for (mesh_id, shard_dims), grads in grad_groups.items():
        local_val = _compute_local_norm(grads, norm_type, device)
        if mesh_id is not None:
            mesh = mesh_cache[mesh_id]
            for dim in shard_dims:
                dist.all_reduce(
                    local_val, op=dist.ReduceOp.SUM,
                    group=mesh.get_group(dim),
                )
        total = total + local_val
    return total


def _participate_partial_zero(
    param: torch.Tensor,
    mesh: Optional[object],
    partial_info: Tuple[_PartialReduceInfo, ...],
) -> None:
    """Join Partial all-reduce with a zero tensor for a grad-free param.

    Frozen params (``requires_grad=False``) are consistently grad-free
    across all ranks, so zero participation is unnecessary — this avoids
    per-param zero all-reduces in fine-tuning / param-freezing scenarios.

    Trainable params with transient ``grad=None`` (e.g. unused in this
    forward) may differ across ranks, so we must participate to match the
    Partial all-reduce that other ranks enter.
    """
    if not param.requires_grad or mesh is None or not partial_info:
        return
    local_p = (
        param._local_tensor  # pylint: disable=W0212
        if isinstance(param, DTensor) else param.data
    )
    zero = torch.zeros_like(local_p)
    for pdim, reduce_op in partial_info:
        dist.all_reduce(zero, op=reduce_op, group=mesh.get_group(pdim))


def _pre_reduce_partial(
    local_grad: torch.Tensor,
    mesh: Optional[object],
    partial_info: Tuple[_PartialReduceInfo, ...],
) -> torch.Tensor:
    """Pre-reduce a Partial gradient via all-reduce for norm computation.

    Returns a clone with the reduced values when Partial placements
    exist, so the original ``local_grad`` is not mutated (the clip step
    later operates on the original via scalar multiplication which
    distributes over the reduction).
    """
    if mesh is None or not partial_info:
        return local_grad
    norm_grad = local_grad.clone()
    for pdim, reduce_op in partial_info:
        dist.all_reduce(
            norm_grad, op=reduce_op, group=mesh.get_group(pdim),
        )
    return norm_grad


def _build_grad_groups(
    params: List[torch.Tensor],
) -> Tuple[
    Dict[_GradGroupKey, List[torch.Tensor]],
    List[torch.Tensor],
    Dict[int, object],
    torch.device,
]:
    """Classify parameters into grad groups and pre-reduce Partial grads.

    Group structure is derived from *parameter* DTensorSpecs (always
    present on every rank) rather than gradients (which may be ``None``
    on some ranks).  This ensures every rank enters the same set of
    collectives, preventing deadlocks (aligned with FSDP1 where all
    ranks unconditionally execute the same all-reduce path).

    Returns ``(grad_groups, all_grads, mesh_cache, device)``.
    """
    grad_groups: Dict[_GradGroupKey, List[torch.Tensor]] = defaultdict(list)
    all_grads: List[torch.Tensor] = []
    mesh_cache: Dict[int, object] = {}
    device: Optional[torch.device] = None

    for param in params:
        mesh, shard_dims, partial_info = _get_param_mesh_info(param)

        key: _GradGroupKey = (
            id(mesh) if mesh is not None else None, shard_dims,
        )
        if mesh is not None:
            mesh_cache[id(mesh)] = mesh

        if device is None:
            device = _param_device(param)

        local_grad = _get_local_grad(param)
        if local_grad is None:
            if key not in grad_groups:
                grad_groups[key] = []
            _participate_partial_zero(param, mesh, partial_info)
            continue

        all_grads.append(local_grad)
        norm_grad = _pre_reduce_partial(local_grad, mesh, partial_info)
        grad_groups[key].append(norm_grad)

    if device is None:
        device = torch.device("cpu")

    return grad_groups, all_grads, mesh_cache, device


def _clip_grads_with_norm_(
    all_grads: List[torch.Tensor],
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    """Scale gradients in-place so the total norm <= *max_norm*.

    When *foreach* is ``True`` (or ``None`` on a supported device),
    uses ``torch._foreach_mul_`` grouped by (device, dtype) for
    better performance.
    """
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if _group_tensors_by_device_and_dtype is not None:
        grouped_grads = _group_tensors_by_device_and_dtype(
            [all_grads],
        )
        for (device, _), ([device_grads], _) in grouped_grads.items():
            if (
                foreach is None
                and _has_foreach_support(device_grads, device)
            ) or (
                foreach
                and _device_has_foreach_support(device)
            ):
                torch._foreach_mul_(  # pylint: disable=W0212
                    device_grads,
                    clip_coef_clamped.to(device),
                )
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the "
                    f"foreach API on {device.type} tensors"
                )
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_device)
    else:
        # Fallback when _foreach_utils is unavailable.
        if foreach:
            raise RuntimeError(
                "foreach=True was passed, but "
                "torch.utils._foreach_utils is not available"
            )
        for grad in all_grads:
            grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[
        "torch.nn.Module", torch.Tensor, Iterable[torch.Tensor],
    ],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    """Compute and clip gradient norm for distributed models.

    Drop-in replacement for the standard ``clip_grad_norm_`` that
    correctly handles DTensor-sharded parameters by deriving
    communication from each parameter's DTensorSpec.

    .. warning:: This function uses collective communications.  It
        **must be called on all ranks** to avoid deadlocks.  Aligned
        with FSDP1: every rank participates in the same collectives
        regardless of local gradient availability.

    Communication is derived from each parameter's DTensorSpec:

    * ``Shard`` on mesh dim *d* -- all-reduce norm statistics
      across ``device_mesh.get_group(d)``
    * ``Partial`` on mesh dim *d* -- all-reduce gradient values
      using the placement's ``reduce_op`` before norm computation
    * ``Replicate`` / plain tensor -- no communication

    This covers FSDP, HSDP, TP+FSDP, and any combination expressible
    via DTensor placements.  PP cross-stage norm aggregation is not
    yet handled (requires manual all-reduce across stages).

    Args:
        parameters: An ``nn.Module``, a single ``Tensor``, or an
            iterable of ``Tensor`` s whose gradients to clip.
        max_norm: Maximum allowed gradient norm.
        norm_type: Type of the norm (default ``2.0``).
        error_if_nonfinite: If ``True``, raise a ``RuntimeError``
            when the total norm is non-finite.  Default ``False``.
        foreach: Use the faster foreach-based implementation for the
            gradient clipping step.  If ``None``, use the foreach
            implementation for devices that support it and silently
            fall back to the per-tensor implementation for others.
            Default ``None``.

    Returns:
        The total (unclipped) gradient norm as a scalar tensor,
        cast to the promoted dtype of all gradient tensors.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    params = _normalize_parameters(parameters)
    grad_groups, all_grads, mesh_cache, device = _build_grad_groups(params)

    # -- Norm + clip (all ranks participate) --------------------------------
    # _compute_local_norm returns identity elements for empty groups,
    # so the subsequent all-reduce is safe and semantically neutral.
    total_norm = _get_total_norm(
        grad_groups, norm_type, mesh_cache, device,
    )

    if error_if_nonfinite and torch.logical_or(
        total_norm.isnan(), total_norm.isinf()
    ):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To "
            "disable this error and scale the gradients by the "
            "non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )

    if all_grads:
        _clip_grads_with_norm_(all_grads, max_norm, total_norm, foreach)

    # Promote return dtype to match gradient dtypes (FSDP1 convention).
    # When this rank has no gradients, return in the default FP32 dtype
    # (same as FSDP1's behavior to avoid extra communication).
    if not all_grads:
        warnings.warn(
            "clip_grad_norm_ called on this rank with no gradients -- "
            "returning the total norm in the default dtype "
            f"{total_norm.dtype}",
            stacklevel=2,
        )
        return total_norm

    total_norm_dtype = functools.reduce(
        torch.promote_types,
        [g.dtype for g in all_grads],
    )
    return total_norm.to(total_norm_dtype)
