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
"""Packing helpers for fully_shard communication buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from hyper_parallel.core.dtensor.placement_types import StridedShard


@dataclass(frozen=True)
class ReduceScatterPlan:
    """Describe how local tensors map to packed communication layouts."""

    pack_kind: Literal[
        "identity_dim0",
        "same_dim_strided_identity_dim0",
        "chunk_cat_non_dim0",
    ]
    shard_dim: int
    world_size: int
    packed_shape: torch.Size
    packed_tensor_shape: torch.Size
    unpacked_shape: torch.Size


@dataclass(frozen=True)
class _SameDimStridedLayoutContext:
    target_dim: int
    shard_mesh_dim: int
    placements: tuple[Any, ...]
    orig_placements: tuple[Any, ...]


def _has_strided_shard_layout(hsdp_param: Any) -> bool:
    placements = getattr(hsdp_param, "_spmd_placements", ()) or ()
    return any(isinstance(placement, StridedShard) for placement in placements)


def _resolve_same_dim_strided_context(
    hsdp_param: Any,
) -> Optional[_SameDimStridedLayoutContext]:
    if not _has_strided_shard_layout(hsdp_param):
        return None
    if not getattr(hsdp_param, "uses_param_shard", False):
        return None
    if not getattr(hsdp_param, "_orig_param_is_dtensor", False):
        return None
    target_dim = getattr(getattr(hsdp_param, "hsdp_placement", None), "dim", None)
    if target_dim is None:
        return None
    shard_mesh_dim = getattr(hsdp_param, "_spmd_shard_mesh_dim", None)
    placements = tuple(getattr(hsdp_param, "_spmd_placements", ()) or ())
    if shard_mesh_dim is None or shard_mesh_dim >= len(placements):
        return None
    if not isinstance(placements[shard_mesh_dim], StridedShard):
        return None
    orig_placements = getattr(hsdp_param, "_orig_dtensor_placements", None)
    if orig_placements is None:
        return None
    return _SameDimStridedLayoutContext(
        target_dim=target_dim,
        shard_mesh_dim=shard_mesh_dim,
        placements=placements,
        orig_placements=tuple(orig_placements),
    )


def _placements_match_target_dim_only(
    placements: tuple[Any, ...],
    target_dim: int,
) -> bool:
    return all(
        placement.is_replicate() or placement.is_shard(target_dim)
        for placement in placements
    )


def _orig_layout_is_supported(
    orig_placements: tuple[Any, ...],
    target_dim: int,
) -> bool:
    if not _placements_match_target_dim_only(orig_placements, target_dim):
        return False
    return sum(
        placement.is_shard(target_dim) for placement in orig_placements
    ) == 1


def _current_strided_layout_is_supported(
    placements: tuple[Any, ...],
    target_dim: int,
) -> bool:
    if not _placements_match_target_dim_only(placements, target_dim):
        return False
    if sum(placement.is_shard() for placement in placements) != 2:
        return False

    strided_placements = [
        placement for placement in placements if isinstance(placement, StridedShard)
    ]
    if len(strided_placements) != 1:
        return False
    strided_placement = strided_placements[0]
    if strided_placement.dim != target_dim or strided_placement.split_factor <= 1:
        return False

    plain_shards = [
        placement
        for placement in placements
        if placement.is_shard(target_dim) and not isinstance(placement, StridedShard)
    ]
    return len(plain_shards) == 1


def supports_same_dim_strided_layout(hsdp_param: Any) -> bool:
    """Check whether the parameter's StridedShard layout is supported for same-dim packing."""
    ctx = _resolve_same_dim_strided_context(hsdp_param)
    if ctx is None:
        return False
    if not _orig_layout_is_supported(ctx.orig_placements, ctx.target_dim):
        return False
    return _current_strided_layout_is_supported(ctx.placements, ctx.target_dim)


def _resolve_unpacked_shape(
    hsdp_param: Optional[Any],
    local_tensor: torch.Tensor,
) -> torch.Size:
    if hsdp_param is not None and getattr(hsdp_param, "_orig_size", None) is not None:
        return torch.Size(getattr(hsdp_param, "_orig_size"))
    return torch.Size(local_tensor.size())


def _get_packed_tensor_shape(
    unpacked_shape: torch.Size,
    shard_dim: int,
    world_size: int,
) -> torch.Size:
    if world_size == 1 or shard_dim == 0:
        return unpacked_shape
    packed_tensor_shape = list(unpacked_shape)
    packed_tensor_shape[0] *= world_size
    packed_tensor_shape[shard_dim] //= world_size
    return torch.Size(packed_tensor_shape)


def build_rs_plan(
    hsdp_param: Optional[Any],
    local_tensor: torch.Tensor,
    world_size: int,
    *,
    shard_dim: Optional[int] = None,
) -> ReduceScatterPlan:
    """Build the V1 reduce-scatter packing plan for a local gradient tensor."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}")

    resolved_shard_dim = getattr(getattr(hsdp_param, "hsdp_placement", None), "dim", shard_dim)
    if resolved_shard_dim is None:
        raise ValueError("build_rs_plan requires either hsdp_param or shard_dim")
    unpacked_shape = _resolve_unpacked_shape(hsdp_param, local_tensor)
    if resolved_shard_dim < 0 or resolved_shard_dim >= len(unpacked_shape):
        raise ValueError(
            f"Invalid shard dim {resolved_shard_dim} for tensor shape {tuple(unpacked_shape)}"
        )
    if world_size == 1:
        if not local_tensor.is_contiguous():
            raise NotImplementedError(
                "reduce_scatter_grad currently expects contiguous local gradients before packing."
            )
        return ReduceScatterPlan(
            pack_kind="identity_dim0",
            shard_dim=resolved_shard_dim,
            world_size=world_size,
            packed_shape=torch.Size((1, math.prod(unpacked_shape))),
            packed_tensor_shape=unpacked_shape,
            unpacked_shape=unpacked_shape,
        )
    if local_tensor.dim() == 0:
        raise NotImplementedError("reduce_scatter_grad does not support scalar gradients.")
    if unpacked_shape[resolved_shard_dim] % world_size != 0:
        raise NotImplementedError(
            f"reduce_scatter_grad currently only supports even sharding on dim={resolved_shard_dim}."
        )
    if not local_tensor.is_contiguous():
        raise NotImplementedError(
            "reduce_scatter_grad currently expects contiguous local gradients before packing."
        )

    pack_kind: Literal[
        "identity_dim0",
        "same_dim_strided_identity_dim0",
        "chunk_cat_non_dim0",
    ] = "identity_dim0"
    if hsdp_param is not None and _has_strided_shard_layout(hsdp_param):
        if not supports_same_dim_strided_layout(hsdp_param):
            raise NotImplementedError(
                "reduce_scatter_grad only supports same-dim StridedShard layouts "
                "that restore a single contiguous TP-local shard on the fully_shard dimension."
            )
        if resolved_shard_dim == 0:
            pack_kind = "same_dim_strided_identity_dim0"
        else:
            pack_kind = "chunk_cat_non_dim0"
    elif resolved_shard_dim != 0:
        pack_kind = "chunk_cat_non_dim0"

    packed_tensor_shape = _get_packed_tensor_shape(
        unpacked_shape,
        resolved_shard_dim,
        world_size,
    )
    total_numel = math.prod(unpacked_shape)

    return ReduceScatterPlan(
        pack_kind=pack_kind,
        shard_dim=resolved_shard_dim,
        world_size=world_size,
        packed_shape=torch.Size((world_size, total_numel // world_size)),
        packed_tensor_shape=packed_tensor_shape,
        unpacked_shape=unpacked_shape,
    )


def pack_for_reduce_scatter(
    local_tensor: torch.Tensor,
    plan: ReduceScatterPlan,
) -> torch.Tensor:
    """Pack one local gradient into the row-major reduce-scatter layout."""

    if plan.pack_kind not in (
        "identity_dim0",
        "same_dim_strided_identity_dim0",
        "chunk_cat_non_dim0",
    ):
        raise NotImplementedError(f"Unsupported reduce-scatter pack kind: {plan.pack_kind}")
    if not local_tensor.is_contiguous():
        raise NotImplementedError(
            "reduce_scatter_grad currently expects contiguous local gradients before packing."
        )
    if local_tensor.size() != plan.unpacked_shape:
        raise AssertionError(
            "pack_for_reduce_scatter expects the unsharded local tensor shape to match "
            f"plan.unpacked_shape, but got {tuple(local_tensor.size())} and "
            f"{tuple(plan.unpacked_shape)}"
        )
    if plan.pack_kind == "chunk_cat_non_dim0":
        chunks = torch.chunk(local_tensor, plan.world_size, dim=plan.shard_dim)
        packed_tensor = torch.cat(chunks, dim=0)
        return packed_tensor.contiguous().view(plan.packed_shape)
    return local_tensor.view(plan.packed_shape)


def unpack_from_all_gather(
    full_packed: torch.Tensor,
    plan: ReduceScatterPlan,
) -> torch.Tensor:
    """Inverse of the V1 reduce-scatter packing plan for all-gather outputs."""

    if plan.pack_kind not in (
        "identity_dim0",
        "same_dim_strided_identity_dim0",
        "chunk_cat_non_dim0",
    ):
        raise NotImplementedError(f"Unsupported all-gather unpack kind: {plan.pack_kind}")
    packed_tensor = full_packed.view(plan.packed_tensor_shape)
    if plan.pack_kind == "chunk_cat_non_dim0":
        chunks = torch.chunk(packed_tensor, plan.world_size, dim=0)
        return torch.cat(chunks, dim=plan.shard_dim).contiguous()
    return packed_tensor.view(plan.unpacked_shape)


__all__ = [
    "ReduceScatterPlan",
    "build_rs_plan",
    "pack_for_reduce_scatter",
    "unpack_from_all_gather",
    "supports_same_dim_strided_layout",
]
