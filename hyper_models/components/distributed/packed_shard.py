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
"""Packed-parameter ordering helpers used by TP apply and checkpoint paths."""

from typing import Sequence

import torch

from hyper_models.components.distributed.sharding_config import PackedShard


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Packed shard dim {dim} is invalid for a {ndim}D tensor")
    return normalized


def validate_packed_shape(tensor: torch.Tensor, placement: PackedShard, world_size: int) -> None:
    """Validate that every packed logical projection divides evenly over TP.

    Args:
        tensor: Full logical packed tensor.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Raises:
        ValueError: If the packed dimension cannot be split by parts and TP.
    """
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    dim = _normalize_dim(placement.dim, tensor.ndim)
    packed_size = tensor.shape[dim]
    if packed_size % placement.parts != 0:
        raise ValueError(
            f"Packed dimension size {packed_size} is not divisible by parts={placement.parts}"
        )
    logical_size = packed_size // placement.parts
    if logical_size % world_size != 0:
        raise ValueError(
            f"Packed logical projection size {logical_size} is not divisible by TP size {world_size}"
        )


def pack_tensor_for_shard(
    tensor: torch.Tensor,
    placement: PackedShard,
    world_size: int,
) -> torch.Tensor:
    """Reorder a logical packed tensor into contiguous rank-major shards.

    Args:
        tensor: Tensor ordered as ``[part0, part1, ...]`` on the packed dim.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Returns:
        Tensor ordered as ``[rank0(part0, part1), rank1(...), ...]``.
    """
    validate_packed_shape(tensor, placement, world_size)
    dim = _normalize_dim(placement.dim, tensor.ndim)
    logical_parts = tensor.chunk(placement.parts, dim=dim)
    rank_parts = [part.chunk(world_size, dim=dim) for part in logical_parts]
    ordered = [rank_parts[part_idx][rank] for rank in range(world_size) for part_idx in range(placement.parts)]
    return torch.cat(ordered, dim=dim).contiguous()


def unpack_tensor_from_shard(
    tensor: torch.Tensor,
    placement: PackedShard,
    world_size: int,
) -> torch.Tensor:
    """Restore logical part-major ordering from contiguous rank-major shards.

    Args:
        tensor: Gathered tensor in rank-major packed order.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Returns:
        Tensor restored to ``[part0, part1, ...]`` ordering.
    """
    validate_packed_shape(tensor, placement, world_size)
    dim = _normalize_dim(placement.dim, tensor.ndim)
    rank_chunks = tensor.chunk(world_size, dim=dim)
    rank_parts = [chunk.chunk(placement.parts, dim=dim) for chunk in rank_chunks]
    logical_parts = [
        torch.cat([rank_parts[rank][part_idx] for rank in range(world_size)], dim=dim)
        for part_idx in range(placement.parts)
    ]
    return torch.cat(logical_parts, dim=dim).contiguous()


def pack_tensor_for_placements(
    tensor: torch.Tensor,
    placements: Sequence[object],
    mesh,
) -> torch.Tensor:
    """Apply rank-major packing for every PackedShard in mesh-axis order."""
    result = tensor
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, PackedShard):
            result = pack_tensor_for_shard(result, placement, mesh.size(mesh_dim))
    return result


def unpack_tensor_from_placements(
    tensor: torch.Tensor,
    placements: Sequence[object],
    mesh,
) -> torch.Tensor:
    """Undo rank-major packing in reverse mesh-axis order."""
    result = tensor
    for mesh_dim in reversed(range(len(placements))):
        placement = placements[mesh_dim]
        if isinstance(placement, PackedShard):
            result = unpack_tensor_from_shard(result, placement, mesh.size(mesh_dim))
    return result
