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

from hyper_parallel.auto_models.components.distributed.sharding_config import PackedShard


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Packed shard dim {dim} is invalid for a {ndim}D tensor")
    return normalized


def validate_packed_shape(tensor: torch.Tensor, placement: PackedShard, world_size: int) -> None:
    """Validate packed sections against a tensor and TP size.

    Args:
        tensor: Full logical packed tensor.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Raises:
        ValueError: If sections do not cover the dimension or cannot be TP-sharded.
    """
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    dim = _normalize_dim(placement.dim, tensor.ndim)
    packed_size = tensor.shape[dim]
    if sum(placement.sections) != packed_size:
        raise ValueError(
            f"Packed sections {placement.sections} do not cover dimension size {packed_size}"
        )
    invalid = [size for size in placement.sections if size % world_size != 0]
    if invalid:
        raise ValueError(
            f"Packed section sizes {invalid} are not divisible by TP size {world_size}"
        )


def pack_tensor_for_shard(
    tensor: torch.Tensor,
    placement: PackedShard,
    world_size: int,
) -> torch.Tensor:
    """Reorder section-major data into contiguous rank-major shards.

    Args:
        tensor: Tensor ordered as ``[section0, section1, ...]``.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Returns:
        Tensor ordered as ``[rank0(section0, ...), rank1(...), ...]``.
    """
    validate_packed_shape(tensor, placement, world_size)
    dim = _normalize_dim(placement.dim, tensor.ndim)
    logical_sections = tensor.split(placement.sections, dim=dim)
    rank_sections = [section.chunk(world_size, dim=dim) for section in logical_sections]
    ordered = [
        rank_sections[section_index][rank]
        for rank in range(world_size)
        for section_index in range(len(placement.sections))
    ]
    return torch.cat(ordered, dim=dim).contiguous()


def unpack_tensor_from_shard(
    tensor: torch.Tensor,
    placement: PackedShard,
    world_size: int,
) -> torch.Tensor:
    """Restore section-major ordering from gathered rank-major data.

    Args:
        tensor: Gathered tensor in rank-major packed order.
        placement: Packed sharding rule.
        world_size: Tensor-parallel group size.

    Returns:
        Tensor restored to ``[section0, section1, ...]`` ordering.
    """
    validate_packed_shape(tensor, placement, world_size)
    dim = _normalize_dim(placement.dim, tensor.ndim)
    local_section_sizes = tuple(size // world_size for size in placement.sections)
    rank_size = sum(local_section_sizes)
    rank_chunks = tensor.split((rank_size,) * world_size, dim=dim)
    rank_sections = [chunk.split(local_section_sizes, dim=dim) for chunk in rank_chunks]
    logical_sections = [
        torch.cat([rank_sections[rank][section_index] for rank in range(world_size)], dim=dim)
        for section_index in range(len(placement.sections))
    ]
    return torch.cat(logical_sections, dim=dim).contiguous()


def pack_tensor_for_placements(tensor: torch.Tensor, placements: Sequence[object], mesh) -> torch.Tensor:
    """Apply rank-major packing for every PackedShard in mesh-axis order."""
    result = tensor
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, PackedShard):
            result = pack_tensor_for_shard(result, placement, mesh.size(mesh_dim))
    return result


def unpack_tensor_from_placements(tensor: torch.Tensor, placements: Sequence[object], mesh) -> torch.Tensor:
    """Undo rank-major packing in reverse mesh-axis order."""
    result = tensor
    for mesh_dim in reversed(range(len(placements))):
        placement = placements[mesh_dim]
        if isinstance(placement, PackedShard):
            result = unpack_tensor_from_shard(result, placement, mesh.size(mesh_dim))
    return result
