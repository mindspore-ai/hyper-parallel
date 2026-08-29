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
"""Derive Source Loader and Data Constructor ownership from a named mesh."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

_DEFAULT_DP_DIM_NAMES = ("dp_replicate", "dp_shard", "dp")


def _flatten_coordinate(coordinate: tuple[int, ...], shape: tuple[int, ...]) -> int:
    flat_index = 0
    for value, size in zip(coordinate, shape, strict=True):
        flat_index = flat_index * size + value
    return flat_index


def _unflatten_coordinate(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinate = [0] * len(shape)
    remaining = flat_index
    for dim_index in range(len(shape) - 1, -1, -1):
        coordinate[dim_index] = remaining % shape[dim_index]
        remaining //= shape[dim_index]
    return tuple(coordinate)


def _mesh_layout(mesh: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mesh_shape = getattr(mesh, "mesh_shape", None)
    rank_list = getattr(mesh, "rank_list", None)
    if mesh_shape is not None and rank_list is not None:
        return tuple(int(size) for size in mesh_shape), tuple(int(rank) for rank in rank_list)

    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is None:
        raise ValueError("mesh must expose mesh_shape/rank_list or a PyTorch mesh tensor.")
    return (
        tuple(int(size) for size in mesh_tensor.shape),
        tuple(int(rank) for rank in mesh_tensor.reshape(-1).tolist()),
    )


@dataclass(frozen=True)
class DataTopology:
    """DP constructors and their model-parallel consumer groups."""

    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    rank_list: tuple[int, ...]
    global_rank: int
    dp_dim_names: tuple[str, ...]
    coordinate: tuple[int, ...]
    data_rank: int
    data_parallel_size: int
    constructor_rank: int
    constructor_ranks: tuple[int, ...]
    model_parallel_rank_groups: tuple[tuple[int, ...], ...]
    model_parallel_ranks: tuple[int, ...]

    @classmethod
    def from_mesh(
            cls,
            mesh: Any,
            *,
            global_rank: int | None = None,
            dp_dim_names: tuple[str, ...] | None = None,
    ) -> "DataTopology":
        """Build topology from a named HyperParallel or PyTorch DeviceMesh.

        Args:
            mesh: Root training mesh with named dimensions.
            global_rank: Current global rank. Defaults to torch.distributed or
                rank zero in a standalone process.
            dp_dim_names: Mesh dimensions defining independent data consumers.

        Returns:
            Derived data topology.
        """
        mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
        if not mesh_dim_names:
            raise ValueError("Distributed data loading requires a mesh with named dimensions.")
        rank = global_rank
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        mesh_shape, rank_list = _mesh_layout(mesh)
        return cls.from_layout(
            mesh_shape,
            tuple(mesh_dim_names),
            rank_list,
            rank,
            dp_dim_names=dp_dim_names,
        )

    @classmethod
    def from_layout(
            cls,
            mesh_shape: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            rank_list: tuple[int, ...],
            global_rank: int,
            *,
            dp_dim_names: tuple[str, ...] | None = None,
    ) -> "DataTopology":
        """Build topology from a raw named layout.

        Args:
            mesh_shape: Size of every named mesh dimension.
            mesh_dim_names: Unique mesh dimension names.
            rank_list: Global ranks in row-major mesh order.
            global_rank: Rank for which ownership is derived.
            dp_dim_names: Dimensions that form the DP coordinate.

        Returns:
            Derived data topology.
        """
        cls._validate_layout(mesh_shape, mesh_dim_names, rank_list, global_rank)
        selected_dp_names = dp_dim_names
        if selected_dp_names is None:
            selected_dp_names = tuple(name for name in mesh_dim_names if name in _DEFAULT_DP_DIM_NAMES)
        if len(set(selected_dp_names)) != len(selected_dp_names):
            raise ValueError(f"dp_dim_names must be unique, but got {selected_dp_names}.")
        if any(name not in mesh_dim_names for name in selected_dp_names):
            raise ValueError(f"dp_dim_names {selected_dp_names} must be present in {mesh_dim_names}.")

        coordinate = _unflatten_coordinate(rank_list.index(global_rank), mesh_shape)
        dp_indices = tuple(mesh_dim_names.index(name) for name in selected_dp_names)
        dp_shape = tuple(mesh_shape[index] for index in dp_indices)
        dp_coordinate = tuple(coordinate[index] for index in dp_indices)
        data_parallel_size = math.prod(dp_shape) if dp_shape else 1
        data_rank = _flatten_coordinate(dp_coordinate, dp_shape) if dp_shape else 0
        constructor_ranks = cls._build_constructor_ranks(mesh_shape, rank_list, dp_indices, dp_shape)
        model_groups = cls._build_model_groups(mesh_shape, rank_list, dp_indices, dp_shape)
        return cls(
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
            rank_list=rank_list,
            global_rank=global_rank,
            dp_dim_names=selected_dp_names,
            coordinate=coordinate,
            data_rank=data_rank,
            data_parallel_size=data_parallel_size,
            constructor_rank=constructor_ranks[data_rank],
            constructor_ranks=constructor_ranks,
            model_parallel_rank_groups=model_groups,
            model_parallel_ranks=model_groups[data_rank],
        )

    @staticmethod
    def _validate_layout(
            mesh_shape: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            rank_list: tuple[int, ...],
            global_rank: int,
    ) -> None:
        invalid_sizes = any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in mesh_shape
        )
        if not mesh_shape or invalid_sizes:
            raise ValueError(f"mesh_shape must contain positive integers, but got {mesh_shape}.")
        if len(mesh_shape) != len(mesh_dim_names):
            raise ValueError("mesh_shape and mesh_dim_names must have the same length.")
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise ValueError(f"mesh_dim_names must be unique, but got {mesh_dim_names}.")
        if math.prod(mesh_shape) != len(rank_list) or len(set(rank_list)) != len(rank_list):
            raise ValueError("rank_list must uniquely fill every mesh position.")
        if global_rank not in rank_list:
            raise ValueError(f"global_rank {global_rank} is not present in rank_list {rank_list}.")

    @staticmethod
    def _build_constructor_ranks(
            mesh_shape: tuple[int, ...],
            rank_list: tuple[int, ...],
            dp_indices: tuple[int, ...],
            dp_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        data_parallel_size = math.prod(dp_shape) if dp_shape else 1
        constructors = []
        for data_rank in range(data_parallel_size):
            dp_coordinate = _unflatten_coordinate(data_rank, dp_shape) if dp_shape else ()
            coordinate = [0] * len(mesh_shape)
            for dim_index, value in zip(dp_indices, dp_coordinate, strict=True):
                coordinate[dim_index] = value
            constructors.append(rank_list[_flatten_coordinate(tuple(coordinate), mesh_shape)])
        return tuple(constructors)

    @staticmethod
    def _build_model_groups(
            mesh_shape: tuple[int, ...],
            rank_list: tuple[int, ...],
            dp_indices: tuple[int, ...],
            dp_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        data_parallel_size = math.prod(dp_shape) if dp_shape else 1
        groups: list[list[int]] = [[] for _ in range(data_parallel_size)]
        for flat_index, candidate_rank in enumerate(rank_list):
            coordinate = _unflatten_coordinate(flat_index, mesh_shape)
            dp_coordinate = tuple(coordinate[index] for index in dp_indices)
            data_rank = _flatten_coordinate(dp_coordinate, dp_shape) if dp_shape else 0
            groups[data_rank].append(candidate_rank)
        return tuple(tuple(group) for group in groups)

    @property
    def is_constructor(self) -> bool:
        """Return whether this rank constructs its DP coordinate's local batch."""
        return self.global_rank == self.constructor_rank

    @property
    def fingerprint(self) -> str:
        """Return a stable topology identifier for checkpoint validation."""
        layout = (self.mesh_shape, self.mesh_dim_names, self.rank_list, self.dp_dim_names)
        return hashlib.sha256(repr(layout).encode("utf-8")).hexdigest()[:24]
