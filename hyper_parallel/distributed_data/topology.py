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
"""Topology derivation for distributed data loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hyper_parallel.platform import get_platform

platform = get_platform()

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


@dataclass(frozen=True)
class DataTopology:
    """Rank ownership and consumer groups derived from a named root mesh."""

    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    rank_list: tuple[int, ...]
    global_rank: int
    dp_dim_names: tuple[str, ...]
    coordinate: tuple[int, ...]
    data_rank: int
    data_parallel_size: int
    data_owner_rank: int
    data_owner_ranks: tuple[int, ...]
    model_parallel_rank_groups: tuple[tuple[int, ...], ...]
    model_parallel_ranks: tuple[int, ...]
    cp_rank: int
    cp_size: int
    tp_rank: int
    tp_size: int

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        *,
        global_rank: int | None = None,
        dp_dim_names: tuple[str, ...] | None = None,
    ) -> "DataTopology":
        """Build topology from a HyperParallel ``DeviceMesh``.

        Args:
            mesh: Root named device mesh used by the training job.
            global_rank: Current global rank. Defaults to the platform rank.
            dp_dim_names: Base mesh dimensions defining a DP coordinate.

        Returns:
            Derived :class:`DataTopology`.
        """
        if not mesh.mesh_dim_names:
            raise ValueError("Distributed data loading requires a DeviceMesh with named dimensions.")
        rank = platform.get_rank() if global_rank is None else global_rank
        return cls.from_layout(
            tuple(mesh.mesh_shape),
            tuple(mesh.mesh_dim_names),
            tuple(mesh.rank_list),
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
        """Build topology from a raw named mesh layout.

        This constructor is useful for configuration validation before
        process groups are created.
        """
        if not mesh_shape or any(size < 1 for size in mesh_shape):
            raise ValueError(f"mesh_shape must contain positive dimensions, but got {mesh_shape}.")
        if len(mesh_shape) != len(mesh_dim_names):
            raise ValueError("mesh_shape and mesh_dim_names must have the same length.")
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise ValueError(f"mesh_dim_names must be unique, but got {mesh_dim_names}.")
        if math.prod(mesh_shape) != len(rank_list):
            raise ValueError(
                f"mesh_shape contains {math.prod(mesh_shape)} positions, but rank_list has {len(rank_list)} ranks."
            )
        if len(set(rank_list)) != len(rank_list):
            raise ValueError(f"rank_list must not contain duplicates, but got {rank_list}.")
        if global_rank not in rank_list:
            raise ValueError(f"global_rank {global_rank} is not present in rank_list {rank_list}.")

        selected_dp_names = dp_dim_names
        if selected_dp_names is None:
            selected_dp_names = tuple(name for name in mesh_dim_names if name in _DEFAULT_DP_DIM_NAMES)
        if len(set(selected_dp_names)) != len(selected_dp_names):
            raise ValueError(f"dp_dim_names must be unique, but got {selected_dp_names}.")
        if any(name not in mesh_dim_names for name in selected_dp_names):
            raise ValueError(
                f"dp_dim_names {selected_dp_names} must be present in mesh_dim_names {mesh_dim_names}."
            )

        coordinate = _unflatten_coordinate(rank_list.index(global_rank), mesh_shape)
        dp_indices = tuple(mesh_dim_names.index(name) for name in selected_dp_names)
        dp_shape = tuple(mesh_shape[index] for index in dp_indices)
        dp_coordinate = tuple(coordinate[index] for index in dp_indices)
        data_parallel_size = math.prod(dp_shape) if dp_shape else 1
        data_rank = _flatten_coordinate(dp_coordinate, dp_shape) if dp_shape else 0

        data_owner_ranks = []
        for candidate_data_rank in range(data_parallel_size):
            candidate_dp_coordinate = _unflatten_coordinate(candidate_data_rank, dp_shape) if dp_shape else ()
            owner_coordinate = [0] * len(mesh_shape)
            for index, value in zip(dp_indices, candidate_dp_coordinate, strict=True):
                owner_coordinate[index] = value
            data_owner_ranks.append(rank_list[_flatten_coordinate(tuple(owner_coordinate), mesh_shape)])

        model_parallel_rank_groups = [[] for _ in range(data_parallel_size)]
        for flat_index, candidate_rank in enumerate(rank_list):
            candidate_coordinate = _unflatten_coordinate(flat_index, mesh_shape)
            candidate_dp_coordinate = tuple(candidate_coordinate[index] for index in dp_indices)
            candidate_data_rank = _flatten_coordinate(candidate_dp_coordinate, dp_shape) if dp_shape else 0
            model_parallel_rank_groups[candidate_data_rank].append(candidate_rank)
        model_parallel_rank_groups = tuple(tuple(ranks) for ranks in model_parallel_rank_groups)
        model_parallel_ranks = model_parallel_rank_groups[data_rank]

        owner_coordinate = list(coordinate)
        for index, _ in enumerate(owner_coordinate):
            if index not in dp_indices:
                owner_coordinate[index] = 0
        data_owner_rank = rank_list[_flatten_coordinate(tuple(owner_coordinate), mesh_shape)]

        cp_rank, cp_size = cls._axis_coordinate("cp", mesh_dim_names, mesh_shape, coordinate)
        tp_rank, tp_size = cls._axis_coordinate("tp", mesh_dim_names, mesh_shape, coordinate)
        return cls(
            mesh_shape=mesh_shape,
            mesh_dim_names=mesh_dim_names,
            rank_list=rank_list,
            global_rank=global_rank,
            dp_dim_names=selected_dp_names,
            coordinate=coordinate,
            data_rank=data_rank,
            data_parallel_size=data_parallel_size,
            data_owner_rank=data_owner_rank,
            data_owner_ranks=tuple(data_owner_ranks),
            model_parallel_rank_groups=model_parallel_rank_groups,
            model_parallel_ranks=model_parallel_ranks,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    @staticmethod
    def _axis_coordinate(
        name: str,
        mesh_dim_names: tuple[str, ...],
        mesh_shape: tuple[int, ...],
        coordinate: tuple[int, ...],
    ) -> tuple[int, int]:
        if name not in mesh_dim_names:
            return 0, 1
        index = mesh_dim_names.index(name)
        return coordinate[index], mesh_shape[index]

    @property
    def is_data_owner(self) -> bool:
        """Return whether the current rank fetches data for its DP coordinate."""
        return self.global_rank == self.data_owner_rank

    def validate_metadata_group(self, group: Any) -> None:
        """Validate that a metadata group contains exactly the data owners."""
        self._validate_group(group, self.data_owner_ranks, "metadata_group")

    def validate_model_parallel_group(self, group: Any) -> None:
        """Validate the model-parallel group for this DP coordinate."""
        self._validate_group(group, self.model_parallel_ranks, "model_parallel_group")

    @staticmethod
    def _validate_group(group, expected_ranks: tuple[int, ...], name: str) -> None:
        actual_ranks = tuple(platform.get_process_group_ranks(group))
        if set(actual_ranks) != set(expected_ranks):
            raise ValueError(f"{name} ranks must be {expected_ranks}, but got {actual_ranks}.")
