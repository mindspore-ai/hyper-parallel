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
"""Root-preserving submeshes for KDA state and Ulysses combinations."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

import torch.distributed as dist

from hyper_parallel.components.functional.kimi_delta_attention_cp import (
    AllGatherBoundary,
    GroupedAllGatherBoundary,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh


def split_kda_mesh(mesh: DeviceMesh, degree: int, names: tuple[str, str]) -> DeviceMesh:
    """Split a CP axis without losing the root rank map or sibling DP groups.

    Args:
        mesh: One-dimensional chronological CP mesh.
        degree: Size of the inner split axis.
        names: Names of the outer and inner axes.

    Returns:
        Two-dimensional submesh attached to the original root.
    """
    if mesh.ndim != 1 or not isinstance(degree, int) or isinstance(degree, bool):
        raise ValueError("KDA mesh splitting requires a 1-D mesh and integer degree.")
    if degree < 1 or mesh.size() % degree:
        raise ValueError("KDA split degree must be positive and divide the CP size.")
    ranks = tuple(mesh.rank_list)
    if ranks != tuple(sorted(set(ranks))):
        raise ValueError("KDA CP currently requires naturally ordered unique global ranks.")
    if mesh.root_mesh is None and mesh.size() != dist.get_world_size():
        raise ValueError("Hybrid KDA requires a CP submesh retaining its root mesh.")
    # Like the existing hybrid CP helper, preserve all sibling groups through the root layout.
    return mesh._unflatten(0, (mesh.size() // degree, degree), names)  # pylint: disable=protected-access


def build_kda_boundary(mesh: DeviceMesh, protocol: str, group_size: int) -> object | None:
    """Construct a tensor-free boundary executor during model initialization.

    Args:
        mesh: One-dimensional mesh ordered along the state sequence.
        protocol: p2p, allgather, or grouped_allgather_p2p.
        group_size: Consecutive ranks per local gather, only for the grouped protocol.

    Returns:
        Gather boundary, or None to select the existing P2P path.
    """
    if protocol not in ("p2p", "allgather", "grouped_allgather_p2p"):
        raise ValueError(f"Unknown KDA boundary protocol: {protocol!r}.")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("KDA group_size must be a positive integer.")
    if protocol != "grouped_allgather_p2p" and group_size != 1:
        raise ValueError("KDA group_size is only valid for grouped_allgather_p2p.")
    if protocol == "p2p":
        return None
    ranks = tuple(mesh.rank_list)
    if ranks != tuple(sorted(set(ranks))):
        raise ValueError("KDA gathers require naturally ordered unique global ranks.")
    rank, size = mesh.get_local_rank(), mesh.size()
    if protocol == "allgather":
        return AllGatherBoundary(mesh.get_group(), rank, size)
    if size % group_size:
        raise ValueError("KDA group_size must divide the state CP size.")
    if group_size == 1:
        return None
    if group_size == size:
        return AllGatherBoundary(mesh.get_group(), rank, size)
    split = split_kda_mesh(mesh, group_size, ("kda_outer", "kda_group"))
    return GroupedAllGatherBoundary(
        mesh.get_group(), split["kda_group"].get_group(), rank, ranks, group_size)
