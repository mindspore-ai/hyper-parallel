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
"""Mesh-scoped collectives for :func:`distribute_tensor` (PyTorch DTensor parity)."""
from __future__ import annotations

from typing import Optional, Sequence

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor


def _ensure_mesh_process_groups(mesh: DeviceMesh) -> None:
    """Lazily create per-axis process groups when mesh was built with ``init_backend=False``."""
    if hasattr(mesh, "_dim_group_names") and mesh._dim_group_names is not None:
        return
    mesh._dim_group_names = DeviceMesh._init_process_groups(  # pylint: disable=protected-access
        mesh._mesh_shape,
        mesh.mesh_dim_names,
        mesh._rank_list,
    )


def mesh_scatter(
    output: Tensor,
    scatter_list: Sequence[Tensor],
    mesh: DeviceMesh,
    mesh_dim: int,
    *,
    group_src: int = 0,
) -> Tensor:
    """Scatter tensor chunks along one mesh dimension (PyTorch ``mesh_scatter`` parity)."""
    _ensure_mesh_process_groups(mesh)
    group = mesh.get_group(mesh_dim)
    contiguous_list = [
        chunk.contiguous() if hasattr(chunk, "is_contiguous") and not chunk.is_contiguous() else chunk
        for chunk in scatter_list
    ]
    if platform.get_group_rank(group) == group_src:
        platform.scatter(output, list(contiguous_list), group=group, group_src=group_src)
    else:
        platform.scatter(output, None, group=group, group_src=group_src)
    return output


def mesh_scatter_ragged(
    output: Tensor,
    scatter_list: Optional[Sequence[Tensor]],
    mesh: DeviceMesh,
    mesh_dim: int,
    *,
    group_src: int = 0,
) -> Tensor:
    """Scatter variable-length flat tensors with point-to-point communication.

    Args:
        output: Preallocated receive buffer for the current rank.
        scatter_list: Source-rank tensors ordered by group rank. Non-source ranks
            may pass ``None``.
        mesh: Device mesh containing the communication group.
        mesh_dim: Mesh dimension along which to scatter.
        group_src: Source rank relative to the mesh-dimension group.

    Returns:
        The populated current-rank output buffer.

    Raises:
        ValueError: If the source rank or source scatter list is invalid.
    """
    _ensure_mesh_process_groups(mesh)
    group = mesh.get_group(mesh_dim)
    group_size = mesh.size(mesh_dim)
    if group_src < 0 or group_src >= group_size:
        raise ValueError(
            f"group_src must be in [0, {group_size}), but got {group_src}"
        )

    group_rank = platform.get_group_rank(group)
    source_global_rank = platform.get_global_rank(group, group_src)
    if group_rank == group_src:
        if scatter_list is None or len(scatter_list) != group_size:
            raise ValueError(
                "source scatter_list length must equal the mesh dimension size, "
                f"got scatter_list={scatter_list!r}, group_size={group_size}"
            )
        output.copy_(scatter_list[group_src])
        works = []
        for destination_group_rank, chunk in enumerate(scatter_list):
            if destination_group_rank == group_src:
                continue
            destination_global_rank = platform.get_global_rank(
                group, destination_group_rank
            )
            works.append(
                platform.isend(
                    chunk.contiguous(),
                    dst=destination_global_rank,
                    group=group,
                )
            )
        for work in works:
            work.wait()
        return output

    work = platform.irecv(
        output,
        src=source_global_rank,
        group=group,
    )
    work.wait()
    return output


def mesh_broadcast(
    tensor: Tensor,
    mesh: DeviceMesh,
    mesh_dim: int,
    *,
    group_src: int = 0,
) -> Tensor:
    """Broadcast a tensor along one mesh dimension (PyTorch ``mesh_broadcast`` parity)."""
    _ensure_mesh_process_groups(mesh)
    group = mesh.get_group(mesh_dim)
    if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    platform.broadcast(tensor, group=group, group_src=group_src)
    return tensor
