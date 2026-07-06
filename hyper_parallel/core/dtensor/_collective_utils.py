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

from typing import Sequence

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
