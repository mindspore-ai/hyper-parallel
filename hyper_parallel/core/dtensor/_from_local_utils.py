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
"""Utilities for :meth:`DTensor.from_local` with ``run_check=True`` (PyTorch parity)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor


def _ensure_mesh_process_groups(mesh: DeviceMesh) -> None:
    if hasattr(mesh, "_dim_group_names") and mesh._dim_group_names is not None:
        return
    mesh._dim_group_names = DeviceMesh._init_process_groups(  # pylint: disable=protected-access
        mesh._mesh_shape,
        mesh.mesh_dim_names,
        mesh._rank_list,
    )


def mesh_broadcast(
    tensor: Tensor,
    mesh: DeviceMesh,
    mesh_dim: int,
    *,
    group_src: int = 0,
) -> Tensor:
    """Broadcast *tensor* along one mesh dimension."""
    _ensure_mesh_process_groups(mesh)
    group = mesh.get_group(mesh_dim)
    if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    rank_list = mesh.get_rank_list_along_axis(mesh_dim)
    src = rank_list[group_src]
    platform.broadcast(tensor, src, group=group)
    return tensor


def _tensor_meta(local_tensor: Tensor, *, check_shape_stride: bool) -> dict:
    meta = {
        "dtype": str(local_tensor.dtype),
        "requires_grad": bool(getattr(local_tensor, "requires_grad", False)),
    }
    if check_shape_stride:
        meta["shape"] = tuple(local_tensor.shape)
        if hasattr(local_tensor, "stride"):
            meta["stride"] = tuple(local_tensor.stride())
    return meta


def check_tensor_meta(
    local_tensor: Tensor,
    group,
    group_size: int,
    *,
    check_shape_stride: bool,
) -> None:
    """Gather tensor metadata across *group* and verify consistency."""
    local_meta = _tensor_meta(local_tensor, check_shape_stride=check_shape_stride)
    gathered = [None] * group_size
    platform.all_gather_object(gathered, local_meta, group=group)
    if not all(meta == local_meta for meta in gathered if meta is not None):
        raise ValueError(
            "Inconsistent tensor metadata across ranks in from_local(run_check=True): "
            f"local={local_meta}, gathered={gathered}"
        )


def _mesh_check_group(device_mesh: DeviceMesh):
    """Return the process group and size covering all ranks in *device_mesh*."""
    _ensure_mesh_process_groups(device_mesh)
    if device_mesh.ndim == 1:
        return device_mesh.get_group(0), device_mesh.size(0)
    flat_mesh = device_mesh.flatten()
    _ensure_mesh_process_groups(flat_mesh)
    return flat_mesh.get_group(0), flat_mesh.size(0)


def run_from_local_checks(
    local_tensor: Tensor,
    device_mesh: DeviceMesh,
    resolved_placements: Sequence[Placement],
    *,
    shape: Optional[Tuple[int, ...]] = None,
    stride: Optional[Tuple[int, ...]] = None,
) -> None:
    """Validate local shards and align replicate placements before wrapping as DTensor."""
    check_shape_stride = shape is None and stride is None
    group, group_size = _mesh_check_group(device_mesh)
    check_tensor_meta(
        local_tensor,
        group,
        group_size,
        check_shape_stride=check_shape_stride,
    )
    for mesh_dim, placement in enumerate(resolved_placements):
        if placement.is_replicate():
            mesh_broadcast(local_tensor, device_mesh, mesh_dim)
