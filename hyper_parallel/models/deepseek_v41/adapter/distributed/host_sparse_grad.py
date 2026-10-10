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
"""Sparse row synchronization along existing same-owner mesh axes."""
# pylint: disable=forbidden-backend-import

from typing import Any
import torch
import torch.distributed as dist


def _communication_device(mesh) -> torch.device:
    """Place control and row staging tensors on the collective backend device."""
    device_type = getattr(mesh, "device_type", "cpu")
    return torch.device(device_type)


def gather_sparse_along_axis(coo_cpu: torch.Tensor, axis_mesh: Any, width: int) -> torch.Tensor:
    """Gather variable COO rows and coalesce them without a dense table grad.

    Args:
        coo_cpu: Owner-local sparse COO gradient on CPU.
        axis_mesh: Mesh axis used to synchronize sparse rows.
        width: Number of values in each table row.
    """
    if axis_mesh is None or axis_mesh.size() == 1:
        return coo_cpu
    group = axis_mesh.get_group()
    device = _communication_device(axis_mesh)
    coo_cpu = coo_cpu.coalesce()
    local_count = torch.tensor([coo_cpu._nnz()], dtype=torch.long, device=device)
    count_parts = [torch.empty_like(local_count) for _ in range(axis_mesh.size())]
    dist.all_gather(count_parts, local_count, group=group)
    counts = [int(part.item()) for part in count_parts]
    max_count = max(counts)
    if max_count == 0:
        return coo_cpu
    ids = torch.zeros(max_count, dtype=torch.long, device=device)
    values = torch.zeros((max_count, width), dtype=torch.float32, device=device)
    if coo_cpu._nnz():
        ids[:coo_cpu._nnz()] = coo_cpu.indices()[0].to(device)
        values[:coo_cpu._nnz()] = coo_cpu.values().to(device)
    id_parts = [torch.empty_like(ids) for _ in counts]
    value_parts = [torch.empty_like(values) for _ in counts]
    dist.all_gather(id_parts, ids, group=group)
    dist.all_gather(value_parts, values, group=group)
    gathered_ids = torch.cat([part[:count].cpu() for part, count in zip(id_parts, counts)])
    gathered_values = torch.cat([part[:count].cpu() for part, count in zip(value_parts, counts)])
    return torch.sparse_coo_tensor(
        gathered_ids.unsqueeze(0), gathered_values, coo_cpu.shape,
    ).coalesce()


def sparse_sum_same_ep(coo_cpu: torch.Tensor, mesh_context: Any, width: int) -> torch.Tensor:
    """Sum row contributions over every existing same-owner replica axis.

    Args:
        coo_cpu: Owner-local sparse COO gradient on CPU.
        mesh_context: Parallel mesh and loss-domain configuration.
        width: Number of values in each table row.
    """
    if mesh_context is None or mesh_context.device_mesh is None:
        return coo_cpu
    moe = mesh_context.fsdp_moe_mesh
    if moe is None:
        for name in ("dp", "cp", "tp"):
            if name in (mesh_context.device_mesh.mesh_dim_names or ()):
                coo_cpu = gather_sparse_along_axis(coo_cpu, mesh_context.device_mesh[name], width)
    else:
        for name in ("edp_shard", "edp_replicate"):
            if name in (moe.mesh_dim_names or ()):
                coo_cpu = gather_sparse_along_axis(coo_cpu, moe[name], width)
    return coo_cpu


def stage_reduce_scalar(value: torch.Tensor, mesh_context: Any,
                        op: Any = dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce a scalar across the current PP stage in fixed mesh-axis order.

    Args:
        value: Scalar value to reduce.
        mesh_context: Parallel mesh and loss-domain configuration.
        op: Distributed reduction operation.
    """
    if mesh_context is None or mesh_context.device_mesh is None:
        return value
    for name in mesh_context.device_mesh.mesh_dim_names or ():
        axis = mesh_context.device_mesh[name]
        if axis.size() > 1:
            dist.all_reduce(value, op=op, group=axis.get_group())
    return value


def stage_check(local_bad: bool, mesh_context: Any) -> bool:
    """Align validation failure across every rank in this PP stage.

    Args:
        local_bad: Whether this rank detected an invalid state.
        mesh_context: Parallel mesh and loss-domain configuration.
    """
    mesh = mesh_context.device_mesh if mesh_context is not None else None
    flag = torch.tensor(int(local_bad), device=_communication_device(mesh))
    stage_reduce_scalar(flag, mesh_context, dist.ReduceOp.MAX)
    return bool(flag.item())
