# Copyright 2025 Huawei Technologies Co., Ltd
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
"""utils"""

from typing import List, Optional
from mindspore import Tensor, mint
from mindspore.communication import get_rank, create_group
from hyper_parallel import Layout, DTensor
from hyper_parallel.core.layout import _DeviceMatrix


def create_dtensor(data, layout, dtype=None):
    """create_dtensor"""
    if dtype:
        return DTensor.from_local(Tensor(data, dtype=dtype), layout)
    return DTensor.from_local(Tensor(data), layout)

def _infer_slice_area_by_rank(dev_matrix, tensor_map, rank_id: int, full_shape: tuple): # -> tuple[tuple[int]]:
    """Return the range of each axis from full tensor for slice in current rank."""
    def _get_dev_num_alone_dim(matrix, dim):
        return matrix[-dim - 1] if dim != -1 else 1

    def _rank_id_to_dev_id_list(dev_matrix, rank_id):
        """Infer dev id list by rank_id and dev_matrix"""
        dims = len(dev_matrix)
        dev_id_list = [0] * dims
        for i in range(dims - 1, -1, -1):
            dev_id_list[i] = rank_id % dev_matrix[i]
            rank_id = rank_id // dev_matrix[i]
        return dev_id_list

    dev_id_list = _rank_id_to_dev_id_list(dev_matrix, rank_id)

    dims = len(full_shape)
    area = []
    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_alone_dim(dev_matrix, dim)

        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_alone_dim(dev_matrix, dim)
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))
    return area


def global_to_local(global_tensor, layout):
    """Transfer global tensor to local tensor by layout"""
    inner_rank_id = layout.rank_list.index(layout.mesh.rank)
    slice_area = _infer_slice_area_by_rank(layout.device_matrix, layout.tensor_map, inner_rank_id, global_tensor.shape)

    def get_slice_data(full_data, offset):
        area = ()
        for begin, end in offset:
            area += (slice(begin, end),)
        return full_data[area]

    local_tensor = get_slice_data(global_tensor, slice_area)
    return DTensor.from_local(local_tensor, layout)


def local_to_global(local_tensor):
    """Transfer local tensor to global tensor, preserving layout structure."""
    layout = local_tensor.layout
    def update_layout(dim):
        if isinstance(dim, tuple):
            return tuple("None" for _ in dim)
        return "None"
    tensor_map = layout.tensor_map
    none_structure = tuple(update_layout(dim) for dim in tensor_map)
    to_layout = layout(*none_structure)
    return local_tensor.redistribute(to_layout).to_local()


def mesh_scatter(output: Tensor, scatter_list: List[Tensor], dev_mesh: _DeviceMatrix,
                 mesh_alias: str, group_src_rank: Optional[int] = 0):
    """mesh_scatter"""
    rank_id = get_rank()
    if rank_id not in dev_mesh.rank_list:
        raise ValueError(f"rank {rank_id} not in {dev_mesh}")
    scatter_ranks = dev_mesh.get_rank_list_along_axis(mesh_alias)
    scatter_group_name = f"mesh_processgroup:{'-'.join([str(i) for i in scatter_ranks])}"
    create_group(scatter_group_name, scatter_ranks)
    source_rank = scatter_ranks[group_src_rank]
    scatter_list = [chunk.contiguous() if not chunk.is_contiguous() else chunk for chunk in scatter_list]
    mint.distributed.scatter(output, scatter_list, source_rank, scatter_group_name)
    return output


def mesh_broadcast(tensor: Tensor, dev_mesh: _DeviceMatrix,
                   mesh_alias: str, group_src_rank: Optional[int] = 0):
    """mesh_broadcast"""
    rank_id = get_rank()
    if rank_id not in dev_mesh.rank_list:
        raise ValueError(f"rank {rank_id} not in {dev_mesh}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    broadcast_ranks = dev_mesh.get_rank_list_along_axis(mesh_alias)
    broadcast_group_name = f"mesh_processgroup:{'-'.join([str(i) for i in broadcast_ranks])}"
    create_group(broadcast_group_name, broadcast_ranks)
    source_rank = broadcast_ranks[group_src_rank]
    mint.distributed.broadcast(tensor, source_rank, broadcast_group_name)
    return tensor


def distribute_tensor(tensor: Tensor,
                      layout: Layout,
                      src_data_rank: Optional[int] = 0):
    """
        Shard global tensor by layout
        Do scatter if the tensor sharded along an axis.
        Do broadcast if the tensor replicated along an axis.
    """
    if not layout.tensor_map:
        raise RuntimeError("For distribute_tensor, layout should config 'tensor_map'")
    local_tensor = tensor
    shardmap_alias_names = layout.alias_tensor_map
    dev_matrix = layout.mesh
    dev_mat_aliases = dev_matrix.alias_name
    sharded_dim_set = set()
    for dim, shard_alias in enumerate(shardmap_alias_names):
        if shard_alias != "None":
            sharded_dim_set.add(shard_alias)
            num_chunk = dev_matrix.get_device_num_along_axis(shard_alias)
            if local_tensor.shape[dim] < num_chunk and local_tensor.shape[dim] % num_chunk != 0:
                raise ValueError((f"For local tensor shape: {local_tensor.shape},"),
                                 (f"dim '{dim}' should >= num_chunk and can be evenly divided by num_chunk."),
                                 (f"but got dim({dim})={local_tensor.shape[dim]}, num_chunk={num_chunk}"))
            scatter_list = mint.chunk(local_tensor, num_chunk, dim)
            output = mint.empty(size=scatter_list[0].shape, dtype=scatter_list[0].dtype)
            local_tensor = mesh_scatter(output, scatter_list, dev_matrix, shard_alias, src_data_rank)
    # Do broadcast on the rest of dim
    for _, dev_mat_dim_alias in enumerate(dev_mat_aliases):
        if dev_mat_dim_alias in sharded_dim_set:
            continue
        local_tensor = mesh_broadcast(local_tensor, dev_matrix, dev_mat_dim_alias, src_data_rank)
    return DTensor.from_local(local_tensor, layout)
