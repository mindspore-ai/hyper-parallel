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

from hyper_parallel import DTensor

def _infer_slice_area_by_rank(mesh_shape, tensor_map, rank_id: int, full_shape: tuple):  # -> tuple[tuple[int]]:
    """Return the range of each axis from full tensor for slice in current rank."""

    def _get_dev_num_alone_dim(matrix, dim):
        return matrix[-dim - 1] if dim != -1 else 1

    def _rank_id_to_dev_id_list(mesh_shape, rank_id):
        """Infer dev id list by rank_id and mesh_shape"""
        dims = len(mesh_shape)
        dev_id_list = [0] * dims
        for i in range(dims - 1, -1, -1):
            dev_id_list[i] = rank_id % mesh_shape[i]
            rank_id = rank_id // mesh_shape[i]
        return dev_id_list

    dev_id_list = _rank_id_to_dev_id_list(mesh_shape, rank_id)

    dims = len(full_shape)
    area = []
    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_alone_dim(mesh_shape, dim)

        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_alone_dim(mesh_shape, dim)
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))
    return area


def global_to_local(global_tensor, layout):
    """Transfer global tensor to local tensor by layout"""
    inner_rank_id = layout.rank_list.index(layout.mesh.rank)
    slice_area = _infer_slice_area_by_rank(layout.mesh_shape, layout.tensor_map, inner_rank_id, global_tensor.shape)

    def get_slice_data(full_data, offset):
        area = ()
        for begin, end in offset:
            area += (slice(begin, end),)
        return full_data[area]

    local_tensor = get_slice_data(global_tensor, slice_area)
    return DTensor.from_local(local_tensor, layout.mesh, layout.placements)


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
    return local_tensor.redistribute(to_layout.mesh, to_layout.placements).to_local()
