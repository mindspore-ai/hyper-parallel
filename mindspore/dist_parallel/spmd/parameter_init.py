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
"""Parameter init"""

def init_parameters(cell, stage_index=0):
    r"""
        init parameters.

        Args:
            cell(Cell): The cell to init parameters.
            stage_index: stage index for init.
        Raises:
            ValueError: If the `cell` is not a cell.
    """
    import mindspore as ms
    from mindspore.nn.cell import Cell
    from mindspore.parallel._tensor import _get_slice_index
    import dist_parallel
    if not isinstance(cell, Cell):
        raise ValueError("cell's type must be Cell but got {}.".format(type(cell)))
    if not isinstance(stage_index, int):
        raise ValueError("stage_index's type must be int but got {}.".format(type(stage_index)))
    for param in cell.get_parameters(expand=True):
        param_is_dtensor = isinstance(param, dist_parallel.DTensor)
        if not param.has_init:
            continue
        data_slice_index = None
        if hasattr(param, "hsdp_init_index"):
            data_slice_index = param.hsdp_init_index
        elif param_is_dtensor and param.layout is not None:
            data_slice_index = _get_slice_index(param.layout.device_matrix, param.layout.tensor_map, None)
        local_shape = param.shape
        init_tensor = param.init_mode
        if param_is_dtensor:
            local_shape = param.local_shape
            init_tensor = param.init_mode.to_local()
            if isinstance(init_tensor, ms.Parameter):
                init_tensor = init_tensor.init_mode

        if data_slice_index is not None:
            init_data = init_tensor.init_data(slice_index=int(data_slice_index) + stage_index, shape=local_shape)
        else:
            init_data = init_tensor.init_data(shape=local_shape)
        param.init_mode = None
        param.init = None
        param.set_data(init_data)
    return cell
