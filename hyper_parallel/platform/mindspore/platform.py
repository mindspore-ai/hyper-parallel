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
"""MindSpore platform api"""
import mindspore as ms
from mindspore.nn import Cell
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.communication import get_group_size
from mindspore.communication import create_group as new_group
from mindspore.communication import get_rank as get_rank_id
import mindspore.communication.comm_func as comm_func
from hyper_parallel.platform.platform import Platform
from hyper_parallel.platform.mindspore.dtensor import DTensorBase
from hyper_parallel.platform.mindspore.parameter_init import init_parameters as _init_parameters


class MindSporePlatform(Platform):
    """MindSpore platform api"""
    Tensor = Tensor
    Parameter = Parameter
    Module = Cell
    DTensorBase = DTensorBase

    @staticmethod
    def get_rank():
        return get_rank_id()

    @staticmethod
    def get_world_size():
        return get_group_size()

    @staticmethod
    def get_op_name(func):
        return func.name

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output, _ = comm_func.all_gather_into_tensor(data, group=group)
        if concat_dim == 0:
            return output
        output_tensors = ms.ops.Split(output_num=concat_size)(output)
        return ms.mint.concat(output_tensors, concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return ms.ops.Split(axis=split_dim, output_num=split_size)(data)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor, _ = comm_func.all_to_all_single_with_output_shape(
            output_shape=output_shape,
            tensor=input_data,
            group=group,
            async_op=False
        )
        return output_tensor

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        output, _ = comm_func.all_reduce(data, op, group)
        return output

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        if axis > 0:
            data = ms.mint.concat(ms.ops.Split(axis=axis, output_num=dev_num)(data), dim=0)
        output_tensor, _ = comm_func.reduce_scatter_tensor(data, 'sum', group)
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def init_parameters(module, stage_index):
        return _init_parameters(module, stage_index)

    #pylint: disable=W0212
    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        if isinstance(param, DTensorBase):
            param.set_data(data)
        else:
            param._update_data(data)

    @staticmethod
    def get_param_local_shape(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """get param local shape"""
        if isinstance(param, DTensorBase):
            return param.to_local()
        return param

    @staticmethod
    def get_param_type_size(param):
        return type_size_in_bytes(param.dtype)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad):
        param = Parameter(initializer("zeros", param_shape, param_type), requires_grad=requires_grad)
        return param

    @staticmethod
    def new_tensor(tensor_shape, tensor_type):
        return Tensor(shape=tensor_shape, dtype=tensor_type)

    def _create_group(self, rank_list, group_name=None):
        if group_name is None:
            hash_str_rank_list = '-'.join([str(rank) for rank in rank_list])
            group_name = f"{len(rank_list)}-{hash_str_rank_list}"
        new_group(rank_ids=rank_list, group=group_name)
        return group_name

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        return comm_func.all_gather_into_tensor(data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        return comm_func.all_reduce(data, group=group_info.group_name, async_op=async_op)

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        return comm_func.reduce_scatter_tensor(data, group=group_info.group_name, async_op=async_op)

    def new_stream(self):
        return ms.runtime.Stream()

    def get_stream_context(self):
        return ms.runtime.StreamCtx
