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
from mindspore.common.dtype import type_size_in_bytes
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.communication import create_group as new_group
import mindspore.communication.comm_func as comm_func
from mindspore.parallel import DTensor
from dist_parallel.hsdp.platform.platform import Platform


class MindSporePlatform(Platform):
    """MindSpore platform api"""

    #pylint: disable=W0212
    @staticmethod
    def update_param_data(param, data):
        """update param data"""
        if isinstance(param, DTensor):
            param.set_data(data)
        else:
            param._update_data(data)

    @staticmethod
    def get_param_local_shape(param):
        """get param local shape"""
        if isinstance(param, DTensor):
            return param.local_shape
        return param.shape

    @staticmethod
    def get_param_local_data(param):
        """get param local shape"""
        if isinstance(param, DTensor):
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

    def create_group(self, group_name, rank_list):
        new_group(group_name, rank_list)
        return None

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
