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
"""Torch platform api"""
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter, Module
from torch._ops import OpOverload, OpOverloadPacket
import torch.distributed.nn.functional as dist_func
import torch.distributed as dist
from hyper_parallel.platform.platform import Platform
from hyper_parallel.platform.torch.dtensor import DTensorBase
from hyper_parallel.platform.torch.group_utils import create_sub_groups


class TorchPlatform(Platform):
    """Torch platform api"""
    Tensor = Tensor
    Parameter = Parameter
    Module = Module
    DTensorBase = DTensorBase

    @staticmethod
    def get_rank():
        return dist.get_rank()

    @staticmethod
    def get_world_size():
        return dist.get_world_size()

    @staticmethod
    def get_op_name(func):
        if hasattr(func, "__name__"):
            return func.__name__
        elif isinstance(func, OpOverload):
            full_name = func.name
            core_name = full_name.split("::")[-1].split(".")[0]
            return core_name
        elif isinstance(func, OpOverloadPacket):
            return func.name.split("::")[-1]
        else:
            func_str = str(func)
            if "built-in function" in func_str:
                return func_str.split()[-1].strip(">")
            elif "function" in func_str:
                return func_str.split()[1]
            return "unknown_op"

    @staticmethod
    def differentiable_all_gather_concat(data, group, concat_size, concat_dim):
        output = dist_func.all_gather(data, group=group)
        return torch.cat(output, dim=concat_dim)

    @staticmethod
    def chunk(data, split_dim, split_size, index):
        return torch.chunk(data, split_size, dim=split_dim)[index]

    @staticmethod
    def differentiable_all_to_all(input_data, output_shape, group):
        output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
        output_tensor = dist_func.all_to_all_single(
            output_tensor,
            input_data,
            group=group
        )
        return output_tensor

    @staticmethod
    def differentiable_all_reduce(data, op, group):
        return dist_func.all_reduce(data, group=group)

    @staticmethod
    def differentiable_reduce_scatter(data, dev_num, axis, op, group):
        input_tuple = torch.chunk(data, dev_num, dim=axis)
        output_tensor = torch.empty(input_tuple[0].shape, device=data.device, dtype=data.dtype)
        output_tensor = dist_func.reduce_scatter(output_tensor, input_tuple, group=group)
        if op == 'avg':
            output_tensor = output_tensor / dev_num
        return output_tensor

    @staticmethod
    def get_device_handle(device_type="cuda"):
        return getattr(torch, device_type, None)

    @staticmethod
    def register_backward_pre_hook(cell, hook):
        return cell.register_full_backward_pre_hook(hook)

    @staticmethod
    def register_backward_hook(cell, hook):
        return cell.register_full_backward_hook(hook)

    @staticmethod
    def get_param_type_size(param):
        return torch._utils._element_size(param.dtype)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad):
        return nn.Parameter(torch.zeros(param_shape, dtype=param_type), requires_grad=requires_grad)

    @staticmethod
    def new_tensor(tensor_shape, tensor_type):
        return torch.empty(tensor_shape, tensor_type)

    def _create_group(self, rank_list, group_name):
        group_dict = create_sub_groups(rank_list)
        return group_dict[tuple(rank_list)]

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] * group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype)
        handle = dist.all_gather_into_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        handle = dist.all_reduce(data.data, group=group_info.group, async_op=async_op)
        return data, handle

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        output_shape = list(data.shape)
        output_shape[0] = output_shape[0] // group_info.rank_size
        output = torch.empty(output_shape, dtype=data.dtype)
        handle = dist.reduce_scatter_tensor(output, data, group=group_info.group, async_op=async_op)
        return output, handle

    def new_stream(self):
        device = get_device_handle()
        return device.Stream()

    def get_stream_context(self):
        device = get_device_handle()
        return device.stream
