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
import torch.distributed as dist
from dist_parallel.hsdp.platform.platform import Platform


class TorchPlatform(Platform):
    """Torch platform api"""

    @staticmethod
    def get_device_handle(device_type="cuda"):
        return getattr(torch, device_type, None)

    @staticmethod
    def register_backward_pre_hook(cell, hook):
        return cell.register_fully_backward_pre_hook(hook)

    @staticmethod
    def register_backward_hook(cell, hook):
        return cell.register_fully_backward_hook(hook)

    @staticmethod
    def get_param_type_size(param):
        return torch._utils._element_size(param.dtype)

    @staticmethod
    def new_zero_parameter(param_shape, param_type, requires_grad):
        return nn.Parameter(torch.zeros(param_shape, dtype=param_type), requires_grad=requires_grad)

    @staticmethod
    def new_tensor(tensor_shape, tensor_type):
        return torch.empty(tensor_shape, tensor_type)

    def create_group(self, group_name, rank_list):
        return dist.new_group(ranks=rank_list)

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
