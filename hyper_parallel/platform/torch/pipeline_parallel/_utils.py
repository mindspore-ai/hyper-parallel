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
"""pipeline parallel utils"""
import hyper_parallel
from torch import nn


class _MicroBatch(nn.Module):
    """
    Split inputs into micro_batch in pipeline parallel.

    Args:
        micro_batch_num (int): The number of micro-batch.
        args_batch_dim (list, optional): Specify the batch dim of the args.
            Default ``None``.
        kwargs_batch_dim(dict, optional): Specify the batch dim of the kwargs.
            Default ``None``.
    Inputs:
        - **args** (list) - Input args.
        - **kwargs** (dict) - Input kwargs.

    Outputs:
        - **args_after_split** (list) - Input args after split into micro_batches.
        - **kwargs_after_split** (list) - Input kwargs after split into micro_batches.
    """

    def __init__(self, micro_batch_num, args_batch_dim=None, kwargs_batch_dim=None):
        super().__init__()
        self.micro_batch_num = micro_batch_num
        self.args_batch_dim = args_batch_dim
        self.kwargs_batch_dim = kwargs_batch_dim

    def forward(self, args, kwargs):
        """forward of _MicroBatch"""
        args_after_split = []
        kwargs_after_split = []
        for micro_idx in range(self.micro_batch_num):
            micro_args = []
            micro_kwargs = {}
            for arg_idx, cur_arg in enumerate(args):
                cur_arg_batch_dim = 0
                if self.args_batch_dim and self.args_batch_dim[arg_idx] is not None:
                    cur_arg_batch_dim = self.args_batch_dim[arg_idx].batch_dim
                if isinstance(cur_arg, hyper_parallel.DTensor):
                    micro_arg = self.split_inputs_with_custom_shard(cur_arg, cur_arg_batch_dim, micro_idx)
                else:
                    micro_arg = self.split_inputs(cur_arg, cur_arg_batch_dim, micro_idx)
                micro_args.append(micro_arg)
            args_after_split.append(micro_args)

            for key, cur_kwarg in kwargs.items():
                cur_kwarg_batch_dim = 0
                if self.kwargs_batch_dim is not None:
                    cur_kwarg_batch_dim = self.kwargs_batch_dim[key].batch_dim
                if isinstance(cur_kwarg, hyper_parallel.DTensor):
                    micro_kwarg = self.split_inputs_with_custom_shard(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                else:
                    micro_kwarg = self.split_inputs(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                micro_kwargs[key] = micro_kwarg
            kwargs_after_split.append(micro_kwargs)
        return args_after_split, kwargs_after_split

    def split_inputs_with_custom_shard(self, input_tensor, cur_arg_batch_dim, micro_idx):
        input_layout = input_tensor.layout
        func_wrap = hyper_parallel.custom_shard(self.split_inputs,
                                 device_mesh=input_layout.mesh,
                                 out_placements=(input_layout.placements,),
                                 in_placements=(input_layout.placements, None, None)
                                 )
        return func_wrap(input_tensor, cur_arg_batch_dim, micro_idx)

    def split_inputs(self, input_tensor, cur_arg_batch_dim, micro_idx):
        """
        Split the input along the specified batch_dim and micro_idx
        """
        if cur_arg_batch_dim == -1:
            return input_tensor
        batch_dim_shape = input_tensor.shape[cur_arg_batch_dim]
        if batch_dim_shape % self.micro_batch_num != 0:
            raise ValueError(f"Batch dimension size {batch_dim_shape} is not divisible by \
                micro_batch_num {self.micro_batch_num}")
        micro_batch_size = batch_dim_shape // self.micro_batch_num

        # Calculate start and end idx
        start = micro_batch_size * micro_idx
        end = micro_batch_size * (micro_idx + 1)

        # Create slicing tuple
        slices = [slice(None)] * input_tensor.ndim
        slices[cur_arg_batch_dim] = slice(start, end)
        return input_tensor[slices]
