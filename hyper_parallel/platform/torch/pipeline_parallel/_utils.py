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
from torch import nn
from hyper_parallel import custom_shard, DTensor


class BatchDimSpec:
    """
    Specify the batch dimension of a Tensor.

    Args:
        batch_dim (int): batch dimension.
    """
    __slots__ = ("batch_dim",)

    def __init__(self, batch_dim):
        if not isinstance(batch_dim, int):
            raise TypeError(f"batch_dim must be int, but got type {type(batch_dim)}.")
        self.batch_dim = batch_dim

    def __repr__(self):
        return f"BatchDimSpec({self.batch_dim})"

    def __str__(self):
        return f"BatchDim(dim={self.batch_dim})"

    @staticmethod
    def from_tuple(batch_dims):
        if not isinstance(batch_dims, tuple):
            raise TypeError(f"batch_dims must be tuple, but got type {type(batch_dims)}.")
        return tuple(BatchDimSpec(dim) for dim in batch_dims)

    @staticmethod
    def from_dict(batch_dims):
        if not isinstance(batch_dims, dict):
            raise TypeError(f"batch_dims must be dict, but got type {type(batch_dims)}.")
        return {k: BatchDimSpec(v) for k, v in batch_dims.items()}


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
                if isinstance(cur_arg, DTensor):
                    micro_arg = self.split_inputs_with_custom_shard(cur_arg, cur_arg_batch_dim, micro_idx)
                else:
                    micro_arg = self.split_inputs(cur_arg, cur_arg_batch_dim, micro_idx)
                micro_args.append(micro_arg)
            args_after_split.append(micro_args)

            for key, cur_kwarg in kwargs.items():
                cur_kwarg_batch_dim = 0
                if self.kwargs_batch_dim is not None:
                    cur_kwarg_batch_dim = self.kwargs_batch_dim[key].batch_dim
                if isinstance(cur_kwarg, DTensor):
                    micro_kwarg = self.split_inputs_with_custom_shard(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                else:
                    micro_kwarg = self.split_inputs(cur_kwarg, cur_kwarg_batch_dim, micro_idx)
                micro_kwargs[key] = micro_kwarg
            kwargs_after_split.append(micro_kwargs)
        return args_after_split, kwargs_after_split

    def split_inputs_with_custom_shard(self, input_tensor, cur_arg_batch_dim, micro_idx):
        input_layout = input_tensor.layout
        func_wrap = custom_shard(self.split_inputs, out_layouts=(input_layout,), in_layouts=(input_layout, None, None))
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


class _RecvInfo:
    """
    Used for construct forward Receive operation and backward Send operation.
    """

    def __init__(self, global_rank, buffer=None):
        self._global_rank = global_rank
        self._buffer = buffer

    @property
    def global_rank(self):
        return self._global_rank

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, val):
        self._buffer = val
