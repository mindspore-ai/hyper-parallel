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
"""MindSpore HSDP parameter"""
from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.common.initializer import initializer
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.hsdp.hsdp_param import HSDPParam


class MindSporeHSDPParam(HSDPParam):
    """
    MindSpore HSDP parameter.
    """
    def _init_rank_info(self):
        """init parameter rank info"""
        self.rank_id = get_rank()
        self.hsdp_rank = self.rank_id
        self.local_rank = self.rank_id
        self.tp_rank = 0
        if not isinstance(self.param, DTensor) or self.param.layout is None:
            self.rank_size = get_group_size()
            return

        if len(self.param.layout.rank_list) == 1:
            self.rank_size = 1
            return

        try:
            self.local_rank = self.param.layout.rank_list.index(self.rank_id)
        except ValueError as e:
            raise ValueError(f"HSDP invalid rank {self.rank_id} with rank list {self.param.layout.rank_list}.") from e

        tensor_map = self.param.layout.tensor_map
        sharded_axis_set = set()
        for axis in tensor_map:
            if isinstance(axis, int) and axis != -1:
                sharded_axis_set.add(axis)
                continue
            if isinstance(axis, tuple):
                for item in axis:
                    sharded_axis_set.add(item)
        self.sharded_axis_set = sharded_axis_set
        self.rank_size = 1
        self.unsharded_reverse_axis_list = []
        self.global_rank_stride_list = []
        self.hsdp_rank_stride_list = []
        self.tp_rank_stride_list = []
        device_dims = len(self.param.layout.device_matrix)
        stride = 1
        hsdp_stride = 1
        tp_stride = 1
        for axis in range(device_dims):
            r_axis = device_dims - 1 - axis
            self.global_rank_stride_list.append(stride)
            self.hsdp_rank_stride_list.append(hsdp_stride)
            self.tp_rank_stride_list.append(tp_stride)
            stride = stride * self.param.layout.device_matrix[r_axis]
            if axis in self.sharded_axis_set:
                tp_stride = tp_stride * self.param.layout.device_matrix[r_axis]
                continue

            hsdp_stride = hsdp_stride * self.param.layout.device_matrix[r_axis]
            self.unsharded_reverse_axis_list.append(r_axis)
            self.rank_size = self.rank_size * self.param.layout.device_matrix[r_axis]
        self.global_rank_stride_list.reverse()
        self.hsdp_rank_stride_list.reverse()
        self.tp_rank_stride_list.reverse()
        self.unsharded_reverse_axis_list.reverse()

        rank_indices = []
        index = self.local_rank
        for stride in self.global_rank_stride_list:
            rank_indices.append(index // stride)
            index = index % stride
        self.rank_indices = rank_indices
        hsdp_rank = 0
        for axis in self.unsharded_reverse_axis_list:
            hsdp_rank = hsdp_rank + rank_indices[axis] * self.hsdp_rank_stride_list[axis]
        self.hsdp_rank = hsdp_rank
        tp_rank = 0
        for axis in range(device_dims):
            if axis in self.sharded_axis_set:
                r_axis = device_dims - 1 - axis
                tp_rank = tp_rank + rank_indices[r_axis] * self.tp_rank_stride_list[r_axis]
        self.tp_rank = tp_rank

    def _hsdp_rank_to_global_rank(self, hsdp_rank_list):
        """transform from hsdp rank to global rank"""
        rank_list = []
        for hsdp_rank in hsdp_rank_list:
            local_index = hsdp_rank
            local_indices_dict = {}
            for axis in self.unsharded_reverse_axis_list:
                stride = self.hsdp_rank_stride_list[axis]
                local_indices_dict[axis] = local_index // stride
                local_index = local_index % stride
            global_rank = 0
            for axis, index in enumerate(self.rank_indices):
                index = local_indices_dict.get(axis, index)
                global_rank = global_rank + index * self.global_rank_stride_list[axis]
            if self.param.layout is not None:
                if global_rank >= len(self.param.layout.rank_list):
                    raise ValueError(f"HSDP invalid index {global_rank} with"
                                     f"rank list len {len(self.param.layout.rank_list)}.")
                global_rank = self.param.layout.rank_list[global_rank]
            rank_list.append(global_rank)
        return rank_list

    def _get_op_rank_list(self):
        """get data parallel rank list"""
        if isinstance(self.param, DTensor):
            rank_base = self.hsdp_rank // self.shard_size * self.shard_size
            hsdp_rank_list = [i + rank_base for i in range(self.shard_size)]
            return self._hsdp_rank_to_global_rank(hsdp_rank_list)
        return super()._get_op_rank_list()

    def _get_dp_rank_list(self):
        """get optimizer parallel rank list"""
        if isinstance(self.param, DTensor):
            rank_stride = self.shard_size
            rank_base = self.hsdp_rank % rank_stride
            hsdp_rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
            return self._hsdp_rank_to_global_rank(hsdp_rank_list)
        return super()._get_dp_rank_list()

    def _init_sharded_param(self):
        """add and init sharded param"""
        if not self.param.has_init:
            slice_index = self.hsdp_rank % self.shard_size
            local_param = self.platform.get_param_local_data(self.param)
            param_slice = ops.split(local_param, local_param.shape[0] // self.shard_size)[slice_index] + 0
            self.platform.update_param_data(self.param, param_slice)
            self.sharded_param = Parameter(param_slice,
                                           name="sharded_"+self.param.name,
                                           requires_grad=self.param.requires_grad)
        else:
            dp_slice_index = self.hsdp_rank % self.shard_size
            data_slice_index = self.tp_rank * self.shard_size + dp_slice_index
            if isinstance(self.param.init_mode, DTensor):
                init_mode_local_shape = self.param.init_mode.local_shape
            else:
                init_mode_local_shape = self.param.init_mode.shape
            init_shape = list(init_mode_local_shape)
            init_shape[0] = init_shape[0] // self.shard_size
            if isinstance(self.param.init_mode, DTensor):
                self.param.init_mode.to_local().shape = init_shape
            else:
                self.param.init_mode.shape = init_shape
            self.param.hsdp_init_index = data_slice_index
            self.sharded_param = Parameter(initializer("zeros", init_shape, self.param.dtype),
                                           name="sharded_"+self.param.name,
                                           requires_grad=self.param.requires_grad)

    def _init_unsharded_param(self):
        """add and init unshared param when using graph hook"""
        if self.config.use_eager_hook:
            return
        local_shape = self.param.local_shape if isinstance(self.param, DTensor) else self.param.shape
        self.unsharded_param = Parameter(initializer("zeros", local_shape, self.param.dtype),
                                         name="unsharded_"+self.param.name,
                                         requires_grad=self.param.requires_grad)
        self.unsharded_param_available = Parameter(Tensor(False),
                                                   name="available_unsharded_"+self.param.name,
                                                   requires_grad=False)
