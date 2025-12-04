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
"""HSDP parameter"""
import functools
from mindspore import ops
from mindspore.common.api import _no_grad
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.common.initializer import initializer
from mindspore.common.dtype import type_size_in_bytes
from dist_parallel.spmd.hsdp.hsdp_utils import OptimizerLevel
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm
from mindspore.parallel import DTensor


class HSDPParam:
    """
    HSDP parameter.
    """
    def __init__(self, cell, param_name, param, config):
        self.cell = cell
        self.param_name = param_name
        self.param = param
        self.config = config
        self.shard_size = 1
        self.unsharded_param = None
        self.sharded_param = None
        self.acc_grad = None
        self.grad = None
        self.sharded = False
        self.fully_sharded = True
        self.prefetch_handle = None
        self.prefetch_data = None
        self.param_buffer_start_index = 0
        self.param_buffer_end_index = 0
        self.grad_buffer_start_index = 0
        self.grad_buffer_end_index = 0
        self._init_rank_info()
        self._init_param_shard_size()
        self._init_param()
        self.dp_size = self.rank_size // self.shard_size
        self.sharded_group_name = self._create_sharded_dp_group()
        self.unsharded_group_name = self._create_unsharded_dp_group()

    def _init_param_shard_size(self):
        """init parameter dp shard size"""
        if hasattr(self.param, "hsdp_shard_size"):
            if not isinstance(self.param.hsdp_shard_size, int) or \
                    (self.param.hsdp_shard_size <= 0 and self.param.hsdp_shard_size != -1):
                raise ValueError(f"param's hsdp_shard_size must be a positive integer, "
                                 f"but got {self.param.hsdp_shard_size}.")
            self.shard_size = self.param.hsdp_shard_size
        else:
            self.shard_size = self.config.shard_size
        local_shape = self.param.local_shape if isinstance(self.param, DTensor) else self.param.shape
        if len(local_shape) < 1:
            self.shard_size = 1
            return
        param_size = functools.reduce(lambda x, y: x * y, local_shape, type_size_in_bytes(self.param.dtype))
        if param_size < self.config.threshold:
            self.shard_size = 1
            return
        if self.shard_size == -1 or local_shape[0] < self.shard_size:
            self.shard_size = local_shape[0]

        def _gcd(m, n):
            if m < n:
                m, n = n, m
            if n == 0:
                raise ValueError("HSDP invalid gcd input 0.")
            r = m % n
            if r == 0:
                return n
            return _gcd(n, r)

        rank_gcd = _gcd(local_shape[0], self.rank_size)
        self.shard_size = min(self.shard_size, rank_gcd)
        if rank_gcd % self.shard_size != 0:
            self.shard_size = 1
        self.param.hsdp_effective_shard_size = self.shard_size

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
        if not isinstance(self.param, DTensor):
            rank_base = self.local_rank // self.shard_size * self.shard_size
            rank_list = [i + rank_base for i in range(self.shard_size)]
            return rank_list

        rank_base = self.hsdp_rank // self.shard_size * self.shard_size
        hsdp_rank_list = [i + rank_base for i in range(self.shard_size)]
        return self._hsdp_rank_to_global_rank(hsdp_rank_list)

    def _get_dp_rank_list(self):
        """get optimizer parallel rank list"""
        if not isinstance(self.param, DTensor):
            rank_stride = self.shard_size
            rank_base = self.local_rank % rank_stride
            rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
            return rank_list

        rank_stride = self.shard_size
        rank_base = self.hsdp_rank % rank_stride
        hsdp_rank_list = [i * rank_stride + rank_base for i in range(self.dp_size)]
        return self._hsdp_rank_to_global_rank(hsdp_rank_list)

    def _create_sharded_dp_group(self):
        """create communication group for sharded parameter"""
        if self.shard_size <= 1:
            return "hsdp_sharded_dp_group_invalid"

        rank_list = self._get_op_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_sharded_dp_group_" + rank_list_str
        comm.create_group_if_needed(group_name, rank_list)
        return group_name

    def _create_unsharded_dp_group(self):
        """create communication group for unsharded parameter"""
        if self.dp_size <= 1:
            return "hsdp_unsharded_dp_group_invalid"

        rank_list = self._get_dp_rank_list()
        rank_list_str = "_".join([str(i) for i in rank_list])
        group_name = "hsdp_unshared_dp_group_" + rank_list_str
        comm.create_group_if_needed(group_name, rank_list)
        return group_name

    def _init_sharded_param(self):
        """add and init sharded param"""
        if not self.param.has_init:
            slice_index = self.hsdp_rank % self.shard_size
            local_param = self.param.to_local() if isinstance(self.param, DTensor) else self.param
            param_slice = ops.split(local_param, local_param.shape[0] // self.shard_size)[slice_index] + 0
            self._update_param_data(param_slice)
            self.sharded_param = Parameter(param_slice,
                                           name="sharded_"+self.param.name,
                                           requires_grad=self.param.requires_grad)
        else:
            dp_slice_index = self.hsdp_rank % self.shard_size
            data_slice_index = self.tp_rank * self.shard_size + dp_slice_index
            init_mode_local_shape = self.param.init_mode.local_shape if isinstance(self.param.init_mode, DTensor) \
                else self.param.init_mode.shape
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
        if self.config.use_pynative_hook:
            return
        local_shape = self.param.local_shape if isinstance(self.param, DTensor) else self.param.shape
        self.unsharded_param = Parameter(initializer("zeros", local_shape, self.param.dtype),
                                         name="unsharded_"+self.param.name,
                                         requires_grad=self.param.requires_grad)
        self.unsharded_param_available = Parameter(Tensor(False),
                                                   name="available_unsharded_"+self.param.name,
                                                   requires_grad=False)

    def _init_param(self):
        """init hsdp parameter"""
        self.param.acc_grad = None
        local_shape = self.param.local_shape if isinstance(self.param, DTensor) else self.param.shape
        param_size = functools.reduce(lambda x, y: x * y, local_shape, type_size_in_bytes(self.param.dtype))
        if self.shard_size == 1 or param_size < self.config.threshold:
            self.sharded = False
            self.fully_sharded = False
            if self.config.requires_acc_grad and self.param.requires_grad:
                acc_grad_type = self.param.dtype
                if self.config.reduce_dtype is not None:
                    acc_grad_type = self.config.reduce_dtype
                self.acc_grad = Parameter(initializer("zeros", local_shape, acc_grad_type),
                                          name="acc_grad_"+self.param.name,
                                          requires_grad=False)
            self.param.acc_grad = self.acc_grad
            return

        origin_param_shape = list(local_shape)
        self._init_unsharded_param()
        self._init_sharded_param()
        if self.config.requires_acc_grad and self.param.requires_grad:
            acc_grad_shape = origin_param_shape
            if self.config.shard_level != OptimizerLevel.SHARD_OPT:
                acc_grad_shape = self.sharded_param.shape
            acc_grad_type = self.param.dtype
            if self.config.reduce_dtype is not None:
                acc_grad_type = self.config.reduce_dtype
            self.acc_grad = Parameter(initializer("zeros", acc_grad_shape, acc_grad_type),
                                      name="acc_grad_"+self.param.name,
                                      requires_grad=False)
        self.param.acc_grad = self.acc_grad
        self.sharded = True
        if self.shard_size == self.rank_size:
            self.fully_sharded = True
        else:
            self.fully_sharded = False

    #pylint: disable=W0212
    def _update_param_data(self, data):
        """update param data"""
        if isinstance(self.param, DTensor):
            self.param.set_data(data)
        else:
            self.param._update_data(data)

    @_no_grad()
    def to_sharded(self):
        """change parameter to sharded state"""
        self._update_param_data(self.sharded_param)

    @_no_grad()
    def prefetch_unsharded(self):
        """prefetch unsharded param with async all gather"""
        if self.prefetch_handle is not None:
            return
        unshared_param_data, handle = comm.all_gather_into_tensor(self.param.to_local()
                                                                  if isinstance(self.param, DTensor)
                                                                  else self.param, group=self.sharded_group_name,
                                                                  async_op=True)
        self.prefetch_data = unshared_param_data
        self.prefetch_handle = handle

    #pylint: disable=W0212
    @_no_grad()
    def to_unsharded(self):
        """change parameter to unsharded state"""
        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            self.sharded_param._update_data(self.param)
            self._update_param_data(self.prefetch_data)
            self.prefetch_handle = None
            self.prefetch_data = None
            return

        unshared_param_data, _ = comm.all_gather_into_tensor(self.param.to_local()
                                                             if isinstance(self.param, DTensor) else self.param,
                                                             group=self.sharded_group_name, async_op=False)
        self.sharded_param._update_data(self.param.to_local() if isinstance(self.param, DTensor) else self.param)
        self._update_param_data(unshared_param_data)

    @_no_grad()
    def zero_acc_grad(self):
        """zero accumunication grad"""
        if self.param.acc_grad is not None:
            self.param.acc_grad.zero_()
