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
"""HSDP gradient buffer"""
from mindspore import ops
from mindspore.common.api import _no_grad
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from dist_parallel.spmd.hsdp.hsdp_utils import OptimizerLevel
from dist_parallel.spmd.hsdp.hsdp_context import hsdp_context
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm


class HSDPGradBuffer:
    """
    HSDP gradient buffer.
    """
    def __init__(self, config, init_hsdp_param):
        self.config = config
        self.shard_size = init_hsdp_param.shard_size
        self.dp_size = init_hsdp_param.dp_size
        self.local_rank = init_hsdp_param.hsdp_rank % init_hsdp_param.shard_size
        self.dtype = init_hsdp_param.param.dtype
        self.sharded_group_name = init_hsdp_param.sharded_group_name
        self.unsharded_group_name = init_hsdp_param.unsharded_group_name
        self.sharded = init_hsdp_param.sharded
        self.fully_sharded = init_hsdp_param.fully_sharded
        self.hsdp_params = []
        self.numel = 0
        self.requires_grad_sync = False
        self.sharded_grad_buffer = None
        self.unsharded_grad_buffer = None
        self.prefetch_handle = None
        self.prefetch_data = None
        self.reduce_type = self.dtype
        if self.config.reduce_dtype is not None:
            self.reduce_type = self.config.reduce_dtype

    def add_param(self, hsdp_param):
        """add param to buffer"""
        self.hsdp_params.append(hsdp_param)

    def init(self):
        """init buffer"""
        self._init_grad_buffer_index()
        self._init_acc_grad_buffer()
        self.num_grad = len(self.hsdp_params)
        self.num_ready_grad = 0

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.requires_grad_sync = requires_grad_sync

    def _set_handle(self, handle, process=None):
        """set handle for async comm and wait handle for sync comm"""
        if self.config.comm_async:
            if process is not None:
                hsdp_context.set_grad_reduce_handle(handle, process)
            else:
                hsdp_context.set_grad_reduce_handle(handle)
        else:
            handle.wait()
            if process is not None:
                process()

    def _init_grad_buffer_index(self):
        """init grad buffer index"""
        self.numel = 0
        for hsdp_param in self.hsdp_params:
            start_index = self.numel
            end_index = start_index + hsdp_param.param.numel()
            hsdp_param.grad_buffer_start_index = start_index
            hsdp_param.grad_buffer_end_index = end_index
            self.numel = end_index

    def _init_acc_grad_buffer(self):
        """init acc grad buffer"""
        if not self.config.requires_acc_grad:
            return

        acc_grad_shape = (self.shard_size, self.numel)
        self.acc_grad_buffer_dim0 = self.shard_size
        if self.config.shard_level != OptimizerLevel.SHARD_OPT:
            acc_grad_shape = (1, self.numel)
            self.acc_grad_buffer_dim0 = 1
        self.acc_grad_buffer = Parameter(initializer("zeros", acc_grad_shape, self.reduce_type), requires_grad=False)

    def _copy_grad_to_unsharded_buffer(self):
        """copy grad to buffer"""
        self.unsharded_grad_buffer = Tensor(shape=(self.shard_size, self.numel), dtype=self.reduce_type)
        for hsdp_param in self.hsdp_params:
            if hsdp_param.grad is None:
                raise ValueError("HSDP param grad can't be None with comm fusion.")
            start_index = hsdp_param.grad_buffer_start_index
            end_index = hsdp_param.grad_buffer_end_index
            buffer_view = self.unsharded_grad_buffer[:, start_index:end_index]
            if self.dtype != self.reduce_type:
                buffer_view[:] = ops.cast(hsdp_param.grad.view(self.shard_size, -1), self.reduce_type)[:]
            else:
                buffer_view[:] = hsdp_param.grad.view(self.shard_size, -1)[:]

    def _add_grad_to_acc_buffer(self):
        """add grad to acc buffer"""
        self._copy_grad_to_unsharded_buffer()
        self.acc_grad_buffer.add_(self.unsharded_grad_buffer)
        self.unsharded_grad_buffer = None

    def _set_grad_data(self, buffer=None):
        """set grad with buffer view"""
        if buffer is None:
            buffer = self.sharded_grad_buffer

        if self.dtype != self.reduce_type:
            buffer = op.cast(buffer, self.dtype)

        if self.config.grad_scale != 1.0:
            buffer = buffer * self.config.grad_scale

        for hsdp_param in self.hsdp_params:
            if hsdp_param.grad is None:
                raise ValueError("HSDP param grad can't be None with comm fusion.")
            start_index = hsdp_param.grad_buffer_start_index
            end_index = hsdp_param.grad_buffer_end_index
            view_shape = list(hsdp_param.grad.shape)
            view_shape[0] = view_shape[0] // self.shard_size
            buffer_view = buffer[:, start_index:end_index].view(view_shape)
            hsdp_param.grad.data = buffer_view + 0

    def _handle_single_node_grad(self):
        """single node don't need to reduce grad, only need to acc grad"""
        if not self.config.requires_acc_grad:
            return
        self._add_grad_to_acc_buffer()
        self._set_grad_data(self.acc_grad_buffer)

    def _reduce_no_shard_grad(self):
        """reduce grad when param is not sharded"""
        if not self.config.requires_acc_grad:
            self._copy_grad_to_unsharded_buffer()
            output, handle = comm.all_reduce(self.unsharded_grad_buffer, group=self.unsharded_group_name,
                                             async_op=True)
        else:
            self._add_grad_to_acc_buffer()
            if not self.requires_grad_sync:
                return
            output, handle = comm.all_reduce(self.acc_grad_buffer, group=self.unsharded_group_name, async_op=True)
        self.sharded_grad_buffer = output
        self._set_handle(handle, self._set_grad_data)

    def _reduce_fully_shard_grad(self):
        """reducescatter grad when param is fully sharded"""
        if not self.config.requires_acc_grad:
            self._copy_grad_to_unsharded_buffer()
            output, handle = comm.reduce_scatter_tensor(self.unsharded_grad_buffer, group=self.sharded_group_name,
                                                        async_op=True)
            self.sharded_grad_buffer = output
            self._set_handle(handle, self._set_grad_data)
        elif self.config.shard_level != OptimizerLevel.SHARD_OPT:
            self._copy_grad_to_unsharded_buffer()
            output, handle = comm.reduce_scatter_tensor(self.unsharded_grad_buffer, group=self.sharded_group_name,
                                                        async_op=True)
            self.sharded_grad_buffer = output
            def post_process():
                self.acc_grad_buffer.add_(output)
                self._set_grad_data(self.acc_grad_buffer)
            self._set_handle(handle, post_process)
        else:
            self._add_grad_to_acc_buffer()
            if not self.requires_grad_sync:
                return
            output, handle = comm.reduce_scatter_tensor(self.acc_grad_buffer, group=self.sharded_group_name,
                                                        async_op=True)
            self.sharded_grad_buffer = output
            self._set_handle(handle, self._set_grad_data)

    def _reduce_partial_shard_grad(self):
        """reduce grad after reducescatter grad when param is partial sharded"""
        if not self.config.requires_acc_grad:
            self._copy_grad_to_unsharded_buffer()
            output, _ = comm.reduce_scatter_tensor(self.unsharded_grad_buffer, group=self.sharded_group_name,
                                                   async_op=False)
            output, handle = comm.all_reduce(output, group=self.unsharded_group_name, async_op=True)
            self.sharded_grad_buffer = output
            self._set_handle(handle, self._set_grad_data)
        elif self.config.shard_level != OptimizerLevel.SHARD_OPT:
            self._copy_grad_to_unsharded_buffer()
            output, _ = comm.reduce_scatter_tensor(self.unsharded_grad_buffer, group=self.sharded_group_name,
                                                   async_op=False)
            self.acc_grad_buffer.add_(output)
            if not self.requires_grad_sync:
                return
            output, handle = comm.all_reduce(self.acc_grad_buffer, group=self.unsharded_group_name, async_op=True)
            self.sharded_grad_buffer = output
            self._set_handle(handle, self._set_grad_data)
        else:
            self._add_grad_to_acc_buffer()
            if not self.requires_grad_sync:
                return
            output, _ = comm.reduce_scatter_tensor(self.acc_grad_buffer, group=self.sharded_group_name,
                                                   async_op=False)
            output, handle = comm.all_reduce(output, group=self.unsharded_group_name, async_op=True)
            self.sharded_grad_buffer = output
            self._set_handle(handle, self._set_grad_data)

    def _reduce_grads(self):
        """reduce or accumulate grad buffer"""
        if not self.sharded:
            if self.dp_size == 1:
                self._handle_single_node_grad()
            else:
                self._reduce_no_shard_grad()
        elif self.fully_sharded:
            self._reduce_fully_shard_grad()
        else:
            self._reduce_partial_shard_grad()

    @_no_grad()
    def zero_grads(self):
        """zero grad buffer"""
        if not self.config.requires_acc_grad:
            return
        self.acc_grad_buffer.zero_()

    @_no_grad()
    def set_grad_ready(self, hsdp_param):
        """set grad ready"""
        self.num_ready_grad = self.num_ready_grad + 1
        if self.num_ready_grad == self.num_grad:
            self._reduce_grads()
            self.num_ready_grad = 0
