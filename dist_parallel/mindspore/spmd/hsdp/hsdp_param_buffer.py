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
"""HSDP parameter buffer"""
import mindspore as ms
from mindspore.common.api import _no_grad
from mindspore.common.tensor import Tensor
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm

import dist_parallel

class HSDPParamBuffer:
    """
    HSDP parameter buffer.
    """
    def __init__(self, config, init_hsdp_param):
        self.config = config
        self.shard_size = init_hsdp_param.shard_size
        self.local_rank = init_hsdp_param.hsdp_rank % init_hsdp_param.shard_size
        self.dtype = init_hsdp_param.param.dtype
        self.sharded_group_name = init_hsdp_param.sharded_group_name
        self.hsdp_params = []
        self.numel = 0
        self.sharded_param_buffer = None
        self.unshared_param_buffer = None
        self.prefetch_handle = None
        self.prefetch_data = None

    def init(self):
        """init buffer"""
        self.numel = 0
        for hsdp_param in self.hsdp_params:
            start_index = self.numel
            end_index = start_index + hsdp_param.sharded_param.numel()
            hsdp_param.param_buffer_start_index = start_index
            hsdp_param.param_buffer_end_index = end_index
            self.numel = end_index
        self._init_param_buffer()

    def _init_param_buffer(self):
        """init params buffer"""
        self.sharded_param_buffer = Tensor(shape=(self.numel), dtype=self.dtype)
        for hsdp_param in self.hsdp_params:
            start_index = hsdp_param.param_buffer_start_index
            end_index = hsdp_param.param_buffer_end_index
            self.sharded_param_buffer[start_index:end_index] = hsdp_param.sharded_param.reshape(-1)
            local_shape = hsdp_param.sharded_param.local_shape \
                if isinstance(hsdp_param.sharded_param, dist_parallel.DTensor) else hsdp_param.sharded_param.shape
            data = self.sharded_param_buffer[start_index:end_index].view(local_shape)
            hsdp_param.sharded_param_view = data

    def add_param(self, hsdp_param):
        """add param to buffer"""
        self.hsdp_params.append(hsdp_param)

    @_no_grad()
    def to_sharded(self):
        """change parameter to sharded state"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.sharded_param[:] = hsdp_param.sharded_param_view[:]
            hsdp_param.to_sharded()
        self.unshared_param_buffer = None

    @_no_grad()
    def _update_data_view(self):
        for hsdp_param in self.hsdp_params:
            hsdp_param.sharded_param_view[:] = hsdp_param.param[:]

    @_no_grad()
    def prefetch_unsharded(self):
        """prefetch unsharded params with async all gather"""
        if self.prefetch_handle is not None:
            return
        self._update_data_view()
        unshared_param_buffer, handle = comm.all_gather_into_tensor(self.sharded_param_buffer,
                                                                    group=self.sharded_group_name,
                                                                    async_op=True)
        self.prefetch_data = unshared_param_buffer
        self.prefetch_handle = handle

    @_no_grad()
    def to_unsharded(self):
        """change parameter to unsharded state"""
        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            unshared_param_buffer = self.prefetch_data
            self.prefetch_handle = None
            self.prefetch_data = None
        else:
            self._update_data_view()
            unshared_param_buffer, _ = comm.all_gather_into_tensor(self.sharded_param_buffer,
                                                                   group=self.sharded_group_name,
                                                                   async_op=False)
        unshared_param_buffer = unshared_param_buffer.view((self.shard_size, -1))
        for hsdp_param in self.hsdp_params:
            start_index = hsdp_param.param_buffer_start_index
            end_index = hsdp_param.param_buffer_end_index
            unshared_param_data = unshared_param_buffer[:, start_index:end_index]
            local_shape = hsdp_param.sharded_param.local_shape \
                if isinstance(hsdp_param.sharded_param, dist_parallel.DTensor) else hsdp_param.sharded_param.shape
            unsharded_shape = list(local_shape)
            unsharded_shape[0] = unsharded_shape[0] * self.shard_size
            unshared_param_data = unshared_param_data.view(unsharded_shape)
            hsdp_param.param.set_data(unshared_param_data)
        self.unshared_param_buffer = unshared_param_buffer
