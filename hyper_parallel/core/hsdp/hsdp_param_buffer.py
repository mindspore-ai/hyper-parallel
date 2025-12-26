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


class HSDPParamBuffer:
    """
    HSDP parameter buffer.
    """
    def __init__(self, config, init_hsdp_param, platform):
        self.config = config
        self.platform = platform
        self.shard_size = init_hsdp_param.shard_size
        self.local_rank = init_hsdp_param.hsdp_rank % init_hsdp_param.shard_size
        self.dtype = init_hsdp_param.param.dtype
        self.sharded_group_info = init_hsdp_param.sharded_group_info
        self.device = init_hsdp_param.device
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
        self.sharded_param_buffer = self.platform.new_tensor((self.numel), self.dtype, self.device)
        for hsdp_param in self.hsdp_params:
            start_index = hsdp_param.param_buffer_start_index
            end_index = hsdp_param.param_buffer_end_index
            self.sharded_param_buffer[start_index:end_index] = hsdp_param.sharded_param.reshape(-1)
            local_shape = hsdp_param.sharded_param.shape
            data = self.sharded_param_buffer[start_index:end_index].view(local_shape)
            hsdp_param.sharded_param_view = data

    def add_param(self, hsdp_param):
        """add param to buffer"""
        self.hsdp_params.append(hsdp_param)

    def to_sharded(self):
        """change parameter to sharded state"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.sharded_param[:] = hsdp_param.sharded_param_view[:]
            hsdp_param.to_sharded()
        self.unshared_param_buffer = None

    def _update_data_view(self):
        for hsdp_param in self.hsdp_params:
            hsdp_param.sharded_param_view[:] = hsdp_param.param[:]

    def prefetch_unsharded(self):
        """prefetch unsharded params with async all gather"""
        if self.prefetch_handle is not None:
            return
        self._update_data_view()

        unshared_param_buffer, handle = self.platform.all_gather_into_tensor(self.sharded_param_buffer,
                                                                             self.sharded_group_info,
                                                                             async_op=True)
        self.prefetch_data = unshared_param_buffer
        self.prefetch_handle = handle

    def to_unsharded(self):
        """change parameter to unsharded state"""
        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            unshared_param_buffer = self.prefetch_data
            self.prefetch_handle = None
            self.prefetch_data = None
        else:
            self._update_data_view()
            unshared_param_buffer, _ = self.platform.all_gather_into_tensor(self.sharded_param_buffer,
                                                                            self.sharded_group_info,
                                                                            async_op=True)
        unshared_param_buffer = unshared_param_buffer.view((self.shard_size, -1))
        for hsdp_param in self.hsdp_params:
            start_index = hsdp_param.param_buffer_start_index
            end_index = hsdp_param.param_buffer_end_index
            unshared_param_data = unshared_param_buffer[:, start_index:end_index]
            unshared_param_data = unshared_param_data.view(hsdp_param.param_shape)
            self.platform.update_param_data(hsdp_param.param, unshared_param_data)
        self.unshared_param_buffer = unshared_param_buffer
