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
"""HSDP cell state"""
from mindspore.common.api import _no_grad
from mindspore.common.dtype import type_size_in_bytes
from dist_parallel.spmd.hsdp.hsdp_param import HSDPParam
from dist_parallel.spmd.hsdp.hsdp_param_buffer import HSDPParamBuffer
from dist_parallel.spmd.hsdp.hsdp_grad_buffer import HSDPGradBuffer


class HSDPState:
    """HSDP state for cell"""
    def __init__(self, cell, config):
        self.cell = cell
        self.config = config
        self.hsdp_params = []
        self.sharded_hsdp_params = []
        self.param_buffers = []
        self.grad_buffers = []
        self._init_hsdp_params()
        self._init_param_buffers()
        self._init_grad_buffers()
        self.is_shard = True

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        cells = self.cell.cells_and_names()
        for _, sub_cell in cells:
            params = sub_cell._params.items() #pylint: disable=W0212
            for param_name, param in params:
                if hasattr(param, "has_hsdp_param"):
                    continue
                hsdp_param = HSDPParam(sub_cell, param_name, param, self.config)
                param.has_hsdp_param = True
                self.hsdp_params.append(hsdp_param)
                if hsdp_param.sharded:
                    self.sharded_hsdp_params.append(hsdp_param)

    def _init_param_buffers(self):
        """init param buffers"""
        if not self.config.comm_fusion:
            return

        group_to_buffer = {}
        for hsdp_param in self.sharded_hsdp_params:
            param_buffer_key = hsdp_param.sharded_group_name + str(hsdp_param.param.dtype)
            if param_buffer_key not in group_to_buffer:
                buffer = HSDPParamBuffer(self.config, hsdp_param)
                buffer.add_param(hsdp_param)
                group_to_buffer[param_buffer_key] = buffer
            else:
                buffer = group_to_buffer[param_buffer_key]
                buffer.add_param(hsdp_param)
        self.param_buffers = list(group_to_buffer.values())
        for buffer in self.param_buffers:
            buffer.init()

    def _init_grad_buffers(self):
        """init grad buffers"""
        if not self.config.grad_fusion:
            return

        bucket_infos = {}
        def get_bucket_key(buffer_key, hsdp_param):
            if self.config.bucket_size < 0:
                return buffer_key
            param_size = hsdp_param.param.numel() * type_size_in_bytes(hsdp_param.param.dtype)
            bucket_info = bucket_infos.get(buffer_key, None)
            if bucket_info is None:
                bucket_info = [0, param_size]
                bucket_infos[buffer_key] = bucket_info
            else:
                bucket_size = bucket_info[1] + param_size
                if bucket_size > self.config.bucket_size:
                    bucket_info[0] = bucket_info[0] + 1
                    bucket_info[1] = param_size
            return buffer_key + '_' + str(bucket_info[0])

        self.param_to_buffer = {}
        group_to_buffer = {}
        for hsdp_param in self.hsdp_params:
            if not hsdp_param.param.requires_grad:
                continue
            buffer_key = hsdp_param.sharded_group_name + hsdp_param.unsharded_group_name + str(hsdp_param.param.dtype)
            bucket_key = get_bucket_key(buffer_key, hsdp_param)
            if bucket_key not in group_to_buffer:
                buffer = HSDPGradBuffer(self.config, hsdp_param)
                group_to_buffer[bucket_key] = buffer
            else:
                buffer = group_to_buffer[bucket_key]
            buffer.add_param(hsdp_param)
            self.param_to_buffer[hsdp_param] = buffer
        self.grad_buffers = list(group_to_buffer.values())
        for buffer in self.grad_buffers:
            buffer.init()

    @_no_grad()
    def shard(self):
        """change parameters to sharded state"""
        if self.is_shard:
            return

        if self.config.comm_fusion:
            for buffer in self.param_buffers:
                buffer.to_sharded()
        else:
            for param in self.sharded_hsdp_params:
                param.to_sharded()
        self.is_shard = True

    @_no_grad()
    def unshard(self):
        """change parameters to unsharded state"""
        if not self.is_shard:
            return

        if self.config.comm_fusion:
            for buffer in self.param_buffers:
                buffer.to_unsharded()
        else:
            for param in self.sharded_hsdp_params:
                param.to_unsharded()
        self.is_shard = False

    @_no_grad()
    def prefetch(self):
        """prefetch unsharded parameters"""
        if not self.is_shard:
            return
        if self.config.comm_fusion:
            for buffer in self.param_buffers:
                buffer.prefetch_unsharded()
        else:
            for param in self.sharded_hsdp_params:
                param.prefetch_unsharded()

    @_no_grad()
    def zero_grads(self):
        if not self.config.grad_fusion:
            for hsdp_param in self.hsdp_params:
                if not hsdp_param.param.requires_grad:
                    continue
                hsdp_param.zero_acc_grad()
        else:
            for buffer in self.grad_buffers:
                buffer.zero_grads()

    @_no_grad()
    def set_grad_ready(self, hsdp_param):
        """set grad ready"""
        if not self.config.grad_fusion:
            return
        buffer = self.param_to_buffer.get(hsdp_param, None)
        if buffer is not None:
            buffer.set_grad_ready(hsdp_param)
        else:
            raise ValueError(f"param {hsdp_param.param} is not register to buffer.")

    @_no_grad()
    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        if not self.config.grad_fusion:
            return
        for buffer in self.grad_buffers:
            buffer.set_requires_grad_sync(requires_grad_sync)
