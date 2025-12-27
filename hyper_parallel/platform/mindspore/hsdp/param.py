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
from mindspore.common.initializer import initializer
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.hsdp.hsdp_param import HSDPParam


class MindSporeHSDPParam(HSDPParam):
    """
    MindSpore HSDP parameter.
    """
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
