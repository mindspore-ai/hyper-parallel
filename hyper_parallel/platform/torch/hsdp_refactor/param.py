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
import torch
import torch.distributed as dist
from hyper_parallel import DTensor, DeviceMesh
from hyper_parallel.core.hsdp_refactor.hsdp_param import HSDPParamV2
from hyper_parallel.core.hsdp_refactor.hsdp_utils import OptimizerLevel, ShardedState 


class TorchHSDPParamV2(HSDPParamV2):
    """
    Torch HSDP parameter.
    """
    def __init__(self, cell, param_name, param, config, mesh, platform):
        super().__init__(cell, param_name, param, config, mesh, platform)
        
        self.unsharded_grad = None
        self.unsharded_accumulated_grad_data = None
        self._unsharded_param = None

    @torch.no_grad()
    def _init_sharded_param(self):
        mesh_ndim = self.mesh.ndim
        if mesh_ndim not in (1, 2):
            # 1D mesh for fsdp, 2D mesh for hsdp(Replicate, Shard)
            raise ValueError(f"only support hsdp 1D mesh or 2D mesh, but got {self.mesh}")
        self.is_dtensor = isinstance(self.param, DTensor)
        if self.is_dtensor:
            tp_mesh: DeviceMesh = self.param.layout.mesh
            if tp_mesh._get_root_mesh() is not self.mesh._get_root_mesh():
                raise ValueError(f"In hsdp, param DTensor TP mesh should have same root mesh with hsdp mesh.")
        local_param = self.platform.get_param_local_data(self.param)
        local_shape_record = local_param.shape
        shard_size = self.mesh.mesh_shape[-1]
        if local_shape_record[0] % shard_size != 0:
            raise NotImplementedError(f"Curently not support uneven shard on param 0-dim.")
        shard_dim = 0 if mesh_ndim == 1 else 1
        shard_index = self.mesh.get_local_rank(mesh_dim=shard_dim)
        sharded_param = torch.chunk(local_param, shard_size, 0)[shard_index] + 0
        #TODO: 用DTensor表达切分过的参数
        self.platform.update_param_data(self.param, sharded_param)
        self.sharded_param = sharded_param
        self.shard_size = shard_size
        self.dp_size = 1 if mesh_ndim == 1 else self.mesh.mesh_shape[0]
        self.sharded_state = ShardedState.SHARDED

    def _init_unsharded_param(self):
        """
        Init unsharded param only at non-parameter shard level
        """
        if self.config.shard_level != OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.unsharded_param = torch.empty(self.param_shape, dtype=self.param.dtype, device=self.param.device)
        else:
            self.unsharded_param = None

    def _get_unsharded_param_data(self, async_op):
        """get unsharded param data with async comm"""
        local_param = self.platform.get_param_local_data(self.param)
        if self.unsharded_param is not None:
            unsharded_param = self.unsharded_param
        else:
            unsharded_param = torch.empty(self.param_shape, dtype=self.param.dtype, device=self.param.device)
        handle = dist.all_gather_into_tensor(unsharded_param, local_param, group=self.sharded_group_info.group,
                                             async_op=async_op)
        return unsharded_param, handle
    
    def acc_grad_if_needed(self):
        #TODO: HSDPParam 梯度处理
        if self.unsharded_accumulated_grad_data is not None and self._unsharded_param.grad is not None:
            self.unsharded_accumulated_grad_data += self._unsharded_param.grad
            self._unsharded_param.grad = None
    
    def to_accumulated_grad_if_needed(self):
        #TODO: HSDPParam 梯度处理
        if self._unsharded_param.grad is not None and self.unsharded_accumulated_grad_data is None:
            self.unsharded_accumulated_grad_data = self.param.grad
            self.param.grad = None

