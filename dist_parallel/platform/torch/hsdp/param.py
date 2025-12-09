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
from dist_parallel.core.hsdp.hsdp_param import HSDPParam


class TorchHSDPParam(HSDPParam):
    """
    Torch HSDP parameter.
    """
    def _init_rank_info(self):
        """init parameter rank info"""
        self.rank_id = dist.get_rank()
        self.rank_size = dist.get_world_size()
        self.hsdp_rank = self.rank_id
        self.local_rank = self.rank_id
        self.tp_rank = 0

    def _init_sharded_param(self):
        """add and init sharded param"""
        slice_index = self.hsdp_rank % self.shard_size
        param_slice = torch.trunk(self.param, self.shard_size, 0)[slice_index]
        self.param.data = param_slice
        self.sharded_param = param_slice

    def _init_unsharded_param(self):
        """add and init unshared param"""
        self.unsharded_param = torch.empty(self.param_shape, dtype=self.param.dtype)

    def _update_param_data(self, param, data):
        """update param data"""
        param.data = data

    def _get_unsharded_param_data(self, comm_async):
        """get unsharded param data with async comm"""
        handle = dist.all_gather_into_tensor(self.unsharded_param, self.param, group=self.sharded_group,
                                             async_op=comm_async)
        return self.unsharded_param, handle
