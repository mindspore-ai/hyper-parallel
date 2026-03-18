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
from typing import List
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import HSDPConfigV2
from hyper_parallel.platform.torch.fully_shard.param_group import HSDPParamGroup

class HSDPState:
    """HSDP state for cell"""
    def __init__(self, cell, mesh_info, config: HSDPConfigV2, platform, device=None):
        """
        Initialize HSDPState.

        Args:
            cell (nn.Module): The module whose parameters are managed by this state.
            mesh_info: Mesh topology for shard/replicate dimensions.
            config (HSDPConfigV2): HSDP configuration (mesh, mp_policy, offload_policy, etc.).
            platform: Platform abstraction layer (Torch or MindSpore).
            device (torch.device, optional): Target device for parameters.
        """
        self.cell = cell
        self.mesh_info = mesh_info
        self.config = config
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.platform = platform
        self.device = device
        self.hsdp_params: List[HSDPParamV2] = []
        self.sharded_hsdp_params: List[HSDPParamV2] = []
        self._move_states_to_device()
        self._init_hsdp_params()
        self._init_param_group()
        self.is_shard = True
        self.module_name = None

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        raise NotImplementedError("HSDPState subclasses must implement _init_hsdp_params")

    def _move_states_to_device(self):
        """move states to device"""
        raise NotImplementedError("HSDPState subclasses must implement _move_states_to_device")

    def _init_param_group(self):
        """Initialize fused parameter group for communication fusion.

        When ``comm_fusion`` is enabled, creates an ``HSDPParamGroup`` that packs all
        parameters into a single buffer for fused all-gather and reduce-scatter,
        replacing the per-parameter communication pattern.
        """
        if not self.config.comm_fusion:
            return
        self.param_group = HSDPParamGroup(self.hsdp_params, self.mesh_info, self.device, self.mp_policy)

    def shard(self):
        """change parameters to sharded state"""
        if self.is_shard:
            return

        for param in self.sharded_hsdp_params:
            param.to_sharded()
        self.is_shard = True
        return

    def unshard(self, async_op=False):
        """change parameters to unsharded state"""
        if not self.is_shard:
            return

        if self.config.comm_fusion:
            self.param_group.unshard(async_op)
        else:
            for param in self.sharded_hsdp_params:
                param.unshard(async_op)
        if not async_op:
            self.wait_for_unshard()

    def prefetch(self):
        """prefetch unsharded parameters"""
        self.unshard(async_op=True)

    def wait_for_unshard(self):
        """wait for all unshard parameters"""
        if not self.is_shard:
            return
        if self.config.comm_fusion:
            self.param_group.wait_for_unshard()
        else:
            for param in self.sharded_hsdp_params:
                param.wait_for_unshard()
        self.is_shard = False
