# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
from typing import List, Tuple, Union

from hyper_parallel.platform import get_platform
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import HSDPConfigV2, ShardedState

platform = get_platform()


class HSDPState:
    """HSDP state for cell"""
    # Record pending per-parameter reduce-scatter/all-reduce work across
    # fully_shard states so later backward hooks/root drains can materialize
    # gradients launched by earlier states.
    pre_reduce_scatter_params = []
    pre_all_reduce_params = []

    def __init__(self, cell: Union[platform.Module, Tuple[platform.Module, ...]], mesh_info,
                 config: HSDPConfigV2, platform_impl, device=None):
        """
        Initialize HSDPState.

        Args:
            cell (platform.Module or Tuple[platform.Module, ...]): The module(s) whose parameters
                are managed by this state. When a tuple is passed, all modules are
                treated as one FSDP unit.
            mesh_info: Mesh topology for shard/replicate dimensions.
            config (HSDPConfigV2): HSDP configuration (mesh, mp_policy, offload_policy, etc.).
            platform_impl: Platform abstraction layer (Torch or MindSpore).
            device (torch.device, optional): Target device for parameters.
        """
        self.modules = (cell,) if isinstance(cell, platform.Module) else tuple(cell)
        self.cell = self.modules[0]
        self.mesh_info = mesh_info
        self.config = config
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.platform = platform_impl
        self.device = device
        self.hsdp_params: List[HSDPParamV2] = []
        self.sharded_hsdp_params: List[HSDPParamV2] = []
        self.replicate_params: List[HSDPParamV2] = []
        self._move_states_to_device()
        self._init_hsdp_params()
        self.is_shard = True
        self.is_replicate_shard = True
        self.module_name = None

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        raise NotImplementedError("HSDPState subclasses must implement _init_hsdp_params")

    def _move_states_to_device(self):
        """move states to device"""
        raise NotImplementedError("HSDPState subclasses must implement _move_states_to_device")

    def _assert_replicate_params_unsharded(self) -> None:
        """Validate replicate params are already materialized when state says so."""
        for param in self.replicate_params:
            sharded_state = getattr(param, "sharded_state", None)
            if sharded_state != ShardedState.UNSHARDED:
                param_fqn = getattr(param, "_param_fqn", "<unknown>")
                raise AssertionError(
                    f"Expected replicate parameter {param_fqn} to be "
                    f"{ShardedState.UNSHARDED}, got {sharded_state}"
                )

    def shard(self, shard_replicate: bool = True):
        """change parameters to sharded state"""
        if not self.is_shard:
            for param in self.sharded_hsdp_params:
                param.to_sharded()
            self.is_shard = True
        if shard_replicate and not self.is_replicate_shard:
            for param in self.replicate_params:
                param.to_sharded()
            self.is_replicate_shard = True

    def unshard(self, async_op=False, unshard_replicate: bool = True):
        """change parameters to unsharded state"""
        if not self.is_shard and (not unshard_replicate or not self.is_replicate_shard):
            if unshard_replicate:
                self._assert_replicate_params_unsharded()
            return

        if unshard_replicate:
            if self.is_replicate_shard:
                for param in self.replicate_params:
                    param.unshard(async_op)
            else:
                self._assert_replicate_params_unsharded()
        if self.is_shard:
            if self.config.comm_fusion and self.param_group is not None:
                self.param_group.unshard(async_op)
            else:
                for param in self.sharded_hsdp_params:
                    param.unshard(async_op)
        if not async_op:
            self.wait_for_unshard(unshard_replicate)

    def prefetch(self, unshard_replicate: bool = True):
        """prefetch unsharded parameters"""
        self.unshard(async_op=True, unshard_replicate=unshard_replicate)

    def wait_for_unshard(self, wait_for_replicate: bool = True):
        """wait for all unshard parameters"""
        if not self.is_shard and (not wait_for_replicate or not self.is_replicate_shard):
            if wait_for_replicate:
                self._assert_replicate_params_unsharded()
            return
        if wait_for_replicate:
            if self.is_replicate_shard:
                for param in self.replicate_params:
                    param.wait_for_unshard()
                self.is_replicate_shard = False
            else:
                self._assert_replicate_params_unsharded()
        if self.is_shard:
            if self.config.comm_fusion and self.param_group is not None:
                self.param_group.wait_for_unshard()
            else:
                for param in self.sharded_hsdp_params:
                    param.wait_for_unshard()
            self.is_shard = False

    def _iter_managed_params(self):
        """Return all fully_shard-managed parameters, including replicate_params."""
        return [*self.hsdp_params, *self.replicate_params]
