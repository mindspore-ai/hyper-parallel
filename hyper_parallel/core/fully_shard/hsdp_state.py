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
from typing import List, Set, Tuple, Union

from hyper_parallel.platform import get_platform
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, OffloadPolicy, CommFusionPolicy
from hyper_parallel.tools.logging import get_logger

logger = get_logger("FSDP")

platform = get_platform()

ModuleClass = platform.Module
ParameterClass = platform.Parameter


class HSDPState:
    """HSDP state for cell"""

    def __init__(
        self,
        cell: Union[ModuleClass, Tuple[ModuleClass, ...]],
        mesh,
        shard_placement_fn,
        comm_fusion_policy: CommFusionPolicy,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
        raw_ignored_params: Set[ParameterClass],
        raw_replicate_params: set[ParameterClass],
        platform_impl,
        scheduler_ctx,
        device=None,
    ):
        """
        Initialize HSDPState.

        Args:
            cell: Module or modules managed as one fully_shard unit.
            mesh: Explicit data-parallel device mesh.
            shard_placement_fn: Optional function selecting the parameter shard dimension.
            comm_fusion_policy: Communication fusion configuration.
            mp_policy: Mixed-precision policy.
            offload_policy: Parameter offload policy.
            raw_ignored_params: Parameters excluded from fully_shard management.
            raw_replicate_params: Managed parameters that remain replicated.
            platform_impl: Platform abstraction layer (Torch or MindSpore).
            scheduler_ctx: Scheduler context shared by this module tree.
            device: Optional target device for parameters.
        """
        self.modules = (cell,) if isinstance(cell, platform.Module) else tuple(cell)
        self.cell = self.modules[0]
        self.mesh = mesh
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.comm_fusion_policy = comm_fusion_policy
        self.raw_ignored_params = set(raw_ignored_params or ())
        self.raw_replicate_params = set(raw_replicate_params or ())
        self.platform = platform_impl
        self.scheduler_ctx = scheduler_ctx
        self.device = device
        self.hsdp_params: List[HSDPParamV2] = []
        self.param_group = None
        self._move_states_to_device()
        self._init_hsdp_params()
        self.is_shard = True
        self.module_name = None
        # requires_gradient_sync
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        self.set_reduce_op_type("avg")
        self._reset_sharded_params = False

    def __repr__(self) -> str:
        """Stable debug name used in log lines.

        ``module_name`` is only assigned by the root forward pre-hook, so fall
        back to the managed module class and object id before names are set.
        Logging's ``%s`` calls this lazily -- only when a record is emitted.
        """
        if self.module_name:
            return str(self.module_name)
        return f"{self.cell.__class__.__name__}@{id(self.cell):x}"

    def _init_hsdp_params(self):
        """init hsdp parameters for cell"""
        raise NotImplementedError("HSDPState subclasses must implement _init_hsdp_params")

    def _move_states_to_device(self):
        """move states to device"""
        raise NotImplementedError("HSDPState subclasses must implement _move_states_to_device")

    def set_reduce_op_type(self, reduce_op_type: str) -> None:
        """Set the gradient reduction operation for the current backend."""
        raise NotImplementedError("HSDPState subclasses must implement set_reduce_op_type")

    def shard(self) -> None:
        """change parameters to sharded state"""
        logger.debug(
            "action=reshard module=%s params=%s",
            self,
            self.hsdp_params,
        )
        if self.is_shard:
            return
        for param in self.hsdp_params:
            param.to_sharded()
        self.is_shard = True

    def unshard(self, async_op: bool = False) -> None:
        """change parameters to unsharded state"""
        logger.debug(
            "action=unshard module=%s async_op=%s params=%s",
            self,
            async_op,
            self.hsdp_params,
        )
        if not self.is_shard:
            return

        if self.comm_fusion_policy.enable_comm_fusion and self.param_group is not None:
            self.param_group.unshard(async_op)
        else:
            for param in self.hsdp_params:
                param.unshard(async_op)
        if not async_op:
            self.wait_for_unshard()

    def prefetch(self) -> None:
        """prefetch unsharded parameters"""
        logger.debug(
            "action=prefetch module=%s params=%s",
            self,
            self.hsdp_params,
        )
        self.unshard(async_op=True)

    def wait_for_unshard(self) -> None:
        """wait for all unshard parameters"""
        logger.debug(
            "action=wait_unshard module=%s params=%s",
            self,
            self.hsdp_params,
        )
        if not self.is_shard:
            return

        if self.comm_fusion_policy.enable_comm_fusion and self.param_group is not None:
            self.param_group.wait_for_unshard()
        else:
            for param in self.hsdp_params:
                param.wait_for_unshard()
        self.is_shard = False

    def set_gradient_scaling_factor(self, factor):
        """Propagate the gradient scaling factor to the layer that applies it.

        The factor is consumed on the reduce input: ``param_group.foreach_reduce``
        for the fused (comm_fusion) path, or per-parameter ``reduce_scatter_grad``
        / ``all_reduce_grad`` otherwise. The state does not hold a copy.
        """
        if self.param_group is not None:
            self.param_group.gradient_scaling_factor = factor
        else:
            for hsdp_param in self.hsdp_params:
                hsdp_param.gradient_scaling_factor = factor

    def set_requires_all_reduce(self, requires_all_reduce: bool) -> None:
        """Propagate the HSDP all-reduce switch to the active communication path."""
        self.requires_all_reduce = requires_all_reduce
        if self.param_group is not None:
            self.param_group.requires_all_reduce = requires_all_reduce
