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
"""Torch HSDP cell state"""
# pylint: disable=protected-access

from collections import defaultdict
from typing import List, Mapping, Optional

import torch

from hyper_parallel.tools.logging import get_logger
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import (
    _get_param_module_infos,
)
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    SourceShardMetaInfo,
)
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.param_group import HSDPParamGroup, AllReduceParamGroup

logger = get_logger("FSDP")


def _to_dtype_if_needed(
        tensor: torch.Tensor, dtype: Optional[torch.dtype]
) -> torch.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype.

    Args:
        tensor: The input tensor to potentially cast.
        dtype: Target dtype. If None or same as tensor dtype, no-op.
    """
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


class TorchHSDPStateV2(HSDPState):
    """Torch HSDP cell state"""
    def __init__(
        self,
        cell,
        mesh,
        shard_placement_fn,
        comm_fusion_policy,
        mp_policy,
        offload_policy,
        raw_ignored_params,
        raw_replicate_params,
        platform,
        scheduler_ctx,
        device,
        source_shard_infos: Optional[Mapping[torch.nn.Parameter, SourceShardMetaInfo]] = None,
    ):
        """
        Initialize TorchHSDPStateV2.

        Args:
            cell (nn.Module): The module whose parameters are managed by this state.
            mesh: Mesh topology for shard/replicate dimensions.
            comm_fusion:
            mp_policy:
            offload_policy:
            platform (TorchPlatform): Torch platform abstraction.
            device (torch.device): Target device.
        """
        self.source_shard_infos = source_shard_infos
        super().__init__(
            cell,
            mesh,
            shard_placement_fn,
            comm_fusion_policy,
            mp_policy,
            offload_policy,
            raw_ignored_params,
            raw_replicate_params,
            platform,
            scheduler_ctx,
            device,
        )
        self._init_param_group()

    def _init_param_group(self):
        """Initialize fused parameter group for communication fusion.

        All managed parameters enter one ``HSDPParamGroup``. Parameters without
        an FSDP shard dimension use the group's local all-gather and
        reduce-scatter paths before entering replicate all-reduce buckets.
        """
        self.param_group = None
        if not self.comm_fusion_policy.enable_comm_fusion:
            return
        if self.hsdp_params:
            # pylint: disable=E1128
            self.param_group = HSDPParamGroup(
                self.hsdp_params,
                self.device,
                self.comm_fusion_policy.comm_fusion_zero_copy,
                comm_ctx=self.scheduler_ctx.param_group_comm_ctx,
            )

    def _move_states_to_device(self):
        """move states to device"""
        for mod in self.modules:
            for param in mod.parameters():
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param.device == self.device or param.device.type == "meta":
                    continue
                param.data = param.to(self.device)
            for buffer in mod.buffers():
                if buffer.device == self.device or buffer.device.type == "meta":
                    continue
                buffer.data = buffer.to(self.device)

    def _build_param_source_shard_info(
        self, param: torch.nn.Parameter
    ) -> Optional[SourceShardMetaInfo]:
        """Build normalized source-layout metadata for one managed parameter."""
        if isinstance(param, DTensor):
            if self.source_shard_infos is not None:
                raise ValueError(
                    "source_shard_infos cannot be provided when fully_shard manages a native DTensor parameter"
                )
            return SourceShardMetaInfo(
                mesh=param.device_mesh,
                placements=tuple(param.placements),
                origin_is_dtensor=True,
            )
        if self.source_shard_infos is None:
            return None
        return self.source_shard_infos.get(param)

    def _init_hsdp_params(self):
        """Initialize all fully_shard-managed parameters for the module."""
        # all parameters in the module tree(s), deduplicated
        visited_params = set()
        filtered_params = []
        for mod in self.modules:
            for _, param in mod.named_parameters():
                if param in self.raw_ignored_params:
                    continue
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param in visited_params:
                    continue
                visited_params.add(param)
                filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, tuple(self.modules))
        for param, module_info in zip(filtered_params, module_infos):
            self.hsdp_params.append(
                TorchHSDPParamV2(
                    param,
                    module_info,
                    self._build_param_mesh_info(param),
                    shard_placement_fn=self.shard_placement_fn,
                    mp_policy=self.mp_policy,
                    offload_policy=self.offload_policy,
                    device=self.device,
                    source_shard_info=self._build_param_source_shard_info(param),
                )
            )

    def _build_param_mesh_info(self, parameter):
        if self.mesh.ndim not in (1, 2):
            raise ValueError(
                "fully_shard only supports explicit 1D DP/FSDP meshes or 2D HSDP meshes. "
                f"Got mesh.ndim={self.mesh.ndim}."
            )
        if parameter in self.raw_replicate_params:
            return DDPMeshInfo(
                mesh=self.mesh if self.mesh.ndim == 1 else self.mesh.flatten(),
                replicate_mesh_dim=0,
            )
        if self.mesh.ndim == 1:
            return FSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=0)
        return HSDPMeshInfo(
            mesh=self.mesh,
            shard_mesh_dim=1,
            replicate_mesh_dim=0,
        )

    def _init_mp_dtypes(self):
        """Initialize mixed-precision dtypes for all managed parameters."""
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)

    def _validate_cpu_offload_params(self):
        """Validate that all parameters are on CPU when CPU offload policy is enabled."""
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        hsdp_params_not_on_cpu = [
            hsdp_param
            for hsdp_param in self.hsdp_params
            if hsdp_param.sharded_param.device.type != "cpu"
        ]
        if hsdp_params_not_on_cpu:
            raise RuntimeError(
                "HSDP parameters should be materialized on CPU when enabling CPU offloading. "
                'For example, load a CPU state dict or call module.to_empty(device="cpu"). '
                "Found following parameters on non-CPU device: "
                f"{[(p._param_fqn, p.sharded_param.device) for p in hsdp_params_not_on_cpu]}\n"
            )

    def lazy_init(self):
        """Deferred initialization: reset sharded params, validate devices, and set mixed-precision dtypes."""
        if self.is_shard and not self._reset_sharded_params:
            for hsdp_param in self.hsdp_params:
                hsdp_param.reset_sharded_param()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            hsdp_param._param_fqn
            for hsdp_param in self.hsdp_params
            if hsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "HSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )

    def post_backward_for_comm_fusion(self):
        """post_backward_for_comm_fusion."""
        logger.debug("post_backward module=%s mode=comm_fusion enter", self)
        # Fused gradient reduction path: first apply any pending async reduction
        # from the previous module's backward (pipelined overlap), then issue
        # this module's fused reduce-scatter (+ all-reduce for HSDP).
        comm_ctx = self.scheduler_ctx.param_group_comm_ctx
        # Phase 2: save gradients for the param group whose all-reduce is done.
        if comm_ctx.all_reduce_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_all_reduce", self)
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_save_grad()
            comm_ctx.all_reduce_param_group = None
        # Phase 1: wait reduce_scatter, issue async all_reduce for previous layer
        if comm_ctx.pre_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_reduce_scatter", self)
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if self.param_group is not None:
            logger.debug("post_backward module=%s launch=comm_fusion_reduce_scatter", self)
            self.param_group.foreach_reducescatter(
                reduce_scatter_reduce_op=self.reduce_op_type,
            )

    def post_backward(self, *unused):  # pylint: disable=unused-argument
        """Reduce gradients and reshard parameters after backward."""
        logger.debug(
            "post_backward module=%s enter reduce_grads=%s comm_fusion=%s reshard_after_backward=%s",
            self,
            self.reduce_grads,
            self.comm_fusion_policy.enable_comm_fusion,
            self.reshard_after_backward,
        )
        for hsdp_param in self.hsdp_params:
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.shard()
            for hsdp_param in self.hsdp_params:
                hsdp_param.to_accumulated_grad_if_needed()
            return
        if self.reshard_after_backward:
            # Reshard before gradient communication to reduce backward memory peak.
            self.shard()
        if not self.comm_fusion_policy.enable_comm_fusion:
            # Step 1: wait previous reduce-scatter (for params needing all-reduce)
            prev_group = self._wait_prev_reduce_scatter()

            # Step 2: wait previous reduce-scatter outputs that skip replicate all-reduce
            self._wait_prev_reduce_scatter_without_all_reduce()

            # Step 3: issue current reduce_scatter
            self._issue_reduce_scatter_for_current_module()

            # Step 4: issue previous fused all-reduce asynchronously
            self._issue_prev_fused_all_reduce(prev_group)
        else:
            self.post_backward_for_comm_fusion()

    def _issue_reduce_scatter_for_current_module(self):
        """Issue reduce_scatter for current module's parameters with fused all-reduce support.

        This method groups parameters by their replicate_process_group and:
        1. For params without all_reduce needs: issue reduce_scatter directly
        2. For params with all_reduce needs: allocate fused buffer and issue reduce_scatter
           into aligned views, enabling zero-copy fused all_reduce later.
        """
        # Collect parameters that need gradient reduction
        params_to_reduce = []
        for hsdp_param in self.hsdp_params:
            skip_param = (
                not hsdp_param.unsharded_param_buffers
                or not hsdp_param.sharded_param.requires_grad
                or (
                    hsdp_param.unsharded_param.grad is None
                    and hsdp_param.unsharded_accumulated_grad_data is None
                )
            )
            if skip_param:
                continue
            params_to_reduce.append(hsdp_param)

        if not params_to_reduce:
            return

        # Group by replicate process group and reduction dtype so every fused
        # all-reduce buffer has one communication group and one element type.
        groups_by_comm = defaultdict(list)
        for hsdp_param in params_to_reduce:
            if self.requires_all_reduce and hsdp_param.replicate_world_size > 1:
                replicate_process_group = hsdp_param.mesh_info.replicate_process_group
                group_key = (id(replicate_process_group), hsdp_param.reduce_comm_dtype())
                groups_by_comm[group_key].append(hsdp_param)
            else:
                groups_by_comm[None].append(hsdp_param)

        # Handle params that don't need all_reduce (FSDP or single replica)
        if None in groups_by_comm:
            for hsdp_param in groups_by_comm[None]:
                logger.debug(
                    "post_backward module=%s launch=reduce_scatter param=%s all_reduce=False",
                    self,
                    hsdp_param,
                )
                hsdp_param.reduce_scatter_grad(
                    reduce_op=self.reduce_op_type,
                )
                self.scheduler_ctx.pre_reduce_scatter_params.append(hsdp_param)

        # Handle params that need all_reduce (HSDP with multiple replicas)
        for group_key, hsdp_params in groups_by_comm.items():
            if group_key is None:
                continue

            # Create AllReduceParamGroup for fused all-reduce
            group = AllReduceParamGroup(
                replicate_group=hsdp_params[0].mesh_info.replicate_process_group,
                hsdp_params=hsdp_params,
                reduce_op=self.reduce_op_type,
            )

            # Allocate fused buffer with 512-byte alignment
            group.allocate_fused_buffer(self.device)

            # Issue reduce_scatter with output directly into fused buffer views
            logger.debug(
                "post_backward module=%s launch=fused_reduce_scatter group_params=%s",
                self,
                hsdp_params,
            )
            for idx, hsdp_param in enumerate(hsdp_params):
                buffer_view = group.get_param_buffer_view(idx)
                hsdp_param.reduce_scatter_grad(
                    reduce_op=self.reduce_op_type,
                    output_buffer=buffer_view,
                )

            # Save the group so the next module hook can wait RS and launch AR.
            self.scheduler_ctx.pre_all_reduce_groups.append(group)

    def _wait_prev_reduce_scatter(self) -> List[AllReduceParamGroup]:
        """Step 1: wait prev reduce_scatter.

        This enables overlapping:
        - Layer N-1's reduce_scatter wait with Layer N's backward compute

        Returns:
            List of previous AllReduceParamGroups (one per communication group).
        """
        if self.scheduler_ctx.pre_all_reduce_groups:
            prev_groups = list(self.scheduler_ctx.pre_all_reduce_groups)
            self.scheduler_ctx.pre_all_reduce_groups.clear()
            for prev_group in prev_groups:
                logger.debug(
                    "post_backward module=%s wait=fused_reduce_scatter group_params=%s",
                    self,
                    prev_group.hsdp_params,
                )
                for hsdp_param in prev_group.hsdp_params:
                    hsdp_param.reduce_scatter_output()
                    hsdp_param.clear_reduce_scatter_output()
                    if hsdp_param.unsharded_accumulated_grad_data is not None:
                        hsdp_param.unsharded_accumulated_grad = None
                    elif hsdp_param.unsharded_param.grad is not None:
                        hsdp_param.unsharded_param.grad = None
            return prev_groups
        return []

    def _issue_prev_fused_all_reduce(self, prev_groups: List[AllReduceParamGroup]) -> None:
        """Step 4: issue the previous module's fused all-reduce asynchronously.

        The all-reduce work is collected in ``pending_all_reduce_groups``
        and is waited in the root backward hook.

        Args:
            prev_groups: Previous parameter groups whose all-reduce should be issued.
        """
        for prev_group in prev_groups:
            prev_group.accumulate_reduce_partial_outputs()
            logger.debug(
                "post_backward module=%s launch=fused_all_reduce group_params=%s",
                self,
                prev_group.hsdp_params,
            )
            prev_group.issue_async_allreduce()
            self.scheduler_ctx.pending_all_reduce_groups.append(prev_group)

    def _wait_prev_reduce_scatter_without_all_reduce(self) -> None:
        """Wait previous RS outputs that do not enter a replicate all-reduce.

        When the current micro-step disables all-reduce, outputs accumulate in
        ``reduce_partial_output`` without being cast or applied to the parameter.
        On the final synchronized micro-step, the partial result is merged into
        the current RS output and retained for root-hook finalization.
        """
        while self.scheduler_ctx.pre_reduce_scatter_params:
            pre_hsdp_param = self.scheduler_ctx.pre_reduce_scatter_params.pop(0)
            logger.debug(
                "post_backward module=%s wait=reduce_scatter param=%s",
                self,
                pre_hsdp_param,
            )
            reduced_grad = pre_hsdp_param.reduce_scatter_output()
            if not self.requires_all_reduce:
                if pre_hsdp_param.reduce_partial_output is None:
                    pre_hsdp_param.reduce_partial_output = reduced_grad
                else:
                    pre_hsdp_param.reduce_partial_output.add_(reduced_grad)
                pre_hsdp_param.clear_reduce_scatter_output()
            elif pre_hsdp_param.reduce_partial_output is not None:
                reduced_grad.add_(pre_hsdp_param.reduce_partial_output)
                pre_hsdp_param.reduce_partial_output = None

            if pre_hsdp_param.unsharded_accumulated_grad_data is not None:
                pre_hsdp_param.unsharded_accumulated_grad = None
            elif pre_hsdp_param.unsharded_param.grad is not None:
                pre_hsdp_param.unsharded_param.grad = None

    def wait_and_split_all_reduce_work_groups(self) -> None:
        """Wait fused all-reduce work and expose each parameter result."""
        for group in self.scheduler_ctx.pending_all_reduce_groups:
            logger.debug(
                "post_backward module=%s wait=fused_all_reduce group_params=%s",
                self,
                group.hsdp_params,
            )
            group.wait_and_split_grads()
        self.scheduler_ctx.pending_all_reduce_groups.clear()

    def reset_iter_state(self) -> None:
        """Clear Torch communication bookkeeping without clearing optimizer gradients."""
        self.scheduler_ctx.pre_reduce_scatter_params.clear()
        self.scheduler_ctx.pre_all_reduce_params.clear()
        self.scheduler_ctx.pre_all_reduce_groups.clear()
        self.scheduler_ctx.pending_all_reduce_groups.clear()
        if self.param_group is not None:
            self.param_group.reset_iter_state()
        for hsdp_param in self.hsdp_params:
            hsdp_param.allgather_comm_ctx.allgather_handle = None
            hsdp_param.allgather_comm_ctx.allgather_output = None
            hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle = None
            hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = None
            hsdp_param.all_reduce_comm_ctx.all_reduce_handle = None
            hsdp_param.all_reduce_comm_ctx.all_reduce_output = None
            hsdp_param.reduce_partial_output = None
            hsdp_param.unsharded_accumulated_grad = None
            hsdp_param._grad = None
            if hsdp_param.unsharded_param_buffers:
                hsdp_param.unsharded_param.grad = None

    def set_requires_grad_sync(self, requires_grad_sync: bool) -> None:
        """set requires grad sync flag to control gradient sync."""
        self.reduce_grads = requires_grad_sync

    def set_reduce_op_type(self, reduce_op_type: str) -> None:
        """set reduce op type for gradient reduction."""
        fsdp_support_reduce_op = {
            "sum": torch.distributed.ReduceOp.SUM,
            "avg": torch.distributed.ReduceOp.AVG,
        }
        reduce_op = reduce_op_type.lower().strip() if isinstance(reduce_op_type, str) else reduce_op_type
        reduce_op_value = fsdp_support_reduce_op.get(reduce_op)
        if reduce_op_value is None:
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}"
            )
        self.reduce_op_type = reduce_op_value

    def _sync_current_stream_if_needed(self, need_synchronize):
        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload."
                )
