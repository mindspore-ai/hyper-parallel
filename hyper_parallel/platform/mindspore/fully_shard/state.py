# Copyright 2026 Huawei Technologies Co., Ltd
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
"""MindSpore HSDP state aligned with the Torch fully_shard lifecycle."""

from collections import defaultdict
from typing import List, Optional

import mindspore as ms

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import _get_param_module_infos
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    DDPMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    SourceShardMetaInfo,
)
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2
from hyper_parallel.platform.mindspore.fully_shard.param_group import (
    AllReduceParamGroup,
    HSDPParamGroup,
)
from hyper_parallel.platform.mindspore.utils import normalize_runtime_device
from hyper_parallel.tools.logging import get_logger

logger = get_logger("FSDP")


def _to_dtype_if_needed(tensor: ms.Tensor, dtype: Optional[ms.Type]) -> ms.Tensor:
    """Cast ``tensor`` only when a different MindSpore dtype is requested."""
    if isinstance(dtype, ms.Type) and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


class MindSporeHSDPStateV2(HSDPState):
    """Own MindSpore fully_shard parameters and gradient communication state."""

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
        device=None,
    ):
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

    def _init_param_group(self) -> None:
        """Initialize fused communication for the single managed parameter list."""
        self.param_group = None
        if not self.comm_fusion_policy.enable_comm_fusion or not self.hsdp_params:
            return
        self.param_group = HSDPParamGroup(
            self.hsdp_params,
            self.device,
            # MindSpore optimizers do not update view-backed Parameter storage,
            # so the supported communication-fusion path always uses copy-in.
            False,
            comm_ctx=self.scheduler_ctx.param_group_comm_ctx,
        )

    def _move_states_to_device(self) -> None:
        """Move parameters and buffers to the configured runtime device."""
        for module in self.modules:
            for param in module.get_parameters():
                if getattr(param, "_hsdp_param_initialized", False):
                    continue
                param_device = normalize_runtime_device(param.device)
                if param_device in (self.device, "meta"):
                    continue
                param.data = param.to(self.device)
            for buffer in module.buffers():
                if normalize_runtime_device(buffer.device) in (self.device, "meta"):
                    continue
                buffer.data = buffer.to(self.device)

    @staticmethod
    def _build_param_source_shard_info(param):
        """Build normalized source-layout metadata for a native DTensor parameter."""
        if not isinstance(param, DTensor):
            return None
        return SourceShardMetaInfo(
            mesh=param.device_mesh,
            placements=tuple(param.placements),
            origin_is_dtensor=True,
        )

    def _init_hsdp_params(self) -> None:
        """Initialize all fully_shard-managed parameters for this module unit."""
        visited_params = set()
        filtered_params = []
        for module in self.modules:
            for _, param in module.parameters_and_names():
                if param in self.raw_ignored_params:
                    continue
                if getattr(param, "_hsdp_param_initialized", False):
                    continue
                if param in visited_params:
                    continue
                visited_params.add(param)
                filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, tuple(self.modules))
        for param, module_info in zip(filtered_params, module_infos):
            self.hsdp_params.append(
                MindSporeHSDPParamV2(
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
        """Return the parameter-specific data-parallel route."""
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

    def _init_mp_dtypes(self) -> None:
        """Initialize mixed-precision metadata for all managed parameters."""
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)

    def _validate_cpu_offload_params(self) -> None:
        """Validate CPU placement when CPU offload is configured."""
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        params_not_on_cpu = [
            hsdp_param
            for hsdp_param in self.hsdp_params
            if not str(hsdp_param.sharded_param.device).lower().startswith("cpu")
        ]
        if params_not_on_cpu:
            raise RuntimeError(
                "HSDP parameters should be materialized on CPU when enabling CPU offloading. "
                "Found following parameters on non-CPU device: "
                f"{[(p._param_fqn, p.sharded_param.device) for p in params_not_on_cpu]}\n"
                "MindSpore backend will support this feature in future version."
            )

    def lazy_init(self) -> None:
        """Refresh parameter views and validate runtime state before execution."""
        if self.is_shard and not self._reset_sharded_params:
            for hsdp_param in self.hsdp_params:
                hsdp_param.reset_sharded_param()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _validate_no_meta_params(self) -> None:
        """Validate that managed parameters have been materialized."""
        param_names_on_meta = [
            hsdp_param._param_fqn
            for hsdp_param in self.hsdp_params
            if normalize_runtime_device(hsdp_param.sharded_param.device) == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "HSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, initialize the module weights on a real device before running training."
            )

    def zero_grad(self) -> None:
        """Clear gradients for all managed parameters."""
        for hsdp_param in self.hsdp_params:
            hsdp_param.zero_grad()

    def post_backward_for_comm_fusion(self) -> None:
        """Pipeline fused reduce-scatter and all-reduce communication."""
        logger.debug("post_backward module=%s mode=comm_fusion enter", self)
        comm_ctx = self.scheduler_ctx.param_group_comm_ctx
        if comm_ctx.all_reduce_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_all_reduce", self)
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_save_grad()
            comm_ctx.all_reduce_param_group = None
        if comm_ctx.pre_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_reduce_scatter", self)
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if self.param_group is not None:
            logger.debug("post_backward module=%s launch=comm_fusion_reduce_scatter", self)
            self.param_group.foreach_reducescatter(
                reduce_scatter_reduce_op=self.reduce_op_type,
            )

    def post_backward(self, *unused) -> None:  # pylint: disable=unused-argument
        """Accumulate gradients, reshard parameters, and launch reductions."""
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
            self.shard()
        if self.comm_fusion_policy.enable_comm_fusion:
            self.post_backward_for_comm_fusion()
            return

        previous_groups = self._wait_prev_reduce_scatter()
        self._wait_prev_reduce_scatter_without_all_reduce()
        self._issue_reduce_scatter_for_current_module()
        self._issue_prev_fused_all_reduce(previous_groups)

    def _issue_reduce_scatter_for_current_module(self) -> None:
        """Issue per-parameter reduce-scatter and fuse compatible HSDP all-reduces."""
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
            if not skip_param:
                params_to_reduce.append(hsdp_param)
        if not params_to_reduce:
            return

        groups_by_comm = defaultdict(list)
        for hsdp_param in params_to_reduce:
            if self.requires_all_reduce and hsdp_param.replicate_world_size > 1:
                replicate_group = hsdp_param.mesh_info.replicate_process_group
                group_key = (replicate_group, hsdp_param.reduce_comm_dtype())
                groups_by_comm[group_key].append(hsdp_param)
            else:
                groups_by_comm[None].append(hsdp_param)

        for hsdp_param in groups_by_comm.get(None, ()):
            logger.debug(
                "post_backward module=%s launch=reduce_scatter param=%s all_reduce=False",
                self,
                hsdp_param,
            )
            hsdp_param.reduce_scatter_grad(reduce_op=self.reduce_op_type)
            self.scheduler_ctx.pre_reduce_scatter_params.append(hsdp_param)

        for group_key, hsdp_params in groups_by_comm.items():
            if group_key is None:
                continue
            group = AllReduceParamGroup(
                replicate_group=hsdp_params[0].mesh_info.replicate_process_group,
                hsdp_params=hsdp_params,
                reduce_op=self.reduce_op_type,
            )
            group.allocate_fused_buffer(self.device)
            logger.debug(
                "post_backward module=%s launch=fused_reduce_scatter group_params=%s",
                self,
                hsdp_params,
            )
            for index, hsdp_param in enumerate(hsdp_params):
                hsdp_param.reduce_scatter_grad(
                    reduce_op=self.reduce_op_type,
                    output_buffer=group.get_param_buffer_view(index),
                )
            self.scheduler_ctx.pre_all_reduce_groups.append(group)

    def _wait_prev_reduce_scatter(self) -> List[AllReduceParamGroup]:
        """Wait previous fused reduce-scatter groups before all-reduce."""
        if not self.scheduler_ctx.pre_all_reduce_groups:
            return []
        previous_groups = list(self.scheduler_ctx.pre_all_reduce_groups)
        self.scheduler_ctx.pre_all_reduce_groups.clear()
        for previous_group in previous_groups:
            logger.debug(
                "post_backward module=%s wait=fused_reduce_scatter group_params=%s",
                self,
                previous_group.hsdp_params,
            )
            for hsdp_param in previous_group.hsdp_params:
                hsdp_param.reduce_scatter_output()
                hsdp_param.clear_reduce_scatter_output()
                if hsdp_param.unsharded_accumulated_grad_data is not None:
                    hsdp_param.unsharded_accumulated_grad = None
                elif hsdp_param.unsharded_param.grad is not None:
                    hsdp_param.unsharded_param.grad = None
        return previous_groups

    def _issue_prev_fused_all_reduce(self, previous_groups: List[AllReduceParamGroup]) -> None:
        """Launch the previous module's fused all-reduce asynchronously."""
        for previous_group in previous_groups:
            previous_group.accumulate_reduce_partial_outputs()
            logger.debug(
                "post_backward module=%s launch=fused_all_reduce group_params=%s",
                self,
                previous_group.hsdp_params,
            )
            previous_group.issue_async_allreduce()
            self.scheduler_ctx.pending_all_reduce_groups.append(previous_group)

    def _wait_prev_reduce_scatter_without_all_reduce(self) -> None:
        """Wait reduce-scatter outputs that do not enter a DP all-reduce."""
        while self.scheduler_ctx.pre_reduce_scatter_params:
            hsdp_param = self.scheduler_ctx.pre_reduce_scatter_params.pop(0)
            logger.debug(
                "post_backward module=%s wait=reduce_scatter param=%s",
                self,
                hsdp_param,
            )
            reduced_grad = hsdp_param.reduce_scatter_output()
            if not self.requires_all_reduce:
                if hsdp_param.reduce_partial_output is None:
                    hsdp_param.reduce_partial_output = reduced_grad
                else:
                    hsdp_param.reduce_partial_output.add_(reduced_grad)
                hsdp_param.clear_reduce_scatter_output()
            elif hsdp_param.reduce_partial_output is not None:
                reduced_grad.add_(hsdp_param.reduce_partial_output)
                hsdp_param.reduce_partial_output = None

            if hsdp_param.unsharded_accumulated_grad_data is not None:
                hsdp_param.unsharded_accumulated_grad = None
            elif hsdp_param.unsharded_param.grad is not None:
                hsdp_param.unsharded_param.grad = None

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
        """Clear communication bookkeeping without clearing optimizer gradients."""
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
        """Set whether this state synchronizes gradients in backward."""
        self.reduce_grads = requires_grad_sync

    def set_reduce_op_type(self, reduce_op_type: str) -> None:
        """Set the reduction operation accepted by ``mindspore.mint.distributed``."""
        reduce_op = reduce_op_type.lower().strip()
        if reduce_op not in ("sum", "avg"):
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, supported types are ['sum', 'avg']"
            )
        self.reduce_op_type = reduce_op

    @staticmethod
    def _sync_current_stream_if_needed(need_synchronize: bool) -> None:
        """Synchronize after a non-blocking CPU-offload copy when required."""
        if need_synchronize:
            ms.runtime.current_stream().synchronize()
