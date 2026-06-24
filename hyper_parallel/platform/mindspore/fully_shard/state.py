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
"""MindSpore HSDP cell state"""
from collections import defaultdict
from typing import List, Optional
import mindspore as ms
from mindspore import ops
import mindspore.mint.distributed as dist
from hyper_parallel.tools.logging import get_logger
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import (
    _get_param_module_infos,
    FullyShardParamMode,
    infer_fully_shard_param_mode,
    apply_gradient_scaling_factor,
)
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import build_rs_plan
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.param_group import (
    AllReduceParamGroup,
    HSDPParamGroup,
    get_comm_ctx,
)
from hyper_parallel.platform.mindspore.utils import normalize_runtime_device
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy

logger = get_logger("FSDP")


def _to_dtype_if_needed(
    tensor: ms.Tensor, dtype: Optional[ms.Type]
) -> ms.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype.

    Args:
        tensor: The input tensor to potentially cast.
        dtype: Target dtype. If None or same as tensor dtype, no-op.
    """
    if isinstance(dtype, ms.Type) and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


class MindSporeHSDPStateV2(HSDPState):
    """MindSpore HSDP cell state"""
    # DTensor compat parameters in pure-TP mode can accumulate gradients
    # directly on ``sharded_param.grad`` without materializing an
    # ``_unsharded_param``. Track those async all-reduces separately from the
    # standard unsharded-gradient queues.
    pre_direct_all_reduce_grads = []
    # Reserved for HSDP fused all-reduce pipeline (phase-2); kept for API parity with Torch.
    pre_all_reduce_groups: List = []
    pending_all_reduce_groups: List = []

    @staticmethod
    def _get_pending_unsharded_grad(hsdp_param):
        """Return the pending unsharded gradient tensor for reduction paths."""
        if hsdp_param.unsharded_accumulated_grad is not None:
            return hsdp_param.unsharded_accumulated_grad_data
        return hsdp_param.unsharded_grad_data

    @staticmethod
    def _has_pending_unsharded_grad(hsdp_param):
        """Whether the parameter currently has a gradient waiting for reduction."""
        if hsdp_param.unsharded_accumulated_grad is not None:
            return True
        if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
            return False
        return hsdp_param.unsharded_param.grad is not None

    @staticmethod
    def _get_local_sharded_grad(hsdp_param):
        """Return the local gradient tensor currently stored on ``sharded_param``."""
        grad = hsdp_param.sharded_param.grad
        if grad is None:
            return None
        to_local = getattr(grad, "to_local", None)
        if callable(to_local):
            return to_local()
        return grad

    @staticmethod
    def _synchronize_current_stream_if_needed(need_synchronize: bool) -> None:
        """Synchronize the current device stream after non-blocking CPU offload."""
        if not need_synchronize:
            return
        ms.runtime.current_stream().synchronize()

    def _apply_pending_unsharded_grad_locally(self, hsdp_param) -> bool:
        """Materialize pending unsharded grad onto ``sharded_param.grad`` without communication."""
        pending_grad = self._get_pending_unsharded_grad(hsdp_param)
        apply_gradient_scaling_factor(
            pending_grad, hsdp_param.gradient_scaling_factor
        )
        return hsdp_param.apply_reduced_grad(pending_grad, self._orig_dtype)

    def __init__(self, cell, mesh_info, config, platform, device=None):
        super().__init__(cell, mesh_info, config, platform, device)
        self.comm_fusion = config.comm_fusion
        # Do ReduceScatter/AllReduce for grad
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        # Default reduce op is decided at the fully_shard-state level:
        # if any managed parameter is DTensor-backed, use SUM; otherwise AVG.
        self.reduce_op_type = self._resolve_default_reduce_op()
        self._reset_sharded_params = False
        self._init_param_group()

    def _iter_managed_params(self):
        """Return all fully_shard-managed parameters, including replicate_params."""
        return [*self.hsdp_params, *self.replicate_params]

    def _resolve_default_reduce_op(self):
        """Resolve the default reduce op for the whole fully_shard state."""
        for hsdp_param in self._iter_managed_params():
            if hsdp_param.param_mode in (
                FullyShardParamMode.DTENSOR_COMPAT,
                FullyShardParamMode.DTENSOR_UNIFIED,
            ):
                return ops.ReduceOp.SUM
        return ops.ReduceOp.AVG

    def _resolve_reduce_op(self):
        """Resolve the gradient reduction op for the current fully_shard state."""
        return self.reduce_op_type

    @staticmethod
    def _comm_fusion_unsupported_reason(hsdp_param) -> Optional[str]:
        """Return the reason why ``hsdp_param`` cannot participate in comm_fusion."""
        if not hsdp_param.enable_fsdp_shard:
            return "non-sharded parameters such as replicate_params are not supported"
        if hsdp_param.param_mode not in (
            FullyShardParamMode.LOCAL_PARAM,
            FullyShardParamMode.DTENSOR_UNIFIED,
        ):
            return f"param_mode {hsdp_param.param_mode} is not supported"
        local_shard = getattr(hsdp_param, "_sharded_local_tensor", None)
        if local_shard is None:
            return "missing local shard tensor for comm_fusion plan validation"
        plan_world_size = getattr(hsdp_param, "shard_world_size", None)
        if plan_world_size is None:
            plan_world_size = getattr(hsdp_param, "shard_size", 1)
        try:
            build_rs_plan(hsdp_param, local_shard, plan_world_size)
        except NotImplementedError as exc:
            return str(exc)
        except (AssertionError, ValueError) as exc:
            return f"cannot build comm_fusion pack plan: {exc}"
        return None

    def _init_param_group(self):
        """Initialize fused parameter group when comm_fusion is enabled."""
        if self.config.comm_fusion:
            unsupported_param = next(
                (
                    hsdp_param
                    for hsdp_param in self.hsdp_params
                    if self._comm_fusion_unsupported_reason(hsdp_param) is not None
                ),
                None,
            )
            if unsupported_param is not None:
                param_fqn = getattr(unsupported_param, "_param_fqn", "<unknown>")
                reason = self._comm_fusion_unsupported_reason(unsupported_param)
                raise NotImplementedError(
                    f"comm_fusion does not support parameter {param_fqn}: {reason}."
                )
            self.param_group = None
            if self.hsdp_params:
                self.param_group = HSDPParamGroup(
                    self.hsdp_params,
                    self.mesh_info,
                    self.device,
                    self.mp_policy,
                    self.config.comm_fusion_zero_copy,
                )

    def zero_grad(self):
        """zero grad"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.zero_grad()
        for hsdp_param in self.replicate_params:
            hsdp_param.zero_grad()

    def _move_states_to_device(self):
        """move states to device"""
        for mod in self.modules:
            for param in mod.get_parameters():
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                param_device = normalize_runtime_device(param.device)
                if param_device in (self.device, "meta"):
                    continue
                param.data = param.to(self.device)
            for buffer in mod.buffers():
                if buffer.device in (self.device, "meta"):
                    continue
                buffer.data = buffer.to(self.device)

    def _init_hsdp_params(self):
        """init hsdp parameters for cell and replicate parameters for cell."""
        # all parameters in the module tree(s), deduplicated
        visited_params = set()
        replicate_params = set(self.config.replicate_params or ())
        ignored_params = set(self.config.ignored_params or ())
        filtered_params = []
        for mod in self.modules:
            for _, param in mod.parameters_and_names():
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param in ignored_params:
                    continue
                if param in visited_params:
                    continue
                visited_params.add(param)
                filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, tuple(self.modules))
        for param, module_info in zip(filtered_params, module_infos):
            param_mode = infer_fully_shard_param_mode(self.config.mesh, [param])
            enable_fsdp_shard = param not in replicate_params
            hsdp_param = MindSporeHSDPParamV2(
                param,
                module_info,
                self.mesh_info,
                shard_placement_fn=self.config.shard_placement_fn,
                mp_policy=self.mp_policy,
                offload_policy=self.offload_policy,
                device=self.device,
                param_mode=param_mode,
                enable_fsdp_shard=enable_fsdp_shard,
            )
            if param in replicate_params:
                self.replicate_params.append(hsdp_param)
            else:
                self.hsdp_params.append(hsdp_param)
                self.sharded_hsdp_params.append(hsdp_param)

    def _init_mp_dtypes(self):
        """init mp dtypes for hsdp parameters and replicate parameters"""
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
        for replicate_param in self.replicate_params:
            replicate_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[MindSporeHSDPParamV2] = [
            p for p in self._iter_managed_params() if p.sharded_param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in trainable_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self._orig_dtype = next(iter(orig_dtypes)) if trainable_params else None
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self._reduce_dtype = next(iter(reduce_dtypes)) if trainable_params else None

    def lazy_init(self):
        """Refresh parameter views and validate runtime state before first execution."""
        if self.is_shard and not self._reset_sharded_params:
            for hsdp_param in self.hsdp_params:
                if hsdp_param.is_sharded:
                    hsdp_param.reset_sharded_param()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _validate_cpu_offload_params(self):
        """Validate that all parameters are on CPU when CPU offload policy is enabled."""
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        hsdp_params_not_on_cpu = [
            hsdp_param
            for hsdp_param in self._iter_managed_params()
            if not str(hsdp_param.sharded_param.device).lower().startswith("cpu")
        ]
        if hsdp_params_not_on_cpu:
            raise RuntimeError(
                "HSDP parameters should be materialized on CPU when enabling CPU offloading. "
                "For example, load a CPU state dict before training. "
                "Found following parameters on non-CPU device: "
                f"{[(p._param_fqn, p.sharded_param.device) for p in hsdp_params_not_on_cpu]}\n"
            )

    def _validate_no_meta_params(self):
        """Validate that all parameters have been materialized from meta device."""
        param_names_on_meta = [
            hsdp_param._param_fqn
            for hsdp_param in self._iter_managed_params()
            if hsdp_param.sharded_param.device == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "HSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, initialize the module weights on a real device before running training."
            )

    def _queue_replicate_params_allreduce(self) -> None:
        """Queue async all-reduce for config.replicate_params (aligned with Torch)."""
        for hsdp_param in self.replicate_params:
            if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                continue
            if not hsdp_param.sharded_param.requires_grad:
                continue
            if not self._has_pending_unsharded_grad(hsdp_param):
                continue
            if self._should_run_all_reduce(hsdp_param):
                self._queue_compat_all_reduce(hsdp_param)
            else:
                need_synchronize = self._apply_pending_unsharded_grad_locally(hsdp_param)
                self._synchronize_current_stream_if_needed(need_synchronize)

    def _drain_reduce_scatter_params(self) -> bool:
        """Wait pending reduce-scatter ops and apply sharded grads."""
        need_synchronize = False
        while HSDPState.pre_reduce_scatter_params:
            hsdp_param, pre_orig_dtype = HSDPState.pre_reduce_scatter_params.pop(0)
            logger.debug(
                "post_backward module=%s wait=reduce_scatter param=%s",
                self,
                hsdp_param,
            )
            reduced_grad = hsdp_param.reduce_scatter_output()
            hsdp_param.clear_reduce_scatter_output()
            need_synchronize = (
                hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype)
                or need_synchronize
            )
            hsdp_param.accumulated_allreduced_grad = False
        return need_synchronize

    def reduce_scattered_params(self):
        """Wait pending reduce-scatter ops and apply sharded grads (FSDP pipeline step 2)."""
        need_synchronize = self._drain_reduce_scatter_params()
        self._synchronize_current_stream_if_needed(need_synchronize)

    def reduce_params(self):
        """Apply reduced gradients from pre-staged all-reduce queues (aligned with Torch).

        Drains ``pre_all_reduce_params`` and ``pre_direct_all_reduce_grads``. For
        pending reduce-scatter work, call ``reduce_scattered_params()`` separately.
        """
        need_synchronize = False
        while HSDPState.pre_all_reduce_params:
            hsdp_param, pre_orig_dtype = HSDPState.pre_all_reduce_params.pop(0)
            logger.debug(
                "post_backward module=%s wait=all_reduce param=%s",
                self,
                hsdp_param,
            )
            reduced_grad = hsdp_param.all_reduce_output()
            hsdp_param.clear_all_reduce_output()
            need_synchronize = (
                hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype)
                or need_synchronize
            )
        while MindSporeHSDPStateV2.pre_direct_all_reduce_grads:
            hsdp_param, handle, reduced_grad, target_grad, *_ = (
                MindSporeHSDPStateV2.pre_direct_all_reduce_grads.pop(0)
            )
            if handle is not None:
                logger.debug("post_backward module=%s wait=direct_compat_all_reduce", self)
                handle.wait()
            # all-reduce already applied SUM/AVG via _resolve_reduce_op(); skip legacy manual AVG div.
            if hsdp_param.mp_policy.apply_grad_on_fp32_main_grad:
                need_synchronize = (
                    hsdp_param.apply_reduced_grad(reduced_grad, self._orig_dtype)
                    or need_synchronize
                )
            elif reduced_grad is not target_grad:
                if reduced_grad.dtype != target_grad.dtype:
                    reduced_grad = reduced_grad.to(target_grad.dtype)
                copy_without_bumping_version(target_grad, reduced_grad)
        self._synchronize_current_stream_if_needed(need_synchronize)

    def _wait_prev_reduce_scatter(self) -> List:
        """Step 1: wait previous module RS for HSDP fused all-reduce groups."""
        if MindSporeHSDPStateV2.pre_all_reduce_groups:
            prev_groups = list(MindSporeHSDPStateV2.pre_all_reduce_groups)
            MindSporeHSDPStateV2.pre_all_reduce_groups.clear()
            for prev_group in prev_groups:
                for hsdp_param in prev_group.hsdp_params:
                    hsdp_param.reduce_scatter_output()
                    hsdp_param.clear_reduce_scatter_output()
                    if hsdp_param.unsharded_accumulated_grad_data is not None:
                        hsdp_param.unsharded_accumulated_grad = None
                    elif hsdp_param.unsharded_param.grad is not None:
                        hsdp_param.unsharded_param.grad = None
            return prev_groups
        return []

    def _wait_and_apply_prev_no_allreduce_params(self):
        """Step 2: wait/apply previous reduce-scatter for pure FSDP params."""
        self.reduce_scattered_params()

    def _should_skip_reduce_scatter_issue(self, hsdp_param) -> bool:
        """Return True when a parameter should not enter the HSDP RS/fused-AR pipeline."""
        return (
            not hasattr(hsdp_param, "_unsharded_param")
            or hsdp_param.unsharded_param is None
            or not hasattr(hsdp_param, "sharded_param")
            or not hsdp_param.sharded_param.requires_grad
            or hsdp_param.shard_size <= 1
            or self._can_direct_all_reduce_compat_grad(hsdp_param)
            or not self._has_pending_unsharded_grad(hsdp_param)
        )

    def _collect_params_for_reduce_scatter(self):
        """Collect parameters that need the HSDP RS/fused-AR overlap pipeline."""
        return [
            hsdp_param
            for hsdp_param in self._iter_managed_params()
            if not self._should_skip_reduce_scatter_issue(hsdp_param)
        ]

    def _needs_overlap_post_backward_steps(self) -> bool:
        """Whether the 4-step RS/AR overlap pipeline has pending work this hook."""
        if MindSporeHSDPStateV2.pre_all_reduce_groups:
            return True
        if HSDPState.pre_reduce_scatter_params:
            return True
        return bool(self._collect_params_for_reduce_scatter())

    def _run_overlap_post_backward_steps(self) -> None:
        """Run the 4-step HSDP RS/AR overlap pipeline for the current module."""
        prev_group = self._wait_prev_reduce_scatter()
        self._wait_and_apply_prev_no_allreduce_params()
        self._issue_reduce_scatter_for_current_module()
        self._issue_prev_fused_allreduce(prev_group)

    def _issue_reduce_scatter_for_current_module(self):
        """Issue reduce_scatter for current module with fused all-reduce when needed."""
        params_to_reduce = self._collect_params_for_reduce_scatter()
        if not params_to_reduce:
            return

        groups_by_comm = defaultdict(list)
        for hsdp_param in params_to_reduce:
            if self._should_run_all_reduce(hsdp_param):
                replicate_group = hsdp_param.unsharded_group_info.group
                key = id(replicate_group) if replicate_group is not None else None
                groups_by_comm[key].append(hsdp_param)
            else:
                groups_by_comm[None].append(hsdp_param)

        if None in groups_by_comm:
            for hsdp_param in groups_by_comm[None]:
                hsdp_param.reduce_scatter_grad(
                    async_op=True,
                    dtype=self._reduce_dtype,
                    reduce_op=self._resolve_reduce_op(),
                )
                HSDPState.pre_reduce_scatter_params.append(
                    (hsdp_param, self._orig_dtype)
                )

        for key, hsdp_params in groups_by_comm.items():
            if key is None:
                continue
            group_info = hsdp_params[0].unsharded_group_info
            group = AllReduceParamGroup(
                replicate_group=group_info.group,
                hsdp_params=hsdp_params,
                orig_dtypes=[self._orig_dtype] * len(hsdp_params),
                reduce_dtype=self._reduce_dtype,
                reduce_op=self._resolve_reduce_op(),
                mp_policy=self.mp_policy,
                replicate_world_size=group_info.rank_size,
            )
            group.allocate_fused_buffer(self.device)
            for idx, hsdp_param in enumerate(hsdp_params):
                buffer_view = group.get_param_buffer_view(idx)
                hsdp_param.reduce_scatter_grad(
                    async_op=True,
                    dtype=self._reduce_dtype,
                    reduce_op=self._resolve_reduce_op(),
                    output_buffer=buffer_view,
                )
            MindSporeHSDPStateV2.pre_all_reduce_groups.append(group)

    def _issue_prev_fused_allreduce(self, prev_groups: List) -> None:
        """Step 4: issue async all-reduce for previous HSDP groups (no-op without fusion groups)."""
        for prev_group in prev_groups:
            prev_group.accumulate_existing_grads_to_buffer()
            prev_group.issue_async_allreduce()
            MindSporeHSDPStateV2.pending_all_reduce_groups.append(prev_group)

    @classmethod
    def delay_apply_reduce_grads(cls) -> None:
        """Wait pending fused all-reduce groups at root backward."""
        need_synchronize = False
        for group in cls.pending_all_reduce_groups:
            need_synchronize = group.wait_and_apply_grads() or need_synchronize
        cls.pending_all_reduce_groups.clear()
        if need_synchronize:
            ms.runtime.current_stream().synchronize()

    def post_backward_for_comm_fusion(self):
        """Drive the fused gradient-reduction pipeline for sharded params."""
        logger.debug("post_backward module=%s mode=comm_fusion enter", self)
        self.reduce_params()
        comm_ctx = get_comm_ctx()
        if comm_ctx.all_reduce_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_all_reduce", self)
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
            comm_ctx.all_reduce_param_group = None
        if comm_ctx.pre_param_group is not None:
            logger.debug("post_backward module=%s wait=comm_fusion_reduce_scatter", self)
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if self.param_group is not None:
            logger.debug("post_backward module=%s launch=comm_fusion_reduce_scatter", self)
            self.param_group.foreach_reduce(
                reduce_scatter_reduce_op=self._resolve_reduce_op(),
            )
        self._queue_replicate_params_allreduce()

    def _post_backward_without_reduce(self):
        """Finish backward when gradient communication is disabled."""
        if self.reshard_after_backward:
            self.shard()
        for hsdp_param in self._iter_managed_params():
            hsdp_param.to_accumulated_grad_if_needed()

    def _should_run_all_reduce(self, hsdp_param) -> bool:
        """Whether the current parameter should issue an all-reduce in this backward pass."""
        return self.requires_all_reduce and hsdp_param.dp_size > 1

    def _queue_compat_all_reduce(self, hsdp_param):
        """Queue the compatibility all-reduce path without FSDP sharding."""
        if not self._should_run_all_reduce(hsdp_param):
            return
        # Pure all-reduce path: pass grad=None so all_reduce_grad fetches the
        # unsharded grad itself and owns the scaling (no reduce-scatter here).
        hsdp_param.all_reduce_grad(
            dtype=self._reduce_dtype,
            async_op=True,
            reduce_op=self._resolve_reduce_op(),
        )
        logger.debug(
            "post_backward module=%s launch=compat_all_reduce param=%s",
            self,
            hsdp_param,
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, self._orig_dtype))

    def _can_direct_all_reduce_compat_grad(self, hsdp_param) -> bool:
        """Whether ``hsdp_param`` should reduce its existing ``sharded_param.grad`` directly."""
        if not hasattr(hsdp_param, "param_mode"):
            return False
        return (
            hsdp_param.param_mode == FullyShardParamMode.DTENSOR_COMPAT
            and hsdp_param.enable_fsdp_shard
            and not hsdp_param.is_sharded
            and hsdp_param.shard_size == 1
            and hsdp_param.sharded_param.requires_grad
            and self._should_run_all_reduce(hsdp_param)
            and self._get_local_sharded_grad(hsdp_param) is not None
        )

    def _queue_direct_compat_all_reduce(self, hsdp_param):
        """Queue all-reduce for DTENSOR_COMPAT params whose grad stays on ``sharded_param``."""
        grad = self._get_local_sharded_grad(hsdp_param)
        if grad is None:
            return
        reduced_grad = _to_dtype_if_needed(grad, self._reduce_dtype)
        # All-reduce needs a contiguous buffer; the local sharded grad may be a
        # non-contiguous view. No-op when already contiguous; the copy is written
        # back to grad in reduce_params().
        reduced_grad = reduced_grad.contiguous()
        # Pure all-reduce path (no reduce-scatter): this leg owns the scaling.
        # all-reduce below is in-place, so scale in-place before it.
        apply_gradient_scaling_factor(reduced_grad, hsdp_param.gradient_scaling_factor)
        reduce_group_info = getattr(hsdp_param, "unsharded_group_info", None)
        reduce_group = reduce_group_info.group if reduce_group_info is not None else None
        reduce_group_size = reduce_group_info.rank_size if reduce_group_info is not None else 1
        handle = None
        if reduce_group_size > 1:
            if reduce_group is None:
                raise RuntimeError("Expected a valid unsharded all-reduce group when rank_size > 1")
            handle = dist.all_reduce(
                reduced_grad,
                group=reduce_group,
                op=self._resolve_reduce_op(),
                async_op=True,
            )
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads.append(
            (hsdp_param, handle, reduced_grad, grad, reduce_group_size, False)
        )

    def post_backward(self, *_):
        """Post-backward hook that accumulates, reduces, and reshards gradients for all managed parameters."""
        for hsdp_param in self._iter_managed_params():
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            self._post_backward_without_reduce()
            return
        if not self.comm_fusion:
            self.reduce_params()
            for hsdp_param in self._iter_managed_params():
                # replicate_params are queued once by _queue_replicate_params_allreduce().
                if not getattr(hsdp_param, "enable_fsdp_shard", True):
                    continue
                if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                    if self._can_direct_all_reduce_compat_grad(hsdp_param):
                        self._queue_direct_compat_all_reduce(hsdp_param)
                    continue
                if not hasattr(hsdp_param, "sharded_param") or not hsdp_param.sharded_param.requires_grad:
                    continue
                if not self._has_pending_unsharded_grad(hsdp_param):
                    continue
                if hsdp_param.shard_size <= 1:
                    if self._should_run_all_reduce(hsdp_param):
                        self._queue_compat_all_reduce(hsdp_param)
                    else:
                        logger.debug(
                            "post_backward module=%s apply=no_comm_grad param=%s",
                            self,
                            hsdp_param,
                        )
                        # No-communication path (shard_size == 1, no all-reduce):
                        # this leg owns the scaling since the grad never goes through
                        # reduce_scatter_grad / all_reduce_grad.
                        need_synchronize = self._apply_pending_unsharded_grad_locally(
                            hsdp_param
                        )
                        self._synchronize_current_stream_if_needed(need_synchronize)

            if self._needs_overlap_post_backward_steps():
                self._run_overlap_post_backward_steps()
            self._queue_replicate_params_allreduce()
        else:
            self.post_backward_for_comm_fusion()
        if self.reshard_after_backward:
            self.shard()

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.reduce_grads = requires_grad_sync

    def set_reduce_op_type(self, reduce_op_type: str):
        """set reduce op type for gradient reduction."""
        fsdp_support_reduce_op = {
            "sum": ops.ReduceOp.SUM,
            "avg": ops.ReduceOp.AVG,
        }
        if reduce_op_type not in fsdp_support_reduce_op:
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}")
        reduce_op: str = reduce_op_type.lower().strip()
        self.reduce_op_type = fsdp_support_reduce_op.get(reduce_op)
