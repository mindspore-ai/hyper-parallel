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
from functools import partial
from typing import List, Optional
import mindspore as ms
from mindspore import ops
import mindspore.mint.distributed as dist
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
    # Feature-only fused reduce-scatter groups. Each entry owns its handle and
    # packed input, so several states can remain pending across pipeline work.
    pending_sharded_grad_param_groups: List[HSDPParamGroup] = []

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
        need_synchronize = hsdp_param.apply_reduced_grad(
            pending_grad, hsdp_param.orig_dtype
        )
        if self._is_sharded_grad_accumulation_active():
            hsdp_param.accumulated_allreduced_grad = False
        return need_synchronize

    def __init__(self, cell, mesh_info, config, platform, device=None):
        super().__init__(cell, mesh_info, config, platform, device)
        self.comm_fusion = config.comm_fusion
        self.sharded_accumulated_grad = getattr(config, "sharded_accumulated_grad", False)
        self.sharded_grad_ready_overlap = getattr(
            config, "sharded_grad_ready_overlap", False
        )
        self.sharded_accumulated_grad_max_pending = getattr(
            config, "sharded_accumulated_grad_max_pending", 1
        )
        self.sharded_grad_reduce_dtype = getattr(
            config, "sharded_grad_reduce_dtype", None
        )
        self._pending_sharded_grad_all_reduce_groups = []
        self._param_group_grad_ready_expected = frozenset()
        self._param_group_grad_ready_params = set()
        self._param_group_grad_ready_reduced = False
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
        self._register_param_group_grad_ready_hooks()

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
                    self.sharded_grad_reduce_dtype,
                )

    def _register_param_group_grad_ready_hooks(self) -> None:
        """Register one internal gradient-ready hook per trainable managed parameter."""
        if (
            not self.sharded_accumulated_grad
            or not self.sharded_grad_ready_overlap
            or not self.comm_fusion
            or self.param_group is None
        ):
            return
        trainable_params = [
            hsdp_param
            for hsdp_param in self._iter_managed_params()
            if hsdp_param.sharded_param.requires_grad
        ]
        self._param_group_grad_ready_expected = frozenset(
            id(hsdp_param) for hsdp_param in trainable_params
        )
        for hsdp_param in trainable_params:
            grad_ready_hook = partial(
                self._param_group_grad_ready_hook,
                hsdp_param,
            )
            hsdp_param._register_internal_backward_hook(grad_ready_hook)

    def _param_group_grad_ready_hook(self, hsdp_param, grad):
        """Stage one final leaf gradient and reduce once the whole parameter group is ready."""
        if not self._is_sharded_grad_accumulation_active():
            return grad
        if id(hsdp_param) not in self._param_group_grad_ready_expected:
            return grad
        if self._param_group_grad_ready_reduced:
            raise RuntimeError(
                "A fully_shard parameter gradient became ready again after its "
                "parameter group had already launched reduce-scatter in the same "
                "micro-batch. This hook requires one final accumulated gradient per "
                "parameter."
            )
        if id(hsdp_param) in self._param_group_grad_ready_params:
            raise RuntimeError(
                "A fully_shard parameter gradient-ready hook ran more than once "
                "in the same micro-batch. The hook must receive one final "
                "accumulated gradient per parameter."
            )

        local_grad = hsdp_param._to_local_unsharded_grad(grad)
        stage_dtype = hsdp_param.reduce_dtype
        if stage_dtype is not None and local_grad.dtype != stage_dtype:
            local_grad = local_grad.to(stage_dtype)
        if hsdp_param.unsharded_accumulated_grad is None:
            hsdp_param.unsharded_accumulated_grad = local_grad
        else:
            hsdp_param.unsharded_accumulated_grad += local_grad
        self._param_group_grad_ready_params.add(id(hsdp_param))

        if self._param_group_grad_ready_params == self._param_group_grad_ready_expected:
            with self.platform.profiler_record(
                f"param_group_grad_ready:{self.module_name}"
            ):
                self._reduce_pending_grads()
            self._param_group_grad_ready_reduced = True
        return grad

    def _stage_leaf_grads_after_backward(self) -> None:
        """Discard hook-owned leaf grads and stage parameters that missed the ready hook."""
        for hsdp_param in self._iter_managed_params():
            hsdp_param.clear_released_unsharded_grad()
            if (
                not hasattr(hsdp_param, "_unsharded_param")
                or hsdp_param.unsharded_param is None
            ):
                continue
            if id(hsdp_param) in self._param_group_grad_ready_params:
                hsdp_param.unsharded_param.grad = None
            else:
                hsdp_param.to_accumulated_grad_if_needed()

    def _reset_param_group_grad_ready_state(self) -> None:
        """Reset per-micro-batch parameter-group readiness bookkeeping."""
        self._param_group_grad_ready_params.clear()
        self._param_group_grad_ready_reduced = False

    def zero_grad(self):
        """zero grad"""
        if self.sharded_accumulated_grad:
            self.wait_sharded_accumulated_grad_reduce_scatters()
            self.wait_sharded_accumulated_grad_all_reduces()
        for hsdp_param in self._iter_managed_params():
            if self.sharded_accumulated_grad:
                hsdp_param.clear_released_unsharded_grad()
                hsdp_param.unsharded_accumulated_grad = None
                if (
                    hasattr(hsdp_param, "_unsharded_param")
                    and hsdp_param.unsharded_param is not None
                ):
                    hsdp_param.unsharded_param.grad = None
            hsdp_param.zero_grad()
        self._reset_param_group_grad_ready_state()

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
        fused_trainable_params = []
        fused_orig_dtypes = set()
        fused_reduce_dtypes = set()
        fused_all_gather_dtypes = set()
        for hsdp_param in self.hsdp_params:
            hsdp_param.init_dtype_attrs(self.mp_policy)
            if not self.comm_fusion:
                continue
            all_gather_dtype = hsdp_param.orig_dtype
            if hsdp_param.param_dtype is not None:
                all_gather_dtype = hsdp_param.param_dtype
            fused_all_gather_dtypes.add(all_gather_dtype)
            if hsdp_param.sharded_param.requires_grad:
                fused_trainable_params.append(hsdp_param)
                fused_orig_dtypes.add(hsdp_param.orig_dtype)
                fused_reduce_dtypes.add(hsdp_param.reduce_dtype)
        for replicate_param in self.replicate_params:
            replicate_param.init_dtype_attrs(self.mp_policy)
        if not self.comm_fusion:
            return
        if len(fused_trainable_params) > 0 and len(fused_orig_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform original parameter dtype but got {fused_orig_dtypes}"
            )
        self._orig_dtype = next(iter(fused_orig_dtypes)) if fused_trainable_params else None
        if len(fused_trainable_params) > 0 and len(fused_reduce_dtypes) != 1:
            raise AssertionError(
                f"hsdp expects uniform reduce dtype but got {fused_reduce_dtypes}"
            )
        self._reduce_dtype = next(iter(fused_reduce_dtypes)) if fused_trainable_params else None
        if len(fused_all_gather_dtypes) > 1:
            raise AssertionError(
                "hsdp comm_fusion expects uniform all-gather parameter dtype "
                f"but got {fused_all_gather_dtypes}"
            )

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
                "Found following parameters on non-CPU device: "
                f"{[(p._param_fqn, p.sharded_param.device) for p in hsdp_params_not_on_cpu]}\n"
                "MindSpore backend will support this feature in future version."
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
            if self._is_sharded_grad_accumulation_active():
                need_synchronize = self._apply_pending_unsharded_grad_locally(hsdp_param)
                self._synchronize_current_stream_if_needed(need_synchronize)
            elif self._should_run_all_reduce(hsdp_param):
                self._queue_compat_all_reduce(hsdp_param)
            else:
                need_synchronize = self._apply_pending_unsharded_grad_locally(hsdp_param)
                self._synchronize_current_stream_if_needed(need_synchronize)

    def _drain_reduce_scatter_params(self) -> bool:
        """Wait pending reduce-scatter ops and apply sharded grads."""
        need_synchronize = False
        while HSDPState.pre_reduce_scatter_params:
            pending = HSDPState.pre_reduce_scatter_params.pop(0)
            hsdp_param, pre_orig_dtype = pending[:2]
            defer_all_reduce = len(pending) > 2 and pending[2]
            reduced_grad = hsdp_param.reduce_scatter_output()
            hsdp_param.clear_reduce_scatter_output()
            if defer_all_reduce:
                accumulation_dtype = pending[3] if len(pending) > 3 else None
                reduced_grad = _to_dtype_if_needed(
                    reduced_grad, accumulation_dtype
                )
                apply_need_synchronize = hsdp_param.apply_reduced_grad(
                    reduced_grad,
                    pre_orig_dtype,
                    clear_unsharded_grad=False,
                )
                hsdp_param.clear_released_unsharded_grad()
            else:
                apply_need_synchronize = hsdp_param.apply_reduced_grad(
                    reduced_grad, pre_orig_dtype
                )
            need_synchronize = apply_need_synchronize or need_synchronize
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
                handle.wait()
            # all-reduce already applied SUM/AVG via _resolve_reduce_op(); skip legacy manual AVG div.
            if hsdp_param.mp_policy.apply_grad_on_fp32_main_grad:
                need_synchronize = (
                    hsdp_param.apply_reduced_grad(reduced_grad, hsdp_param.orig_dtype)
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
        defer_all_reduce = self._is_sharded_grad_accumulation_active()
        for hsdp_param in params_to_reduce:
            if self._should_run_all_reduce(hsdp_param) and not defer_all_reduce:
                replicate_group = hsdp_param.unsharded_group_info.group
                key = id(replicate_group) if replicate_group is not None else None
                groups_by_comm[key].append(hsdp_param)
            else:
                groups_by_comm[None].append(hsdp_param)

        if None in groups_by_comm:
            for hsdp_param in groups_by_comm[None]:
                accumulation_dtype = (
                    hsdp_param.reduce_dtype
                    or self._get_pending_unsharded_grad(hsdp_param).dtype
                )
                reduce_dtype = accumulation_dtype
                if defer_all_reduce and self.sharded_grad_reduce_dtype is not None:
                    reduce_dtype = self.sharded_grad_reduce_dtype
                hsdp_param.reduce_scatter_grad(
                    async_op=True,
                    dtype=reduce_dtype,
                    reduce_op=self._resolve_reduce_op(),
                    release_unsharded_grad=defer_all_reduce,
                )
                pending_param = (hsdp_param, hsdp_param.orig_dtype)
                if defer_all_reduce:
                    pending_param += (True,)
                    if reduce_dtype != accumulation_dtype:
                        pending_param += (accumulation_dtype,)
                HSDPState.pre_reduce_scatter_params.append(pending_param)

        for key, hsdp_params in groups_by_comm.items():
            if key is None:
                continue
            group_info = hsdp_params[0].unsharded_group_info
            group = AllReduceParamGroup(
                replicate_group=group_info.group,
                hsdp_params=hsdp_params,
                orig_dtypes=[hsdp_param.orig_dtype for hsdp_param in hsdp_params],
                reduce_dtype=hsdp_params[0].reduce_dtype,
                reduce_op=self._resolve_reduce_op(),
                mp_policy=self.mp_policy,
                replicate_world_size=group_info.rank_size,
            )
            group.allocate_fused_buffer(self.device)
            for idx, hsdp_param in enumerate(hsdp_params):
                buffer_view = group.get_param_buffer_view(idx)
                hsdp_param.reduce_scatter_grad(
                    async_op=True,
                    dtype=hsdp_param.reduce_dtype,
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
        self.reduce_params()
        comm_ctx = get_comm_ctx()
        if comm_ctx.all_reduce_param_group is not None:
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
            comm_ctx.all_reduce_param_group = None
        if comm_ctx.pre_param_group is not None:
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if self.param_group is not None:
            defer_all_reduce = self._is_sharded_grad_accumulation_active()
            if defer_all_reduce:
                self._wait_until_sharded_grad_param_group_available(self.param_group)
            reduce_kwargs = {
                "reduce_scatter_reduce_op": self._resolve_reduce_op(),
            }
            if defer_all_reduce:
                reduce_kwargs["defer_all_reduce"] = True
            reduce_output = self.param_group.foreach_reduce(**reduce_kwargs)
            if defer_all_reduce and reduce_output is not None:
                MindSporeHSDPStateV2.pending_sharded_grad_param_groups.append(
                    self.param_group
                )
                self._drain_pending_sharded_grad_param_groups(
                    self.sharded_accumulated_grad_max_pending
                )
        self._queue_replicate_params_allreduce()

    @classmethod
    def _drain_pending_sharded_grad_param_groups(cls, max_pending: int = 0) -> None:
        """Wait oldest feature RS groups until at most ``max_pending`` remain."""
        while len(cls.pending_sharded_grad_param_groups) > max_pending:
            param_group = cls.pending_sharded_grad_param_groups[0]
            param_group.wait_deferred_reduce_scatter_and_apply_grad()
            cls.pending_sharded_grad_param_groups.pop(0)

    @classmethod
    def _wait_until_sharded_grad_param_group_available(
        cls, target_group: HSDPParamGroup
    ) -> None:
        """Drain FIFO work through ``target_group`` before reusing that object."""
        if not any(
            param_group is target_group
            for param_group in cls.pending_sharded_grad_param_groups
        ):
            return
        while cls.pending_sharded_grad_param_groups:
            param_group = cls.pending_sharded_grad_param_groups[0]
            param_group.wait_deferred_reduce_scatter_and_apply_grad()
            cls.pending_sharded_grad_param_groups.pop(0)
            if param_group is target_group:
                return

    def wait_sharded_accumulated_grad_reduce_scatters(self) -> None:
        """Wait and apply every feature-owned fused reduce-scatter."""
        if not self.sharded_accumulated_grad:
            return
        self._drain_pending_sharded_grad_param_groups()

    def _post_backward_without_reduce(self):
        """Finish backward when gradient communication is disabled."""
        if getattr(self, "sharded_accumulated_grad", False):
            # MindSpore may publish or replace a leaf ``.grad`` after this
            # callback. The pipeline-owned post-backward flush is the single
            # source of truth for staging and reducing this micro-batch.
            return
        if self.reshard_after_backward:
            self.shard()
        for hsdp_param in self._iter_managed_params():
            hsdp_param.to_accumulated_grad_if_needed()

    def _is_sharded_grad_accumulation_active(self) -> bool:
        """Whether this no-sync backward should reduce into local gradient shards."""
        return getattr(self, "sharded_accumulated_grad", False) and not self.reduce_grads

    def flush_sharded_accumulation_after_backward(self) -> None:
        """Stage and reduce one completed no-sync micro-batch.

        The parameter-ready path may already have launched this state's fused
        reduce-scatter. Pipeline calls this after the backward API returns to
        discard the duplicate native leaf ``.grad`` and to provide a fallback
        for unused parameters or backends that did not invoke every hook.
        """
        if not self._is_sharded_grad_accumulation_active():
            return
        self._stage_leaf_grads_after_backward()
        if (
            not self._param_group_grad_ready_reduced
            and any(
                self._has_pending_unsharded_grad(hsdp_param)
                for hsdp_param in self._iter_managed_params()
            )
        ):
            self._reduce_pending_grads()
            for hsdp_param in self._iter_managed_params():
                hsdp_param.clear_released_unsharded_grad()
        if self.reshard_after_backward and not self.is_shard:
            self.shard()
        self._reset_param_group_grad_ready_state()

    def _reduce_pending_grads(self) -> None:
        """Issue reductions for gradients produced by the current backward."""
        if self.comm_fusion:
            self.post_backward_for_comm_fusion()
            return

        defer_all_reduce = self._is_sharded_grad_accumulation_active()
        self.reduce_params()
        for hsdp_param in self._iter_managed_params():
            # replicate_params are handled once by _queue_replicate_params_allreduce().
            if not getattr(hsdp_param, "enable_fsdp_shard", True):
                continue
            if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                if not defer_all_reduce and self._can_direct_all_reduce_compat_grad(hsdp_param):
                    self._queue_direct_compat_all_reduce(hsdp_param)
                continue
            if not hasattr(hsdp_param, "sharded_param") or not hsdp_param.sharded_param.requires_grad:
                continue
            if not self._has_pending_unsharded_grad(hsdp_param):
                continue
            if hsdp_param.shard_size <= 1:
                if self._should_run_all_reduce(hsdp_param) and not defer_all_reduce:
                    self._queue_compat_all_reduce(hsdp_param)
                else:
                    need_synchronize = self._apply_pending_unsharded_grad_locally(hsdp_param)
                    self._synchronize_current_stream_if_needed(need_synchronize)

        if self._needs_overlap_post_backward_steps():
            self._run_overlap_post_backward_steps()
        self._queue_replicate_params_allreduce()

    @staticmethod
    def _get_local_accumulated_grad(hsdp_param):
        """Return the local tensor holding this step's accumulated gradient shard."""
        if hsdp_param.mp_policy.apply_grad_on_fp32_main_grad:
            grad = getattr(hsdp_param.sharded_param, "main_grad", None)
        else:
            grad = hsdp_param.sharded_param.grad
        if grad is None:
            return None
        local_tensor = getattr(grad, "_local_tensor", None)
        if local_tensor is not None:
            return local_tensor
        to_local = getattr(grad, "to_local", None)
        if callable(to_local):
            return to_local()
        return grad

    def launch_sharded_accumulated_grad_all_reduces(self) -> None:
        """Pack local gradient shards and launch one async all-reduce per bucket."""
        if not getattr(self, "sharded_accumulated_grad", False):
            return
        self.wait_sharded_accumulated_grad_reduce_scatters()
        if self._pending_sharded_grad_all_reduce_groups:
            raise RuntimeError(
                "Cannot launch sharded accumulated-gradient all-reduces while a previous "
                "step is still pending."
            )

        grouped_params = defaultdict(list)
        for hsdp_param in self._iter_managed_params():
            target_grad = self._get_local_accumulated_grad(hsdp_param)
            if target_grad is None:
                continue
            if not self._should_run_all_reduce(hsdp_param):
                hsdp_param.accumulated_allreduced_grad = True
                continue
            if hsdp_param.accumulated_allreduced_grad:
                continue
            group_info = hsdp_param.unsharded_group_info
            if group_info.group is None:
                raise RuntimeError(
                    f"Expected a valid replicate group for parameter {hsdp_param._param_fqn}."
                )
            reduce_dtype = hsdp_param.reduce_dtype or target_grad.dtype
            group_key = (
                group_info.group,
                str(reduce_dtype),
                group_info.rank_size,
            )
            grouped_params[group_key].append(hsdp_param)

        for hsdp_params in grouped_params.values():
            first_param = hsdp_params[0]
            group_info = first_param.unsharded_group_info
            first_grad = self._get_local_accumulated_grad(first_param)
            reduce_dtype = first_param.reduce_dtype or first_grad.dtype
            group = AllReduceParamGroup(
                replicate_group=group_info.group,
                hsdp_params=hsdp_params,
                orig_dtypes=[hsdp_param.orig_dtype for hsdp_param in hsdp_params],
                reduce_dtype=reduce_dtype,
                reduce_op=self._resolve_reduce_op(),
                mp_policy=self.mp_policy,
                replicate_world_size=group_info.rank_size,
            )
            group.pack_existing_grads_to_buffer()
            group.issue_async_allreduce()
            self._pending_sharded_grad_all_reduce_groups.append(group)

    def wait_sharded_accumulated_grad_all_reduces(self) -> None:
        """Wait for final all-reduce buckets and restore per-parameter gradients."""
        if not getattr(self, "sharded_accumulated_grad", False):
            return
        need_synchronize = False
        while self._pending_sharded_grad_all_reduce_groups:
            group = self._pending_sharded_grad_all_reduce_groups[0]
            group_need_synchronize = group.wait_and_apply_grads()
            self._pending_sharded_grad_all_reduce_groups.pop(0)
            need_synchronize = group_need_synchronize or need_synchronize
        self._synchronize_current_stream_if_needed(need_synchronize)

    def finalize_sharded_accumulated_grads(self) -> None:
        """Launch and wait for final bucketed HSDP replicate reductions."""
        self.launch_sharded_accumulated_grad_all_reduces()
        self.wait_sharded_accumulated_grad_all_reduces()

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
            dtype=hsdp_param.reduce_dtype,
            async_op=True,
            reduce_op=self._resolve_reduce_op(),
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, hsdp_param.orig_dtype))

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
        reduced_grad = _to_dtype_if_needed(grad, hsdp_param.reduce_dtype)
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
            if (
                self._is_sharded_grad_accumulation_active()
                and id(hsdp_param) in self._param_group_grad_ready_params
            ):
                continue
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            self._post_backward_without_reduce()
            return
        self._reduce_pending_grads()
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
        reduce_op: str = reduce_op_type.lower().strip()
        if reduce_op not in fsdp_support_reduce_op:
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}")
        self.reduce_op_type = fsdp_support_reduce_op[reduce_op]
