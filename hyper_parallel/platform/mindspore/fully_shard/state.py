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
from typing import Optional
import mindspore as ms
from mindspore import ops
import mindspore.mint.distributed as dist
from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import (
    _get_param_module_infos,
    FullyShardParamMode,
    infer_fully_shard_param_mode,
)
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import build_rs_plan
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.param_group import HSDPParamGroup, get_comm_ctx
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
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


class MindSporeHSDPStateV2(HSDPState):
    """MindSpore HSDP cell state"""
    # DTensor compat parameters in pure-TP mode can accumulate gradients
    # directly on ``sharded_param.grad`` without materializing an
    # ``_unsharded_param``. Track those async all-reduces separately from the
    # standard unsharded-gradient queues.
    pre_direct_all_reduce_grads = []

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
        # Keep historical AVG behavior for local parameters while DTensor-aware
        # paths default to SUM semantics without extra division.
        self.reduce_op_type = ops.ReduceOp.SUM
        self._need_div = not any(
            getattr(param, "param_mode", FullyShardParamMode.LOCAL_PARAM)
            != FullyShardParamMode.LOCAL_PARAM
            for param in self._iter_managed_params()
        )
        self._ignored_allreduce_works = []
        self._reset_sharded_params = False
        self._init_param_group()

    def _iter_managed_params(self):
        """Return all fully_shard-managed parameters, including replicate_params."""
        return [*self.hsdp_params, *self.replicate_params]

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

    @staticmethod
    def _div_if_needed(x, divisor, need_div: bool):
        """Apply gradient averaging only when the caller-provided policy requires it.

        ``need_div`` may come from the current state or from metadata captured when
        async reduce work was queued, so this helper is safe for both immediate and
        deferred gradient materialization paths.
        """
        if not need_div:
            return
        if divisor == 1:
            return
        x.div_(divisor)

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
                if hsdp_param.is_sharded:
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
        if not self._reset_sharded_params:
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

    def _allreduce_replicate_params(self, async_op=True) -> None:
        """
        DDP-style all-reduce for parameters in config.replicate_params.

        Use the parameter's layout-driven unsharded group so DTensor-aware
        compatibility and unified modes reduce over the correct axes.
        """
        for param in self.replicate_params:
            if not hasattr(param, "_unsharded_param") or param.unsharded_param is None:
                continue
            if (
                param.unsharded_accumulated_grad is None
                and param.unsharded_param.grad is None
            ):
                continue

            reduced_grad = param.unsharded_accumulated_grad_data
            if reduced_grad is None:
                reduced_grad = param.unsharded_grad_data
            reduced_grad = _to_dtype_if_needed(reduced_grad, self._reduce_dtype)
            reduce_group_info = getattr(param, "unsharded_group_info", None)
            reduce_group = reduce_group_info.group if reduce_group_info is not None else None
            reduce_group_size = reduce_group_info.rank_size if reduce_group_info is not None else 1

            if reduce_group is not None and reduce_group_size > 1:
                # Ascend HCCL DistCommAllReduce rejects non-contiguous tensors;
                # reduced_grad here may still be a view from the no-reduce path
                # of ``unsharded_grad_data`` / ``_to_local_unsharded_grad``.
                # ``Tensor.contiguous()`` is a no-op when storage is already
                # contiguous, so the unconditional call is safe.
                reduced_grad = reduced_grad.contiguous()
                param.all_reduce_handle = dist.all_reduce(
                    reduced_grad, group=reduce_group, op=self.reduce_op_type, async_op=async_op
                )
            self._ignored_allreduce_works.append((param, reduced_grad, reduce_group_size))

    def _finish_ignored_allreduce(self) -> None:
        """
        Wait for async all-reduce of replicate_params and materialize param.grad.

        For each pending work, this:
        Waits on all associated handles to complete;
        Casts reduced_grad back to _orig_dtype if needed;
        Assigns the final tensor to param.grad.
        """
        if not self._ignored_allreduce_works:
            return

        need_synchronize = False
        for param, reduced_grad, reduce_group_size in self._ignored_allreduce_works:
            if param.all_reduce_handle:
                param.all_reduce_handle.wait()
            self._div_if_needed(reduced_grad, reduce_group_size, self._need_div)
            need_synchronize = (
                param.apply_reduced_grad(reduced_grad, self._orig_dtype)
                or need_synchronize
            )

        self._synchronize_current_stream_if_needed(need_synchronize)
        self._ignored_allreduce_works.clear()

    def reduce_params(self):
        """Drain pending sharded parameter reductions and materialize sharded grads."""
        need_synchronize = False
        while HSDPState.pre_reduce_scatter_params:
            hsdp_param, pre_orig_dtype, need_div = HSDPState.pre_reduce_scatter_params.pop(0)
            reduced_grad = hsdp_param.reduce_scatter_output()
            self._div_if_needed(reduced_grad, hsdp_param.shard_world_size, need_div)
            hsdp_param.clear_reduce_scatter_output()
            need_synchronize = (
                hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype)
                or need_synchronize
            )

        while HSDPState.pre_all_reduce_params:
            hsdp_param, pre_orig_dtype, need_div = HSDPState.pre_all_reduce_params.pop(0)
            reduced_grad = hsdp_param.all_reduce_output()
            self._div_if_needed(reduced_grad, hsdp_param.replicate_world_size, need_div)
            hsdp_param.clear_all_reduce_output()
            need_synchronize = (
                hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype)
                or need_synchronize
            )
        while MindSporeHSDPStateV2.pre_direct_all_reduce_grads:
            handle, reduced_grad, target_grad, reduce_group_size, need_div = (
                MindSporeHSDPStateV2.pre_direct_all_reduce_grads.pop(0)
            )
            if handle is not None:
                handle.wait()
            self._div_if_needed(reduced_grad, reduce_group_size, need_div)
            if reduced_grad is not target_grad:
                if reduced_grad.dtype != target_grad.dtype:
                    reduced_grad = reduced_grad.to(target_grad.dtype)
                copy_without_bumping_version(target_grad, reduced_grad)
        self._synchronize_current_stream_if_needed(need_synchronize)

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
            self.param_group.foreach_reduce(
                reduce_scatter_reduce_op=self.reduce_op_type,
                needs_avg_div=self._need_div,
            )
        self._allreduce_replicate_params()

    def _post_backward_without_reduce(self):
        """Finish backward when gradient communication is disabled."""
        if self.reshard_after_backward:
            self.shard()
        for hsdp_param in self._iter_managed_params():
            hsdp_param.to_accumulated_grad_if_needed()

    def _should_run_all_reduce(self, hsdp_param) -> bool:
        """Whether the current parameter should issue an all-reduce in this backward pass."""
        return self.requires_all_reduce and hsdp_param.dp_size > 1

    def _queue_reduce_scatter_then_all_reduce(self, hsdp_param):
        """Queue the standard FSDP/HSDP reduction path."""
        hsdp_param.reduce_scatter_grad(
            async_op=True,
            dtype=self._reduce_dtype,
            reduce_op=self.reduce_op_type
        )
        HSDPState.pre_reduce_scatter_params.append((hsdp_param, self._orig_dtype, self._need_div))
        if not self._should_run_all_reduce(hsdp_param):
            return
        reduced_grad = hsdp_param.reduce_scatter_output()
        if (
            HSDPState.pre_reduce_scatter_params
            and HSDPState.pre_reduce_scatter_params[-1][0] == hsdp_param
        ):
            HSDPState.pre_reduce_scatter_params.pop()
        hsdp_param.clear_reduce_scatter_output()
        self._div_if_needed(reduced_grad, hsdp_param.shard_size, self._need_div)
        hsdp_param.all_reduce_grad(
            grad=reduced_grad,
            dtype=self._reduce_dtype,
            async_op=True,
            reduce_op=self.reduce_op_type,
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, self._orig_dtype, self._need_div))

    def _queue_compat_all_reduce(self, hsdp_param):
        """Queue the compatibility all-reduce path without FSDP sharding."""
        if not self._should_run_all_reduce(hsdp_param):
            return
        hsdp_param.all_reduce_grad(
            grad=self._get_pending_unsharded_grad(hsdp_param),
            dtype=self._reduce_dtype,
            async_op=True,
            reduce_op=self.reduce_op_type,
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, self._orig_dtype, self._need_div))

    def _can_direct_all_reduce_compat_grad(self, hsdp_param) -> bool:
        """Whether ``hsdp_param`` should reduce its existing ``sharded_param.grad`` directly."""
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
                op=self.reduce_op_type,
                async_op=True,
            )
        MindSporeHSDPStateV2.pre_direct_all_reduce_grads.append(
            (handle, reduced_grad, grad, reduce_group_size, self._need_div)
        )

    def post_backward(self, *_):
        for hsdp_param in self._iter_managed_params():
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            self._post_backward_without_reduce()
            return
        if not self.comm_fusion:
            self.reduce_params()
            self._allreduce_replicate_params()
            for hsdp_param in self.hsdp_params:
                if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                    if self._can_direct_all_reduce_compat_grad(hsdp_param):
                        self._queue_direct_compat_all_reduce(hsdp_param)
                    continue
                if not hsdp_param.sharded_param.requires_grad:
                    continue
                if not self._has_pending_unsharded_grad(hsdp_param):
                    continue
                if hsdp_param.shard_size > 1:
                    self._queue_reduce_scatter_then_all_reduce(hsdp_param)
                elif self._should_run_all_reduce(hsdp_param):
                    self._queue_compat_all_reduce(hsdp_param)
                else:
                    need_synchronize = hsdp_param.apply_reduced_grad(
                        self._get_pending_unsharded_grad(hsdp_param),
                        self._orig_dtype,
                    )
                    self._synchronize_current_stream_if_needed(need_synchronize)
            self._finish_ignored_allreduce()
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
            "avg": ops.ReduceOp.SUM,
        }
        if reduce_op_type not in fsdp_support_reduce_op:
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}")
        self._need_div = reduce_op_type == "avg"
        reduce_op: str = reduce_op_type.lower().strip()
        self.reduce_op_type = fsdp_support_reduce_op[reduce_op]
