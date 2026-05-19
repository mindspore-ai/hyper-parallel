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

from typing import Optional, List
from collections import defaultdict
import torch

from hyper_parallel.core.fully_shard.hsdp_state import HSDPState
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    _get_param_module_infos,
    infer_fully_shard_param_mode,
)
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.pack_utils import build_rs_plan
from hyper_parallel.platform.torch.fully_shard.param_group import get_comm_ctx, HSDPParamGroup, AllReduceParamGroup


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
    # DTensor compat parameters in pure-TP mode can accumulate gradients
    # directly on ``sharded_param.grad`` without ever materializing an
    # ``_unsharded_param``. Track their async all-reduce work separately from
    # the standard unsharded-grad queues.
    pre_direct_all_reduce_grads = []
    # Record AllReduceParamGroup that has reduce_scatter issued, waiting for next post_backward to process
    pre_all_reduce_groups: List[AllReduceParamGroup] = []
    # Record AllReduceParamGroup that has all_reduce issued, waiting for root_backward_hook to apply
    pending_all_reduce_groups: List[AllReduceParamGroup] = []
    @staticmethod
    def _get_pending_unsharded_grad(hsdp_param):
        """Return the pending unsharded gradient tensor for all-reduce-based paths."""
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

    def __init__(self, cell, mesh_info, config, platform, device):
        """
        Initialize TorchHSDPStateV2.

        Args:
            cell (nn.Module): The module whose parameters are managed by this state.
            mesh_info: Mesh topology for shard/replicate dimensions.
            config (HSDPConfigV2): HSDP configuration.
            platform (TorchPlatform): Torch platform abstraction.
            device (torch.device): Target device.
        """
        super().__init__(cell, mesh_info, config, platform, device)
        self.comm_fusion = config.comm_fusion
        # Do ReduceScatter/AllReduce for grad
        self.device = device
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.reduce_grads = True
        # Reshard parameter after backward
        self.reshard_after_backward = True
        # Requires AllReduce for grad When HSDP
        self.requires_all_reduce = True
        # Default reduce op is decided at the fully_shard-state level:
        # if any managed parameter is DTensor-backed, use SUM; otherwise AVG.
        self._user_reduce_op_type = None
        self.reduce_op_type = self._resolve_default_reduce_op()
        self._reset_sharded_params = False
        self._init_param_group()

    @staticmethod
    def _comm_fusion_unsupported_reason(hsdp_param) -> Optional[str]:
        """Return the reason why ``hsdp_param`` cannot participate in comm_fusion."""
        if not hsdp_param.enable_fsdp_shard:
            return "non-sharded parameters such as replicate_params are not supported"
        if hsdp_param.param_mode not in (
            FullyShardParamMode.LOCAL_PARAM,
            FullyShardParamMode.DTENSOR_UNIFIED,
        ):
            return (
                "param_mode "
                f"{hsdp_param.param_mode} is not supported"
            )
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
        """Initialize fused parameter group for communication fusion.

        When ``comm_fusion`` is enabled, creates an ``HSDPParamGroup`` that packs all
        parameters into a single buffer for fused all-gather and reduce-scatter,
        replacing the per-parameter communication pattern.
        """
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
                # pylint: disable=E1128
                self.param_group = HSDPParamGroup(
                    self.hsdp_params,
                    self.mesh_info,
                    self.device,
                    self.mp_policy,
                    self.config.comm_fusion_zero_copy,
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

    def _init_hsdp_params(self):
        """init hsdp parameters and replicate parameters for cell."""
        replicate_params = set(self.config.replicate_params or ())
        # all parameters in the module tree(s), deduplicated
        ignored_params = set(self.config.ignored_params or ())
        visited_params = set()
        filtered_params = []
        for mod in self.modules:
            for _, param in mod.named_parameters():
                if param in ignored_params:
                    continue
                if hasattr(param, "_hsdp_param_initialized") and param._hsdp_param_initialized:
                    continue
                if param in visited_params:
                    continue
                visited_params.add(param)
                filtered_params.append(param)

        module_infos = _get_param_module_infos(filtered_params, tuple(self.modules))
        for param, module_info in zip(filtered_params, module_infos):
            param_mode = infer_fully_shard_param_mode(self.config.mesh, [param])
            enable_fsdp_shard = param not in replicate_params
            hsdp_param = TorchHSDPParamV2(param,
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
        trainable_params: list[TorchHSDPParamV2] = [
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

    def _validate_cpu_offload_params(self):
        """Validate that all parameters are on CPU when CPU offload policy is enabled."""
        if not isinstance(self.offload_policy, CPUOffloadPolicy):
            return
        hsdp_params_not_on_cpu = [
            hsdp_param
            for hsdp_param in self._iter_managed_params()
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
        if self.is_shard and not self._reset_sharded_params:
            for hsdp_param in self.hsdp_params:
                if hsdp_param.is_sharded:
                    hsdp_param.reset_sharded_param()
            self._reset_sharded_params = True
        self._validate_no_meta_params()
        self._validate_cpu_offload_params()
        self._init_mp_dtypes()

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            hsdp_param._param_fqn
            for hsdp_param in self._iter_managed_params()
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
        # Replicate-only params still use the non-fused compat all-reduce path.
        # Drain any pending side-path reductions before advancing the fused
        # param-group pipeline for sharded params.
        self.reduce_params()
        # Fused gradient reduction path: first apply any pending async reduction
        # from the previous module's backward (pipelined overlap), then issue
        # this module's fused reduce-scatter (+ all-reduce for HSDP).
        comm_ctx = get_comm_ctx()
        # Phase 2: apply grads for the param group whose all_reduce is done
        if comm_ctx.all_reduce_param_group is not None:
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
            comm_ctx.all_reduce_param_group = None
        # Phase 1: wait reduce_scatter, issue async all_reduce for previous layer
        if comm_ctx.pre_param_group is not None:
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if self.param_group is not None:
            self.param_group.foreach_reduce(
                reduce_scatter_reduce_op=self.reduce_op_type
            )
        for hsdp_param in self.replicate_params:
            if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                continue
            if not hsdp_param.sharded_param.requires_grad:
                continue
            if not self._has_pending_unsharded_grad(hsdp_param):
                continue
            reduce_op = self._resolve_reduce_op(hsdp_param)
            self._queue_compat_all_reduce(hsdp_param, reduce_op)

    def _resolve_default_reduce_op(self):
        """Resolve the default reduce op for the whole fully_shard state."""
        for hsdp_param in self._iter_managed_params():
            if hsdp_param.param_mode in (
                FullyShardParamMode.DTENSOR_COMPAT,
                FullyShardParamMode.DTENSOR_UNIFIED,
            ):
                return torch.distributed.ReduceOp.SUM
        return torch.distributed.ReduceOp.AVG

    def _resolve_reduce_op(self, hsdp_param=None):
        """Resolve the gradient reduction op for the current fully_shard state."""
        if self._user_reduce_op_type is not None:
            return self._user_reduce_op_type
        return self.reduce_op_type

    def _should_run_all_reduce(self, hsdp_param) -> bool:
        """Whether the current parameter should issue an all-reduce in this backward pass."""
        return self.requires_all_reduce and hsdp_param.dp_size > 1

    def _queue_reduce_scatter_then_all_reduce(self, hsdp_param, reduce_op):
        """Queue the standard FSDP/HSDP reduction path."""
        hsdp_param.reduce_scatter_grad(
            dtype=self._reduce_dtype,
            reduce_op=reduce_op,
        )
        HSDPState.pre_reduce_scatter_params.append((hsdp_param, self._orig_dtype))
        if not self._should_run_all_reduce(hsdp_param):
            return
        reduced_grad = hsdp_param.reduce_scatter_output()
        if (
            HSDPState.pre_reduce_scatter_params
            and HSDPState.pre_reduce_scatter_params[-1][0] == hsdp_param
        ):
            HSDPState.pre_reduce_scatter_params.pop()
        hsdp_param.all_reduce_grad(
            grad=reduced_grad,
            dtype=self._reduce_dtype,
            reduce_op=reduce_op,
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, self._orig_dtype))

    def _queue_compat_all_reduce(self, hsdp_param, reduce_op):
        """Queue the compatibility all-reduce path without FSDP sharding."""
        if not self._should_run_all_reduce(hsdp_param):
            return
        hsdp_param.all_reduce_grad(
            grad=self._get_pending_unsharded_grad(hsdp_param),
            dtype=self._reduce_dtype,
            reduce_op=reduce_op,
        )
        HSDPState.pre_all_reduce_params.append((hsdp_param, self._orig_dtype))

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

    def _queue_direct_compat_all_reduce(self, hsdp_param, reduce_op):
        """Queue all-reduce for DTENSOR_COMPAT params whose grad stays on ``sharded_param``."""
        grad = self._get_local_sharded_grad(hsdp_param)
        if grad is None:
            return
        reduced_grad = grad
        if self._reduce_dtype is not None and reduced_grad.dtype != self._reduce_dtype:
            reduced_grad = reduced_grad.to(self._reduce_dtype)
        handle = None
        if hsdp_param.unsharded_group_info.group is not None and hsdp_param.dp_size > 1:
            handle = torch.distributed.all_reduce(
                reduced_grad,
                op=reduce_op,
                group=hsdp_param.unsharded_group_info.group,
                async_op=True,
            )
        TorchHSDPStateV2.pre_direct_all_reduce_grads.append((handle, reduced_grad, grad))

    def post_backward(self, *unused):  # pylint: disable=unused-argument
        """Reduce gradients and reshard parameters after backward."""
        for hsdp_param in self._iter_managed_params():
            hsdp_param.accumulate_unsharded_grad_if_needed()
        if not self.reduce_grads:
            if self.reshard_after_backward:
                self.shard()
            for hsdp_param in self._iter_managed_params():
                hsdp_param.to_accumulated_grad_if_needed()
            return
        if not self.comm_fusion:
            # Handle user config replicate params and mirror params.
            self.reduce_params()
            for hsdp_param in self._iter_managed_params():
                if not hasattr(hsdp_param, "_unsharded_param") or hsdp_param.unsharded_param is None:
                    if self._can_direct_all_reduce_compat_grad(hsdp_param):
                        reduce_op = self._resolve_reduce_op(hsdp_param)
                        self._queue_direct_compat_all_reduce(hsdp_param, reduce_op)
            
            # Step 1: wait prev reduce_scatter (for params needing allreduce)
            prev_group = self._wait_prev_reduce_scatter()

            # Step 2: wait and apply prev reduce_scatter (for params NOT needing allreduce)
            self._wait_and_apply_prev_no_allreduce_params()

            # Step 3: issue current reduce_scatter
            self._issue_reduce_scatter_for_current_module()

            # Step 4: issue prev fused allreduce (async) - using saved prev_group
            self._issue_prev_fused_allreduce(prev_group)
        else:
            self.post_backward_for_comm_fusion()
        if self.reshard_after_backward:
            self.shard()

    def _issue_reduce_scatter_for_current_module(self):
        """Issue reduce_scatter for current module's parameters with fused all-reduce support.

        This method groups parameters by their replicate_process_group and:
        1. For params without all_reduce needs: issue reduce_scatter directly
        2. For params with all_reduce needs: allocate fused buffer and issue reduce_scatter
           into aligned views, enabling zero-copy fused all_reduce later.
        """
        # Collect parameters that need gradient reduction
        params_to_reduce = []
        for hsdp_param in self._iter_managed_params():
            skip_param = (not hasattr(hsdp_param, "_unsharded_param")
                            or hsdp_param.unsharded_param is None
                            or not hsdp_param.sharded_param.requires_grad
                            or self._can_direct_all_reduce_compat_grad(hsdp_param)
                            or (hsdp_param.unsharded_param.grad is None
                            and hsdp_param.unsharded_accumulated_grad_data is None))
            if skip_param:
                continue
            params_to_reduce.append(hsdp_param)

        if not params_to_reduce:
            return

        # Group by replicate_process_group for fused all-reduce
        # Key: id of process group, or None for params that don't need all_reduce
        groups_by_comm = defaultdict(list)
        for hsdp_param in params_to_reduce:
            if self._should_run_all_reduce(hsdp_param):
                key = id(hsdp_param.unsharded_group_info.group)
                groups_by_comm[key].append(hsdp_param)
            else:
                groups_by_comm[None].append(hsdp_param)

        # Handle params that don't need all_reduce (FSDP or single replica)
        if None in groups_by_comm:
            for hsdp_param in groups_by_comm[None]:
                hsdp_param.reduce_scatter_grad(
                    dtype=self._reduce_dtype,
                    reduce_op=self._resolve_reduce_op()
                )
                HSDPState.pre_reduce_scatter_params.append(
                    (hsdp_param, self._orig_dtype))

        # Handle params that need all_reduce (HSDP with multiple replicas)
        for key, hsdp_params in groups_by_comm.items():
            if key is None:
                continue

            # Create AllReduceParamGroup for fused all-reduce
            group = AllReduceParamGroup(
                replicate_group=hsdp_params[0].unsharded_group_info.group,
                hsdp_params=hsdp_params,
                orig_dtypes=[self._orig_dtype] * len(hsdp_params),
                reduce_dtype=self._reduce_dtype,
                reduce_op=self._resolve_reduce_op(),
                mp_policy=self.mp_policy,
            )

            # Allocate fused buffer with 512-byte alignment
            group.allocate_fused_buffer(self.device)

            # Issue reduce_scatter with output directly into fused buffer views
            for idx, hsdp_param in enumerate(hsdp_params):
                buffer_view = group.get_param_buffer_view(idx)
                hsdp_param.reduce_scatter_grad(
                    dtype=self._reduce_dtype,
                    reduce_op=self._resolve_reduce_op(),
                    output_buffer=buffer_view,
                )

            # Save group for later all_reduce in reduce_params()
            TorchHSDPStateV2.pre_all_reduce_groups.append(group)

    def _wait_prev_reduce_scatter(self) -> List[AllReduceParamGroup]:
        """Step 1: wait prev reduce_scatter.

        This enables overlapping:
        - Layer N-1's reduce_scatter wait with Layer N's backward compute

        Returns:
            List of previous AllReduceParamGroups (one per communication group).
        """
        if TorchHSDPStateV2.pre_all_reduce_groups:
            prev_groups = list(TorchHSDPStateV2.pre_all_reduce_groups)
            TorchHSDPStateV2.pre_all_reduce_groups.clear()
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

    def _issue_prev_fused_allreduce(self, prev_groups: List[AllReduceParamGroup]):
        """Step 4: issue previous module's fused allreduce (async).

        The allreduce handle is collected in pending_all_reduce_groups,
        and will be processed in root_backward_hook's delay_apply_reduce_grads().

        Args:
            prev_groups: List of previous AllReduceParamGroups to issue allreduce for.
        """
        for prev_group in prev_groups:
            prev_group.accumulate_existing_grads_to_buffer()
            prev_group.issue_async_allreduce()
            # Move to pending queue for root_backward_hook to process
            TorchHSDPStateV2.pending_all_reduce_groups.append(prev_group)

    def _wait_and_apply_prev_no_allreduce_params(self):
        """Step 2: wait and apply previous reduce_scatter for params NOT needing allreduce.

        These are FSDP params or single-replica HSDP params that don't need
        cross-replica allreduce. Their reduce_scatter was issued by the previous
        module's _issue_reduce_scatter_for_current_module(), and we wait and apply here.
        """
        need_synchronize = False
        while HSDPState.pre_reduce_scatter_params:
            pre_hsdp_param, pre_orig_dtype = HSDPState.pre_reduce_scatter_params.pop(0)
            reduced_grad = pre_hsdp_param.reduce_scatter_output()
            pre_hsdp_param.clear_reduce_scatter_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype) or need_synchronize
            pre_hsdp_param.accumulated_allreduced_grad = False

        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload."
                )

    @classmethod
    def delay_apply_reduce_grads(cls, device: torch.device):
        """Apply all pending allreduce gradients in root_backward_hook.

        This is called at the end of root_backward_hook to wait for all
        async allreduce operations and apply gradients to sharded parameters.

        Args:
            device: Device for CPU offload synchronization.
        """
        need_synchronize = False

        for group in cls.pending_all_reduce_groups:
            need_synchronize = group.wait_and_apply_grads() or need_synchronize

        cls.pending_all_reduce_groups.clear()

        if need_synchronize:
            if device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {device.type} for synchronization after CPU offload."
                )


    def reduce_scattered_params(self):
        """
        reduce_scattered_params
        """
        need_synchronize = False
        while HSDPState.pre_reduce_scatter_params:
            pre_hsdp_param, pre_orig_dtype = HSDPState.pre_reduce_scatter_params.pop(0)
            reduced_grad = pre_hsdp_param.reduce_scatter_output()
            pre_hsdp_param.clear_reduce_scatter_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype) or need_synchronize
            pre_hsdp_param.accumulated_allreduced_grad = False
        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload."
                )

    def reduce_params(self):
        """Apply reduced gradients from pre-staged HSDP parameters to sharded parameters.

        This function processes two lists of pre-queued HSDP parameters (`pre_reduce_scatter_params`
        and `pre_all_reduce_params`), retrieves the reduced gradients from asynchronous
        reduce-scatter/all-reduce operations, clears cached communication outputs, and applies
        the reduced gradients to the corresponding sharded parameters (including reshaping,
        dtype conversion, optional CPU offloading, and gradient accumulation/assignment).

        Note:
            - Parameters are processed in **FIFO (First-In-First-Out)** order (via `pop(0)`), ensuring
            gradient application order matches the order of gradient reduction operations.
            - After retrieving the reduced gradient, the cached communication output (reduce_scatter_output
            or all_reduce_output) is cleared to free memory and avoid stale data.
            - Gradient application logic (in `apply_reduced_grad`) includes:
            1. Reshaping the flat reduced gradient to match the local shard shape
            2. Optional dtype conversion to `param_type`
            3. Optional CPU offloading (per the HSDP parameter's offload policy)
            4. Assigning or accumulating the gradient to `sharded_param.grad`
        """
        need_synchronize = False
        while HSDPState.pre_all_reduce_params:
            pre_hsdp_param, pre_orig_dtype = HSDPState.pre_all_reduce_params.pop(0)
            reduced_grad = pre_hsdp_param.all_reduce_output()
            pre_hsdp_param.clear_all_reduce_output()
            need_synchronize = pre_hsdp_param.apply_reduced_grad(reduced_grad, pre_orig_dtype) or need_synchronize

        while TorchHSDPStateV2.pre_direct_all_reduce_grads:
            handle, reduced_grad, target_grad = TorchHSDPStateV2.pre_direct_all_reduce_grads.pop(0)
            if handle is not None:
                handle.wait()
            if reduced_grad is not target_grad:
                if reduced_grad.dtype != target_grad.dtype:
                    reduced_grad = reduced_grad.to(target_grad.dtype)
                target_grad.copy_(reduced_grad)
        if need_synchronize:
            if self.device.type == "npu":
                torch.npu.current_stream().synchronize()
            elif self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            else:
                raise NotImplementedError(
                    f"Unsupported device type {self.device.type} for synchronization after CPU offload."
                )

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.reduce_grads = requires_grad_sync

    @property
    def _is_hsdp(self) -> bool:
        return isinstance(self.mesh_info, HSDPMeshInfo)

    def set_reduce_op_type(self, reduce_op_type: str):
        """set reduce op type for gradient reduction."""
        fsdp_support_reduce_op = {
            "sum": torch.distributed.ReduceOp.SUM,
            "avg": torch.distributed.ReduceOp.AVG,
        }
        if reduce_op_type not in fsdp_support_reduce_op:
            raise ValueError(
                f"Unsupported reduce op type {reduce_op_type}, "
                f"supported types are {list(fsdp_support_reduce_op.keys())}"
            )
        reduce_op: str = reduce_op_type.lower().strip()
        self._user_reduce_op_type = fsdp_support_reduce_op[reduce_op]
        self.reduce_op_type = self._user_reduce_op_type
