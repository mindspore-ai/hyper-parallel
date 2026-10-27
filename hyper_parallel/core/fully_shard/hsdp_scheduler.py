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
"""HSDP scheduler"""
import functools
import inspect
from typing import Any, Callable, List, Mapping, Optional, ParamSpec, Protocol, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FSDPSchedulerState,
    get_managed_modules_parameters,
)
from hyper_parallel.core.fully_shard.utils import (
    CommFusionPolicy,
    SourceShardMetaInfo,
    apply_to_tensors,
    cast_fp_tensor,
    parameters_dict,
    profiler_record,
)
from hyper_parallel.tools.logging import get_logger

logger = get_logger("FSDP")

nn.Module = nn.Module

_P = ParamSpec("_P")
_R = TypeVar("_R")


class FSDPAllGatherExtension(Protocol):
    """参数恢复的可选前后处理接口，FSDP 始终持有参数及其通信句柄。

    Example:
        扩展可在前置钩子提供转换后的通信输入；FSDP 完成 all-gather 后，
        后置钩子生成调用方视图。未注册扩展的模块沿用原路径。
    """

    def fsdp_pre_all_gather(self, scheduler: "HSDPSchedulerV2", context: Any) -> dict[int, torch.Tensor]:
        """返回参数 id 到通信输入缓冲区的映射。"""

    def fsdp_post_all_gather(self, scheduler: "HSDPSchedulerV2", context: Any) -> Any:
        """在所有 AG 完成后产生调用方视图。"""

    def fsdp_after_backward_prefetch(self, scheduler: "HSDPSchedulerV2") -> None:
        """原生反向 AG 预取提交后继续执行扩展的准备工作。"""


class FSDPGradientExtension(Protocol):
    """在原生梯度归约前调度附加处理。

    前置钩子在计算产生梯度时发起通信，后置钩子在 FSDP 消费参数梯度前
    等待并合并结果。普通单元不注册扩展，沿用原生 RS/AR 流程。
    """

    def fsdp_pre_gradient_reduce(self, scheduler: "HSDPSchedulerV2", context: Any) -> Any:
        """提交附加处理并返回供自动求导累积的梯度。"""

    def fsdp_post_gradient_reduce(self, scheduler: "HSDPSchedulerV2") -> None:
        """在原生梯度归约前完成扩展的等待与归并。"""


def _dynamo_disable(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Disable Dynamo tracing while an FSDP runtime hook executes."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return torch._dynamo.disable(
            func,
            recursive=True,
        )(*args, **kwargs)

    return wrapper


class ParamGroupCommCtx:
    """Track the in-flight parameter-group communication for one module tree."""

    def __init__(self) -> None:
        self.pre_param_group = None
        self.all_reduce_param_group = None
        self.comm_handle = None


class HSDPSchedulerContext:
    """Share scheduler and backward-pipeline state within one HSDP module tree."""

    def __init__(self) -> None:
        self.is_last_backward: bool = True
        self.root_module = None
        # Compile tracing may enter the root more than once; initialize shared
        # parameter state only on the first real forward.
        self.lazy_init_done: bool = False
        self.root_bp_state = False
        # all_hsdp_schedulers (for one module tree structure), deduplicated at
        # registration time because ``fully_shard`` may be given a module list
        # whose modules share one scheduler.
        self.all_hsdp_schedulers = []
        # Parameter FQNs are initialized once after all schedulers share this context.
        self._param_fqn_initialized = False
        # Backward pipeline queues shared only by schedulers in this module tree.
        self.pre_reduce_scatter_params = []
        self.pre_all_reduce_params = []
        self.pre_direct_all_reduce_grads = []
        self.pre_all_reduce_groups = []
        self.pending_all_reduce_groups = []
        self.param_group_comm_ctx = ParamGroupCommCtx()


class HSDPSchedulerV2:
    """HSDPScheduler is used to scheduler hsdp"""

    def __init__(
        self,
        cell: Union[nn.Module, Tuple[nn.Module, ...]],
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params,
        replicate_params,
        device,
        comm_fusion,
        comm_fusion_zero_copy=False,
        source_shard_infos: Optional[Mapping[nn.Parameter, SourceShardMetaInfo]] = None,
    ):
        """init hsdp scheduler.

        Args:
            cell: A single module or tuple of modules to manage as one FSDP unit.
            mesh: Explicit data-parallel device mesh.
            reshard_after_forward: Whether to reshard parameters after forward.
            shard_placement_fn: Optional function selecting the parameter shard dimension.
            mp_policy: Mixed-precision policy.
            offload_policy: Parameter offload policy.
            ignored_params: Parameters excluded from fully_shard management.
            replicate_params: Managed parameters that remain replicated.
            device: Target device for parameters.
            comm_fusion: Communication fusion configuration.
            comm_fusion_zero_copy: Whether fused communication writes into views.
            source_shard_infos: Optional per-parameter TP/EP source metadata.
        """
        self.modules = (cell,) if isinstance(cell, nn.Module) else tuple(cell)
        self.cell = self.modules[0]
        self.mesh: DeviceMesh = mesh
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.comm_fusion_policy = CommFusionPolicy(comm_fusion, comm_fusion_zero_copy)
        self.ignored_params = ignored_params
        self.replicate_params = replicate_params
        self.device = device
        self.reshard_after_forward = reshard_after_forward
        self.source_shard_infos = source_shard_infos
        self.scheduler_state = None
        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []
        self._backup_forward_fetch = None
        # 自定义反向可能先返回输入梯度；输入边就绪不代表参数梯度已累积。
        self._param_grad_gate = False
        self._param_grad_expected = set()
        self._param_grad_ready = set()
        self._param_grad_handles = {}
        self._param_grad_input_done = False
        self._all_gather_extension: Optional[FSDPAllGatherExtension] = None
        self._gradient_extension: Optional[FSDPGradientExtension] = None
        # Flag to identify root module.
        self._is_root = True
        # module and its all sub-modules share one same 'HSDPSchedulerContext'
        self.scheduler_ctx = HSDPSchedulerContext()
        # When ``fully_shard`` is given multiple root modules, forward pre/post hooks coordinate
        # so unshard / PostBackward / reshard run once per forward (aligned with PyTorch FSDP2).
        self._fsdp_group_post_pending: Optional[set] = set() if len(self.modules) > 1 else None
        self._new_cell_state()
        self._register_hooks()

    def _new_cell_state(self):
        """Create the HSDP state that owns this scheduler's managed parameters."""
        # Imported lazily: ``state`` pulls in ``param_group``, which imports this
        # module for ``ParamGroupCommCtx``.
        from hyper_parallel.core.fully_shard.hsdp_state import (  # pylint: disable=C0415
            HSDPStateV2,
        )

        self.hsdp_state = HSDPStateV2(
            self.modules,
            self.mesh,
            self.shard_placement_fn,
            self.comm_fusion_policy,
            self.mp_policy,
            self.offload_policy,
            self.ignored_params,
            self.replicate_params,
            self.scheduler_ctx,
            self.device,
            source_shard_infos=self.source_shard_infos,
        )

    def _register_hooks(self):
        """Register the forward and backward hooks driving shard/unshard."""
        self._register_forward_backward_hooks()

    def _get_managed_params(self):
        """Return deduplicated parameters from all managed modules."""
        return get_managed_modules_parameters(self.modules, self.ignored_params)

    def set_reshard_after_forward(self, reshard_after_forward: bool) -> None:
        """Set reshard_after_forward flag.

        Args:
            reshard_after_forward: Whether to reshard parameters after forward.
        """
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}")
        self.reshard_after_forward = reshard_after_forward

    def set_reshard_after_backward(self, reshard_after_backward: bool) -> None:
        """Set reshard_after_backward flag.

        Args:
            reshard_after_backward: Whether to reshard after backward completes.
        """
        if not isinstance(reshard_after_backward, bool):
            raise ValueError(f"reshard_after_backward should be a bool, got {type(reshard_after_backward)}")
        if self.hsdp_state is not None:
            self.hsdp_state.reshard_after_backward = reshard_after_backward

    def set_requires_all_reduce(self, requires_all_reduce: bool) -> None:
        """Set requires_all_reduce flag.

        Args:
            requires_all_reduce: Whether this unit participates in all-reduce.
        """
        if not isinstance(requires_all_reduce, bool):
            raise ValueError(f"requires_all_reduce should be a bool, got {type(requires_all_reduce)}")
        if self.hsdp_state is not None:
            self.hsdp_state.set_requires_all_reduce(requires_all_reduce)

    @_dynamo_disable
    def reset_iter_state(self) -> None:
        """Reset scheduler bookkeeping after a completed iteration."""
        self.scheduler_ctx.root_bp_state = False
        self.scheduler_ctx.pre_reduce_scatter_params.clear()
        self.scheduler_ctx.pre_all_reduce_params.clear()
        self.scheduler_ctx.pre_direct_all_reduce_grads.clear()
        self.scheduler_ctx.pre_all_reduce_groups.clear()
        self.scheduler_ctx.pending_all_reduce_groups.clear()
        self.scheduler_ctx.param_group_comm_ctx.pre_param_group = None
        self.scheduler_ctx.param_group_comm_ctx.all_reduce_param_group = None
        self.scheduler_ctx.param_group_comm_ctx.comm_handle = None
        self.scheduler_state = None
        if self._fsdp_group_post_pending is not None:
            self._fsdp_group_post_pending.clear()
        self._restore_forward_prefetch_after_recompute()
        self.hsdp_state.reset_iter_state()

    def set_requires_grad_sync(self, requires_grad_sync: bool) -> None:
        """Set flag controlling whether gradients are synchronized.

        Args:
            requires_grad_sync: When True, enable grad sync for this scheduler.
        """
        if not isinstance(requires_grad_sync, bool):
            raise ValueError(f"requires_grad_sync should be a bool, got {type(requires_grad_sync)}")
        self.hsdp_state.set_requires_grad_sync(requires_grad_sync)

    # pylint: disable=W0613
    def _hsdp_forward_pre_hook(self, cell, args, kwargs):
        """Forward pre hook to unsharded parameter for forward process."""
        logger.debug("hook=forward_pre enter module=%s", self.hsdp_state)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            logger.debug("hook=forward_pre skip module=%s reason=pre_backward", self.hsdp_state)
            return args, kwargs
        if self.scheduler_ctx.root_bp_state:
            self._disable_forward_prefetch_for_recompute()
        if self.scheduler_ctx.root_module is None:
            tree_ctx = self.scheduler_ctx
            tree_ctx.root_module = self.cell
            registered_schedulers = set()
            for module_name, module in tree_ctx.root_module.named_modules():
                from hyper_parallel.core.fully_shard.api import HSDPModule  # pylint: disable=C0415
                if isinstance(module, HSDPModule):
                    submod_scheduler = module.hsdp_scheduler
                    if submod_scheduler is None or id(submod_scheduler) in registered_schedulers:
                        continue
                    registered_schedulers.add(id(submod_scheduler))
                    if submod_scheduler.scheduler_ctx is not tree_ctx:
                        if submod_scheduler.scheduler_ctx.root_module is not None:
                            raise ValueError(
                                "HSDP scheduler already belongs to another initialized module tree"
                            )
                        submod_scheduler.scheduler_ctx = tree_ctx
                        submod_scheduler.hsdp_state.scheduler_ctx = tree_ctx
                        if submod_scheduler.hsdp_state.param_group is not None:
                            submod_scheduler.hsdp_state.param_group.comm_ctx = tree_ctx.param_group_comm_ctx
                    submod_scheduler._is_root = submod_scheduler is self  # pylint: disable=protected-access
                    submod_scheduler.hsdp_state.module_name = module_name
                    tree_ctx.all_hsdp_schedulers.append(submod_scheduler)

        self.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        if self._is_root and not self.scheduler_ctx.lazy_init_done:
            self._init_params_fqn()
            self._lazy_init_all_states()
            self.scheduler_ctx.lazy_init_done = True
        if self.mp_policy.cast_forward_inputs and self.mp_policy.param_dtype:
            cast_fn = functools.partial(cast_fp_tensor, self.mp_policy.param_dtype)
            args = apply_to_tensors(cast_fn, args)
            kwargs = apply_to_tensors(cast_fn, kwargs)
        with profiler_record(f"pre_forward unshard:{self.hsdp_state.module_name}"):
            logger.debug("hook=forward_pre action=unshard module=%s", self.hsdp_state)
            self.hsdp_state.unshard()
        if self._param_grad_gate and not self.scheduler_ctx.root_bp_state:
            self._arm_param_grad_gate()
        for prefetch_cell in self.forward_prefetch_cells:
            prefetch_state = prefetch_cell.hsdp_scheduler.hsdp_state
            with profiler_record(f"pre_forward prefetch:"
                                 f"{prefetch_state.module_name}"):
                logger.debug(
                    "hook=forward_pre action=prefetch module=%s target=%s",
                    self.hsdp_state,
                    prefetch_state,
                )
                prefetch_state.prefetch()
        return args, kwargs

    def _lazy_init_all_states(self):
        if self._is_root and self.scheduler_ctx.root_module is not None:
            for submod_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
                hsdp_state = submod_scheduler.hsdp_state
                if hsdp_state:
                    hsdp_state.lazy_init()

    def _init_params_fqn(self):  # pylint: disable=W0212
        if not self._is_root or self.scheduler_ctx.root_module is None:
            return
        if self.scheduler_ctx._param_fqn_initialized:  # pylint: disable=protected-access
            return
        # Build a map from original (sharded) parameter tensor to its HSDPParam wrapper.
        param_to_hsdp_param = {}
        for submod_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            hsdp_state = submod_scheduler.hsdp_state
            if hsdp_state is None:
                continue
            for hsdp_param in hsdp_state.hsdp_params:
                orig_param = hsdp_param.sharded_param
                # Shared parameters: keep only the first mapping to preserve the
                # first-seen FQN (consistent with the deduplication in _init_hsdp_params).
                if orig_param not in param_to_hsdp_param:
                    param_to_hsdp_param[orig_param] = hsdp_param

        # Walk the full parameter tree and assign FQNs; skip params already seen
        # (shared-parameter deduplication: first name wins).
        visited_params = set()
        for param_name, parameter in parameters_dict(self.scheduler_ctx.root_module):
            if parameter in visited_params:
                continue
            visited_params.add(parameter)
            hsdp_param = param_to_hsdp_param.get(parameter)
            if hsdp_param is not None:
                hsdp_param._param_fqn = param_name  # pylint: disable=W0212
        self.scheduler_ctx._param_fqn_initialized = True  # pylint: disable=protected-access

    # pylint: disable=W0613, R1710
    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """Forward hook to shard parameter for saving memory."""
        logger.debug("hook=forward enter module=%s", self.hsdp_state)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            logger.debug("hook=forward skip module=%s reason=pre_backward", self.hsdp_state)
            return
        self.scheduler_state = FSDPSchedulerState.FORWARD
        if self.reshard_after_forward:
            with profiler_record(f"forward reshard:{self.hsdp_state.module_name}"):
                logger.debug("hook=forward action=reshard module=%s", self.hsdp_state)
                self.hsdp_state.shard()
        if self.mp_policy.output_dtype is not None:
            outputs = apply_to_tensors(
                functools.partial(cast_fp_tensor, self.mp_policy.output_dtype),
                outputs,
            )
        return outputs

    # pylint: disable=W0613
    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """Backward pre hook to unsharded parameter for backward process."""
        logger.debug("hook=backward_pre enter module=%s", self.hsdp_state)
        self.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        if self.reshard_after_forward:
            with profiler_record(f"pre_backward unshard:{self.hsdp_state.module_name}"):
                logger.debug("hook=backward_pre action=unshard module=%s", self.hsdp_state)
                self.hsdp_state.unshard()
        for prefetch_cell in self.backward_prefetch_cells:
            prefetch_state = prefetch_cell.hsdp_scheduler.hsdp_state
            with profiler_record(f"pre_backward prefetch:"
                                 f"{prefetch_state.module_name}"):
                logger.debug(
                    "hook=backward_pre action=prefetch module=%s target=%s",
                    self.hsdp_state,
                    prefetch_state,
                )
                prefetch_state.prefetch()

        # 先提交全部原生 AG，再为每个带扩展的目标提交后续工作。
        for prefetch_cell in self.backward_prefetch_cells:
            prefetch_state = prefetch_cell.hsdp_scheduler.hsdp_state
            with profiler_record(f"post_backward prefetch:"
                                 f"{prefetch_state.module_name}"):
                extension = prefetch_cell.hsdp_scheduler._all_gather_extension  # pylint: disable=protected-access
                if extension is not None and hasattr(extension, "fsdp_after_backward_prefetch"):
                    extension.fsdp_after_backward_prefetch(prefetch_cell.hsdp_scheduler)

    # pylint: disable=W0613
    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """Backward hook to shard parameter for optimizer process or saving memory."""
        logger.debug("hook=backward_hook enter module=%s", self.hsdp_state)
        self.scheduler_state = FSDPSchedulerState.BACKWARD
        with profiler_record(f"post_backward:{self.hsdp_state.module_name}"):
            if self._gradient_extension is not None:
                with torch.profiler.record_function("fsdp::gradient_extension"):
                    self._gradient_extension.fsdp_post_gradient_reduce(self)
            logger.debug("hook=backward_hook action=post_backward module=%s", self.hsdp_state)
            self.hsdp_state.post_backward()
        if self._fsdp_group_post_pending is not None:
            self._fsdp_group_post_pending.clear()

    # pylint: disable=W0613
    @staticmethod
    def _grouped_forward_pre_hook_skip(cell, args, kwargs) -> None:
        """Return value when grouped pre-forward should not run (first module already did).

        ``nn.Module`` pre-hooks use ``None`` for a no-op.
        """
        return None

    @staticmethod
    def _grouped_forward_post_hook_skip(outputs) -> None:
        """Return value when grouped post-forward is deferred to a later module.

        ``nn.Module`` forward hooks use ``None`` for a no-op.
        """
        return None

    @_dynamo_disable
    def _grouped_forward_pre_hook(self, cell, args, kwargs):
        """Run FSDP pre-forward only for the first module in the group (PyTorch FSDP2-aligned)."""
        pending = self._fsdp_group_post_pending
        if pending is None:
            return self._forward_pre_hook(cell, args, kwargs)
        if len(pending) == 0:
            pending.update(self.modules)
            return self._forward_pre_hook(cell, args, kwargs)
        return self._grouped_forward_pre_hook_skip(cell, args, kwargs)

    def _make_grouped_forward_post_hook(self, mod):
        """Build post-forward hook: last module in the group runs reshard + output backward hooks."""

        def grouped_post_hook(cell, inputs, outputs):
            pending = self._fsdp_group_post_pending
            if pending is None:
                return self._forward_hook(cell, inputs, outputs)
            pending.discard(mod)
            if len(pending) == 0:
                return self._forward_hook(cell, inputs, outputs)
            return self._grouped_forward_post_hook_skip(outputs)

        return grouped_post_hook

    def set_forward_prefetch_cells(self, hsdp_cell_list: List[Any]) -> None:
        """Set cells prefetched during forward.

        Args:
            hsdp_cell_list: HSDP cells to prefetch ahead of forward.
        """
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list: List[Any]) -> None:
        """Set cells prefetched during backward.

        Args:
            hsdp_cell_list: HSDP cells to prefetch ahead of backward.
        """
        self.backward_prefetch_cells = hsdp_cell_list

    def register_all_gather_extension(self, extension: FSDPAllGatherExtension) -> None:
        """为本 FSDP 单元注册参数恢复扩展，不改变普通前向或梯度归约。

        Args:
            extension: 同时实现 fsdp_pre_all_gather 和 fsdp_post_all_gather 的对象。

        Example:
            扩展在前置钩子提供通信输入，FSDP 等待恢复完成后把参数视图交回扩展。
        """
        if (not callable(getattr(extension, "fsdp_pre_all_gather", None))
                or not callable(getattr(extension, "fsdp_post_all_gather", None))):
            raise ValueError("all-gather 扩展必须同时实现前置和后置 hook")
        if self._all_gather_extension is not None and self._all_gather_extension is not extension:
            raise ValueError("同一 FSDP 单元不能注册多个 all-gather 扩展")
        self._all_gather_extension = extension

    def register_gradient_extension(self, extension: FSDPGradientExtension) -> None:
        """登记原生 FSDP 梯度归约前的扩展。

        Args:
            extension: 实现前置发起和后置等待、归并两个 hook 的对象。

        Example:
            扩展在梯度产生时发起通信，并在 post-backward 的原生归约前完成归并。
        """
        if (not callable(getattr(extension, "fsdp_pre_gradient_reduce", None))
                or not callable(getattr(extension, "fsdp_post_gradient_reduce", None))):
            raise ValueError("梯度扩展必须同时实现前置和后置 hook")
        if self._gradient_extension is not None and self._gradient_extension is not extension:
            raise ValueError("同一 FSDP 单元不能注册多个梯度扩展")
        self._gradient_extension = extension

    def launch_gradient_extension(self, context: Any) -> Any:
        """在梯度产生时提交扩展处理。

        Args:
            context: 扩展定义的梯度与处理上下文。

        Returns:
            扩展给当前自定义反向函数的本地梯度。
        """
        if self._gradient_extension is None:
            raise RuntimeError("本 FSDP 单元没有注册梯度扩展")
        return self._gradient_extension.fsdp_pre_gradient_reduce(self, context)

    def restore_with_all_gather_extension(self, context: Any) -> Any:
        """执行前置 hook、FSDP AG、等待和后置 hook，返回扩展产生的视图。

        Args:
            context: 由扩展定义的单次恢复上下文。

        Returns:
            后置 hook 生成的参数视图。

        Example:
            外部调度器为下一单元恢复参数，扩展在 FSDP 完成 AG 后读取视图。
        """
        extension = self._all_gather_extension
        if extension is None:
            raise RuntimeError("本 FSDP 单元没有注册 all-gather 扩展")
        inputs = extension.fsdp_pre_all_gather(self, context) if self.hsdp_state.is_shard else {}
        if inputs:
            self.hsdp_state.unshard(async_op=True, input_buffers=inputs)
        else:
            self.hsdp_state.unshard(async_op=True)
        self.hsdp_state.wait_for_unshard()
        return extension.fsdp_post_all_gather(self, context)

    def _disable_forward_prefetch_for_recompute(self) -> None:
        """Temporarily disable forward prefetch during activation recompute."""
        self._backup_forward_fetch = self.forward_prefetch_cells
        self.forward_prefetch_cells = []

    def _restore_forward_prefetch_after_recompute(self) -> None:
        """Restore the saved forward prefetch list after activation recompute."""
        if self._backup_forward_fetch is None:
            return
        self.forward_prefetch_cells = self._backup_forward_fetch
        self._backup_forward_fetch = None


    def _has_user_post_accumulate_hooks(self) -> bool:
        """检查完整参数是否还有用户累积后钩子；例如后注册的裁剪应先于 FSDP 消费。

        门控钩子可能排在用户钩子之前；存在后者时把本单元归约推迟至 root 收尾。
        """
        for parameter in self.hsdp_state.hsdp_params:
            tensor = parameter.unsharded_param
            hooks = tensor._post_accumulate_grad_hooks or {}
            gate = self._param_grad_handles.get(id(tensor))
            if hooks and (gate is None or len(hooks) != 1 or gate.id not in hooks):
                return True
        return False

    def enable_param_grad_gate(self) -> None:
        """为独立参数单元开启输入边与AccumulateGrad双条件门控。

        Example:
            自定义反向先交付输入梯度时，延迟归约直到本单元的参数梯度都已累积。
        """
        self._param_grad_gate = True

    def _arm_param_grad_gate(self) -> None:
        """为一次原始前向登记实际受管参数；每个参数需就绪一次。"""
        parameters = [param.unsharded_param for param in self.hsdp_state.hsdp_params
                      if param.unsharded_param.requires_grad]
        self._param_grad_expected = {id(param) for param in parameters}
        self._param_grad_ready = set()
        self._param_grad_input_done = False
        for param in parameters:
            identity = id(param)
            if identity not in self._param_grad_handles:
                self._param_grad_handles[identity] = param.register_post_accumulate_grad_hook(
                    self._on_param_grad_ready,
                )

    def _on_param_grad_ready(self, param) -> None:
        """在真实参数累积之后重试被挂起的post-backward；例如第二矩阵触发RS。"""
        self._param_grad_ready.add(id(param))
        if self._param_grad_input_done:
            self._backward_hook()

    def _register_post_backward_hook(self, args, kwargs):
        """Wrap forward args/kwargs through PostBackwardFunction to register backward hook."""
        if not torch.is_grad_enabled():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        if len(inp_tensors) == 0:
            return args, kwargs  # no tensors that require gradients
        processed_tensors = PostBackwardFunction.apply(self, *inp_tensors)
        for inp_tensor_idx, processed_tensor in zip(inp_tensor_indices, processed_tensors):
            args_kwargs_list[inp_tensor_idx] = processed_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs

    @_dynamo_disable
    def _forward_pre_hook(self, cell, args, kwargs):
        """Execute forward pre hook and set up backward hook."""
        args, kwargs = self._hsdp_forward_pre_hook(cell, args, kwargs)
        return self._register_post_backward_hook(args, kwargs)

    def _register_backward_pre_hook(self, outputs) -> None:
        """Register gradient hooks on all requires-grad outputs to trigger backward pre hook."""
        flat_outputs, _ = tree_flatten(outputs)
        for output in flat_outputs:
            if isinstance(output, torch.Tensor) and output.requires_grad:
                handle_ref = [None]
                # pylint: disable=C0103, W0102

                def wrapper_for_backward_pre_hook(grad, _handle_ref=handle_ref):
                    """Remove this hook after it fires to prevent accmulation"""
                    handle = _handle_ref[0]
                    if handle is not None:
                        handle.remove()
                    return self._backward_pre_hook(grad)
                # pylint: enable=C0103, W0102
                handle = output.register_hook(wrapper_for_backward_pre_hook)
                handle_ref[0] = handle

    @_dynamo_disable
    def _forward_hook(self, cell, inputs, outputs):  # pylint: disable=R1710
        """Execute forward hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self._register_backward_pre_hook(outputs)
        if self.scheduler_ctx.root_bp_state:
            self._restore_forward_prefetch_after_recompute()
            return
        return self._hsdp_forward_hook(cell, inputs, outputs)

    # pylint: disable=W0212
    @_dynamo_disable
    def _backward_pre_hook(self, grad):
        """Execute backward pre hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        if self._is_root:
            Variable._execution_engine.queue_callback(self._root_backward_hook)
        self.scheduler_ctx.root_bp_state = True
        if self._param_grad_gate:
            # 两个保留的前向可分别反向；本次不能沿用上次的参数就绪标志。
            self._param_grad_ready = set()
            self._param_grad_input_done = False
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    @_dynamo_disable
    def _root_backward_hook(self):
        """Drain all DP pipelines, then run final TP reduction and apply gradients."""
        logger.debug("hook=root_backward_hook enter module=%s", self.hsdp_state)
        self.check_state_prefetch_handles_consumed()
        for hsdp_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            # let modules which are not triggered backward_hook launch backward communication.
            if getattr(hsdp_scheduler, "_param_grad_gate", False) is True:
                hsdp_scheduler._backward_hook(force=True)
            else:
                hsdp_scheduler._backward_hook()
        self.scheduler_ctx.root_bp_state = False
        with torch.profiler.record_function(f"root_backward reduce:{self.hsdp_state.module_name}"):
            self._finalize_comm_fusion_reductions()
            self._finalize_per_param_reductions()
            self.launch_tp_replicate_reduce_and_apply()

    def check_state_prefetch_handles_consumed(self) -> None:
        """Check that all parameters in the module tree consumed their prefetch work.

        Raises:
            RuntimeError: If a parameter or fused bucket retains an all-gather
                handle because the configured prefetch order is incorrect.
        """
        unconsumed_param_fqns = []
        for hsdp_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            hsdp_state = hsdp_scheduler.hsdp_state
            if hsdp_state is None:
                continue
            unconsumed_param_fqns.extend(
                str(param) for param in hsdp_state.hsdp_params
                if param.allgather_comm_ctx.allgather_handle is not None
            )
            if hsdp_state.param_group is not None:
                for bucket in hsdp_state.param_group.all_gather_buckets:
                    result = bucket.all_gather_result
                    if result is not None and result.handle is not None:
                        unconsumed_param_fqns.extend(str(param) for param in bucket.hsdp_params)
        if unconsumed_param_fqns:
            param_fqns = ", ".join(dict.fromkeys(unconsumed_param_fqns))
            raise RuntimeError(
                f"Unconsumed prefetch handles for parameters: {param_fqns}. "
                "The prefetch order is incorrect. Set forward/backward prefetch order "
                "according to the model's actual execution order."
            )

    def _finalize_comm_fusion_reductions(self) -> None:
        """Drain the comm_fusion=True RS/AR pipeline."""
        comm_ctx = self.scheduler_ctx.param_group_comm_ctx
        if comm_ctx.all_reduce_param_group is not None:
            logger.debug(
                "hook=root_backward_hook wait=comm_fusion_all_reduce module=%s",
                self.hsdp_state,
            )
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_save_grad()
            comm_ctx.all_reduce_param_group = None
        if comm_ctx.pre_param_group is not None:
            logger.debug(
                "hook=root_backward_hook wait=comm_fusion_reduce_scatter module=%s",
                self.hsdp_state,
            )
            comm_ctx.pre_param_group.wait_reduce_scatter_and_issue_all_reduce()
            comm_ctx.pre_param_group = None
        if comm_ctx.all_reduce_param_group is not None:
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_save_grad()
            comm_ctx.all_reduce_param_group = None

    def _finalize_per_param_reductions(self) -> None:
        """Drain the module-tree-local comm_fusion=False RS/AR queues."""
        # A fused root may own non-fused children, so always drain the tree queues.
        last_all_reduce_groups = self.hsdp_state._wait_prev_reduce_scatter()
        self.hsdp_state._wait_prev_reduce_scatter_without_all_reduce()
        self.hsdp_state._issue_prev_fused_all_reduce(last_all_reduce_groups)
        self.hsdp_state.wait_and_split_all_reduce_work_groups()

    def launch_tp_replicate_reduce_and_apply(self) -> None:
        """Run final TP replicate reductions and apply gradients for all states."""
        for hsdp_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            hsdp_state = hsdp_scheduler.hsdp_state
            if hsdp_state is None:
                continue
            need_synchronize = False
            for hsdp_param in hsdp_state.hsdp_params:
                reduced_grad = hsdp_param.all_reduce_comm_ctx.all_reduce_output
                if reduced_grad is None:
                    reduced_grad = hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output
                if reduced_grad is None:
                    continue
                hsdp_param.all_reduce_source_replicate_grad_inplace(
                    reduced_grad,
                    hsdp_state.reduce_op_type,
                )
                need_synchronize = hsdp_param.apply_reduced_grad(reduced_grad) or need_synchronize
                hsdp_param.clear_all_reduce_output()
                hsdp_param.clear_reduce_scatter_output()
            hsdp_state._sync_current_stream_if_needed(need_synchronize)

    @_dynamo_disable
    def _backward_hook(self, force: bool = False):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        if self._param_grad_gate and not force:
            self._param_grad_input_done = True
            if not self._param_grad_expected.issubset(self._param_grad_ready):
                return
            # 本门控可能先于用户 post_accumulate 钩子执行；root force 才能安全消费 .grad。
            if self._has_user_post_accumulate_hooks():
                return
        self._hsdp_backward_hook(self.cell, None, None)

    @staticmethod
    def _register_forward_module_hook(mod, hook) -> None:
        """Register forward hook; use ``always_call=True`` when supported (matches PyTorch FSDP)."""
        sig = inspect.signature(mod.register_forward_hook)
        if "always_call" in sig.parameters:
            mod.register_forward_hook(hook, prepend=False, always_call=True)
        else:
            mod.register_forward_hook(hook, prepend=False)

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook on all managed modules."""
        if self._fsdp_group_post_pending is None:
            for mod in self.modules:
                mod.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
                mod.register_forward_hook(self._forward_hook)
            return
        for mod in self.modules:
            mod.register_forward_pre_hook(self._grouped_forward_pre_hook, with_kwargs=True)
            grouped_forward_hook = _dynamo_disable(self._make_grouped_forward_post_hook(mod))
            self._register_forward_module_hook(mod, grouped_forward_hook)
