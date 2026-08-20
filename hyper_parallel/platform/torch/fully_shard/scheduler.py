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
"""Torch HSDP scheduler"""
import functools
import inspect
from typing import Callable, List, ParamSpec, TypeVar

import torch
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.platform import get_platform
from hyper_parallel.tools.logging import get_logger
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2, FSDPSchedulerState
from hyper_parallel.platform.torch.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2

logger = get_logger("FSDP")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _dynamo_disable(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Disable Dynamo tracing while an FSDP runtime hook executes."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return torch._dynamo.disable(
            func,
            recursive=True,
            reason="skipping HyperParallel FSDP hooks",
        )(*args, **kwargs)

    return wrapper


class TorchHSDPSchedulerV2(HSDPSchedulerV2):
    """TorchHSDPScheduler is used to implement optimizer level."""

    def __init__(self, *args, **kwargs):
        """Initialize TorchHSDPSchedulerV2 and register forward/backward hooks."""
        super().__init__(*args, **kwargs)

    def _register_hooks(self):
        """Register hooks."""
        self._register_forward_backward_hooks()

    def _init_platform(self):
        """Initialize the platform."""
        # pylint: disable=C0415
        from hyper_parallel.platform.torch.platform import TorchPlatform
        self.platform = get_platform()
        if not isinstance(self.platform, TorchPlatform):
            raise ValueError(f"TorchHSDPSchedulerV2 expect TorchPlatform, but got type: {type(self.platform)}")

    def _new_cell_state(self):
        """Create a new cell state for torch."""
        self.hsdp_state = TorchHSDPStateV2(
            self.modules,
            self.mesh,
            self.shard_placement_fn,
            self.comm_fusion_policy,
            self.mp_policy,
            self.offload_policy,
            self.ignored_params,
            self.replicate_params,
            self.platform,
            self.scheduler_ctx,
            self.device,
            source_shard_infos=self.source_shard_infos,
        )

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

    def _register_backward_pre_hook(self, outputs):
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
        return outputs

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
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    @_dynamo_disable
    def _root_backward_hook(self):
        """Drain all DP pipelines, then run final TP reduction and apply gradients."""
        logger.debug("hook=root_backward_hook enter module=%s", self.hsdp_state)
        for hsdp_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            # let modules which are not triggered backward_hook launch backward communication.
            hsdp_scheduler._backward_hook()
        self.scheduler_ctx.root_bp_state = False
        with torch.profiler.record_function(f"root_backward reduce:{self.hsdp_state.module_name}"):
            logger.debug(
                "hook=root_backward_hook action=final_reduce module=%s",
                self.hsdp_state,
            )
            self._finalize_comm_fusion_reductions()
            self._finalize_per_param_reductions()
            self.launch_tp_replicate_reduce_and_apply()

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
    def reset_iter_state(self) -> None:
        """Reset Torch fully_shard iteration state after communication is complete."""
        super().reset_iter_state()
        self.hsdp_state.reset_iter_state()
        comm_ctx = self.scheduler_ctx.param_group_comm_ctx
        comm_ctx.pre_param_group = None
        comm_ctx.all_reduce_param_group = None

    @_dynamo_disable
    def _backward_hook(self):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        self._hsdp_backward_hook(self.cell, None, None)

    # pylint: disable=W0613
    @staticmethod
    def _grouped_forward_pre_hook_skip(cell, args, kwargs) -> None:
        """Override base ``(args, kwargs)`` return; ``nn.Module`` pre-hook uses ``None`` for no-op."""
        return None

    @staticmethod
    def _grouped_forward_post_hook_skip(outputs) -> None:
        """Override base output pass-through; forward hook uses ``None`` for no-op."""
        return None

    @_dynamo_disable
    def _grouped_forward_pre_hook(self, cell, args, kwargs):
        """Run the grouped FSDP pre-forward hook outside Dynamo tracing."""
        return super()._grouped_forward_pre_hook(cell, args, kwargs)

    def _register_forward_module_hook(self, mod, hook) -> None:
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
