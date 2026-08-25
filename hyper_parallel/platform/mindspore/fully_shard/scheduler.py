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
"""MindSpore HSDP scheduler"""
from typing import List
import mindspore as ms
from mindspore._c_expression import _DisableMsDispatchMode
from mindspore.common.api import _pynative_executor
from mindspore.utils._pytree import tree_flatten, tree_unflatten
from hyper_parallel.tools.logging import get_logger
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2, FSDPSchedulerState
from hyper_parallel.platform.mindspore.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2
from hyper_parallel.platform import get_platform

logger = get_logger("FSDP")


class MindSporeHSDPSchedulerV2(HSDPSchedulerV2):
    """MindSpore HSDP scheduler.

    List-unit grouped forward hooks use :class:`HSDPSchedulerV2` defaults for
    ``_grouped_forward_pre_hook_skip`` / ``_grouped_forward_post_hook_skip`` (no overrides here).
    """
    def zero_grad(self) -> None:
        """Zero grad."""
        self.hsdp_state.zero_grad()

    def _register_hooks(self):
        """Register hooks."""
        self._register_forward_backward_hooks()

    def _init_platform(self):
        """Initialize the platform."""
        from hyper_parallel.platform.mindspore.platform import MindSporePlatform
        self.platform = get_platform()
        if not isinstance(self.platform, MindSporePlatform):
            raise ValueError(f"MindSporeHSDPSchedulerV2 expect MindSporePlatform, but got type: {type(self.platform)}")

    def _new_cell_state(self):
        """Create a new cell state for mindspore."""
        self.hsdp_state = MindSporeHSDPStateV2(
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
        )

    def _register_post_backward_hook(self, args, kwargs):
        """Wrap forward args/kwargs through PostBackwardFunction to register backward hook."""
        if not _pynative_executor.enable_grad():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[ms.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if isinstance(obj, ms.Tensor) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        if len(inp_tensors) == 0:
            return args, kwargs  # no tensors that require gradients
        processed_tensors = PostBackwardFunction.apply(self, *inp_tensors)
        for inp_tensor_idx, processed_tensor in zip(inp_tensor_indices, processed_tensors):
            args_kwargs_list[inp_tensor_idx] = processed_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list):]
        args = tree_unflatten(args_spec, args_list)
        kwargs = tree_unflatten(kwargs_spec, kwargs_list)
        return args, kwargs

    def _forward_pre_hook(self, cell, args, kwargs):
        """Execute forward pre hook and set up backward hook."""
        args, kwargs = self._hsdp_forward_pre_hook(cell, args, kwargs)
        return self._register_post_backward_hook(args, kwargs)

    def _register_backward_pre_hook(self, outputs):
        """Register gradient hooks on outputs to trigger backward pre hook."""
        flat_outputs, _ = tree_flatten(outputs)
        for output in flat_outputs:
            if isinstance(output, ms.Tensor) and output._requires_grad:
                # Removing a MindSpore tensor hook from its own callback corrupts autograd callback traversal.
                # The output tensor owns this hook for the lifetime of its graph, so no separate cleanup is needed.
                output.register_hook(self._backward_pre_hook)
        return outputs

    def _forward_hook(self, cell, inputs, outputs):
        """Execute forward hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self._register_backward_pre_hook(outputs)
        if self.scheduler_ctx.root_bp_state:
            self._restore_forward_prefetch_after_recompute()
            return
        return self._hsdp_forward_hook(cell, inputs, outputs)

    # pylint: disable=W0212
    def _backward_pre_hook(self, grad):
        """Execute backward pre hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        if self._is_root:
            _pynative_executor.queue_backward_final_callback(self._root_backward_hook)
        self.scheduler_ctx.root_bp_state = True
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    def _root_backward_hook(self):
        """Drain all DP pipelines, then run final TP reduction and apply gradients."""
        logger.debug("hook=root_backward_hook enter module=%s", self.hsdp_state)
        for hsdp_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            hsdp_scheduler._backward_hook()
        self.scheduler_ctx.root_bp_state = False
        logger.debug(
            "hook=root_backward_hook action=final_reduce module=%s",
            self.hsdp_state,
        )
        self._finalize_comm_fusion_reductions()
        self._finalize_per_param_reductions()
        self.launch_tp_replicate_reduce_and_apply()

    def _finalize_comm_fusion_reductions(self) -> None:
        """Drain the comm_fusion=True reduce-scatter/all-reduce pipeline."""
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
        """Drain the module-tree-local comm_fusion=False communication queues."""
        last_all_reduce_groups = self.hsdp_state._wait_prev_reduce_scatter()
        self.hsdp_state._wait_prev_reduce_scatter_without_all_reduce()
        self.hsdp_state._issue_prev_fused_all_reduce(last_all_reduce_groups)
        self.hsdp_state.wait_and_split_all_reduce_work_groups()

    def launch_tp_replicate_reduce_and_apply(self) -> None:
        """Run final source-layout reductions and apply gradients for all states."""
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

    def reset_iter_state(self) -> None:
        """Reset MindSpore fully_shard iteration state after communication completes."""
        super().reset_iter_state()
        self.hsdp_state.reset_iter_state()
        comm_ctx = self.scheduler_ctx.param_group_comm_ctx
        comm_ctx.pre_param_group = None
        comm_ctx.all_reduce_param_group = None

    def _backward_hook(self):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        self._hsdp_backward_hook(self.cell, None, None)

    @staticmethod
    def _without_ms_dispatch_mode(hook):
        """Run HSDP hook internals outside any outer MsDispatchMode."""
        def wrapped_hook(*args, **kwargs):
            with _DisableMsDispatchMode():
                return hook(*args, **kwargs)
        return wrapped_hook

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook on all managed modules."""
        if self._fsdp_group_post_pending is None:
            for mod in self.modules:
                mod.register_forward_pre_hook(
                    self._without_ms_dispatch_mode(self._forward_pre_hook),
                    with_kwargs=True,
                )
                mod.register_forward_hook(self._without_ms_dispatch_mode(self._forward_hook))
            return
        for mod in self.modules:
            mod.register_forward_pre_hook(
                self._without_ms_dispatch_mode(self._grouped_forward_pre_hook),
                with_kwargs=True,
            )
            mod.register_forward_hook(
                self._without_ms_dispatch_mode(self._make_grouped_forward_post_hook(mod))
            )
