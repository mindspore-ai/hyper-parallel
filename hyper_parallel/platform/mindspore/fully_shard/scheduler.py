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
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2, FSDPSchedulerState
from hyper_parallel.core.fully_shard.hsdp_utils import get_dtensor_managed_mesh
from hyper_parallel.platform.mindspore.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.platform.mindspore.fully_shard.param_group import get_comm_ctx
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, HSDPMeshInfo, DDPMeshInfo
from hyper_parallel.platform import get_platform


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
        params = self._get_managed_params()
        if self.mesh is None:
            compat_meshes = [get_dtensor_managed_mesh(param) for param in params]
            compat_meshes = [mesh for mesh in compat_meshes if mesh is not None]
            compat_mesh = compat_meshes[0] if compat_meshes else None
            if compat_mesh is None:
                raise ValueError(
                    "Cannot build fully_shard compatibility mesh_info "
                    "without a DTensor parameter mesh."
                )
            compat_mesh_hash = compat_mesh.to_hash()
            for param_mesh in compat_meshes[1:]:
                if param_mesh.to_hash() != compat_mesh_hash:
                    raise ValueError(
                        "fully_shard compatibility mode requires all DTensor parameters to share the same mesh."
                    )
            self.mesh_info = DDPMeshInfo(mesh=compat_mesh, replicate_mesh_dim=0)
        elif self.mesh.ndim == 1:
            self.mesh_info = FSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=0)
        elif self.mesh.ndim == 2:
            self.mesh_info = HSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        else:
            raise ValueError(
                "fully_shard only supports explicit 1D DP/FSDP meshes or 2D HSDP meshes. "
                f"Got mesh.ndim={self.mesh.ndim}."
            )
        self.hsdp_state = MindSporeHSDPStateV2(
            self.modules, self.mesh_info, self.config, self.platform, self.device
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
        """Register output hook to trigger backward pre hook."""
        flat_outputs, _ = tree_flatten(outputs)
        for output in flat_outputs:
            if isinstance(output, ms.Tensor) and output._requires_grad:
                output.register_hook(self._backward_pre_hook)
        return outputs

    def _forward_hook(self, cell, inputs, outputs):
        """Execute forward hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self._register_backward_pre_hook(outputs)
        if HSDPSchedulerV2.root_bp_state:
            self._restore_forward_prefetch_after_recompute()
            return
        return self._hsdp_forward_hook(cell, inputs, outputs)

    # pylint: disable=W0212
    def _backward_pre_hook(self, grad):
        """Execute backward pre hook."""
        ctx = self.scheduler_ctx
        ctx.post_backward_schedulers[self] = None
        # The first entry owns finalization in the enclosing backward task. A reentrant
        # checkpoint's inner FSDP units may register callbacks on a nested task; those
        # callbacks must still finish local gradients without draining the outer queue.
        callback = self._backward_hook
        if not ctx.post_backward_final_callback_queued:
            ctx.post_backward_final_callback_queued = True
            callback = self._root_backward_hook
        _pynative_executor.queue_backward_final_callback(callback)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        HSDPSchedulerV2.root_bp_state = True
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    # pylint: disable=W0613
    def _root_backward_hook(self, force_reduce=False):
        """Finalize participating units, then drain once in their enclosing backward.

        Reentrant inner-task callbacks only call ``_backward_hook``. Non-reentrant
        recomputation can instead defer a unit's local final callback until after this
        one. Finish every participating unit first, including units whose inputs did
        not require gradients and therefore have no ``PostBackwardFunction``.

        Callback ownership is independent of ``_is_root`` and ``scheduler_state``:
        an unwrapped model may expose several FSDP roots with differentiable inputs.
        Such roots still need their terminal reduction even after local post-backward.
        Pipeline callers retain the explicit ``wait_for_pending_reductions`` entry.
        """
        ctx = self.scheduler_ctx
        schedulers = tuple(ctx.post_backward_schedulers) or (self,)
        try:
            for scheduler in schedulers:
                scheduler._backward_hook()
            if any(scheduler._is_root for scheduler in schedulers):
                HSDPSchedulerV2.root_bp_state = False
            self.wait_for_pending_reductions()
        finally:
            ctx.post_backward_schedulers.clear()
            ctx.post_backward_final_callback_queued = False

    def wait_for_pending_reductions(self) -> None:
        """Drain every asynchronous gradient reduction queued by this backend."""
        comm_ctx = get_comm_ctx()
        if comm_ctx.all_reduce_param_group is not None:
            comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
            comm_ctx.all_reduce_param_group = None
        if comm_ctx.pre_param_group is not None:
            comm_ctx.pre_param_group.apply_fusion_reduced_grad()
            comm_ctx.pre_param_group = None
        # Step 1: Wait for previous reduce-scatter groups and get them for all-reduce
        prev_groups = self.hsdp_state._wait_prev_reduce_scatter()
        # Step 2: Accumulate and issue async all-reduce for previous groups
        for group in prev_groups:
            group.accumulate_existing_grads_to_buffer()
            group.issue_async_allreduce()
            MindSporeHSDPStateV2.pending_all_reduce_groups.append(group)
        # Step 3: Wait/apply any remaining reduce-scatter for pure FSDP params
        self.hsdp_state.reduce_scattered_params()
        # Step 4: Wait for pending all-reduce groups and apply grads
        MindSporeHSDPStateV2.delay_apply_reduce_grads()
        # Step 5: Process any remaining all-reduce params (without fusion)
        self.hsdp_state.reduce_params()

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
