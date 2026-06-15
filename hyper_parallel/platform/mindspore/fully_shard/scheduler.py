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
        """Register backward hook using backward function."""
        if not _pynative_executor.enable_grad():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        if not any(
            isinstance(obj, ms.Tensor) and getattr(obj, "requires_grad", False)
            for obj in args_kwargs_list
        ):
            return args, kwargs
        processed_list = list(PostBackwardFunction.apply(self, *args_kwargs_list))
        for idx, (orig_obj, processed_obj) in enumerate(zip(args_kwargs_list, processed_list)):
            if isinstance(orig_obj, ms.Tensor) and isinstance(processed_obj, ms.Tensor):
                try:
                    processed_obj.requires_grad = bool(getattr(orig_obj, "requires_grad", False))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            processed_list[idx] = processed_obj
        args_kwargs_list = processed_list
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
        _pynative_executor.queue_backward_final_callback(self._root_backward_hook)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        HSDPSchedulerV2.root_bp_state = True
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    def _root_backward_hook(self, force_reduce=False):
        """Finalize the outermost backward and clear recompute state.

        ``apply_final_reduce`` selects between two cases by whether this unit's
        forward input was differentiable:

        * input ``requires_grad=False`` (the common case): no ``PostBackwardFunction``
          is inserted, so ``scheduler_state != BACKWARD`` and this hook drains the
          pending fused reductions and applies the per-parameter gradients.
        * input ``requires_grad=True`` (a boundary case where the unit is fed a
          differentiable activation from an enclosing graph): the input's
          ``PostBackwardFunction`` drives ``scheduler_state == BACKWARD`` and the
          natural path does not finalize here. ``force_reduce`` lets a caller that
          owns that backward boundary demand the drain now rather than deferring it.

        ``root_bp_state`` (top-level root backward in flight; gates forward prefetch
        during activation recompute) is independent of the reduce branch and is
        cleared only by the root module's own hook, keyed on ``_is_root``.
        """
        apply_final_reduce = self.scheduler_state != FSDPSchedulerState.BACKWARD
        self._backward_hook()
        if self._is_root:
            HSDPSchedulerV2.root_bp_state = False
        if apply_final_reduce or force_reduce:
            comm_ctx = get_comm_ctx()
            if comm_ctx.all_reduce_param_group is not None:
                comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
                comm_ctx.all_reduce_param_group = None
            if comm_ctx.pre_param_group is not None:
                comm_ctx.pre_param_group.apply_fusion_reduced_grad()
                comm_ctx.pre_param_group = None
            self.hsdp_state.reduce_params()
            self.hsdp_state._finish_ignored_allreduce()

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
