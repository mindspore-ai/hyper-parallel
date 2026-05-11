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
import inspect
import torch
from typing import List
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_unflatten
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2, FSDPSchedulerState
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, DDPMeshInfo, HSDPMeshInfo
from hyper_parallel.platform.torch.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2
from hyper_parallel.platform.torch.fully_shard.param_group import get_comm_ctx
from hyper_parallel.platform import get_platform


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
        params = self._get_managed_params()
        if self.mesh is None:
            compat_meshes = [
                param.device_mesh for param in params if isinstance(param, DTensor)
            ]
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
        self.hsdp_state = TorchHSDPStateV2(
            self.modules, self.mesh_info, self.config, self.platform, self.device
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

    def _forward_pre_hook(self, cell, args, kwargs):
        """Execute forward pre hook and set up backward hook."""
        args, kwargs = self._hsdp_forward_pre_hook(cell, args, kwargs)
        return self._register_post_backward_hook(args, kwargs)

    def _register_backward_pre_hook(self, outputs):
        """Register gradient hooks on all requires-grad outputs to trigger backward pre hook."""
        flat_outputs, _ = tree_flatten(outputs)
        for output in flat_outputs:
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(self._backward_pre_hook)
        return outputs

    def _forward_hook(self, cell, inputs, outputs):  # pylint: disable=R1710
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
        Variable._execution_engine.queue_callback(self._root_backward_hook)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        HSDPSchedulerV2.root_bp_state = True
        self._hsdp_backward_pre_hook(self.cell, None)
        return grad

    def _root_backward_hook(self):
        """Root backward hook: finalize gradient reduction for the outermost HSDP module.

        For the root module (the last to finish backward), this hook drains any
        pending fused reduction from ``CommContext`` and then calls ``reduce_params()``
        to apply the final per-parameter gradient reduction.

        For comm_fusion=False mode, it also:
        1. Processes the last module's reduce_scatter and issues its allreduce
        2. Calls delay_apply_reduce_grads to wait all allreduce and apply gradients
        """
        apply_final_reduce = self.scheduler_state != FSDPSchedulerState.BACKWARD
        self._backward_hook()
        if apply_final_reduce:
            HSDPSchedulerV2.root_bp_state = False
            with torch.profiler.record_function(f"root_backward reduce:{self.hsdp_state.module_name}"):
                # Drain any pending async fused reduction from the last module's backward
                comm_ctx = get_comm_ctx()
                # Drain any pending pipelined HSDP reductions (comm_fusion=True)
                if comm_ctx.all_reduce_param_group is not None:
                    comm_ctx.all_reduce_param_group.wait_all_reduce_and_apply_grad()
                    comm_ctx.all_reduce_param_group = None
                if comm_ctx.pre_param_group is not None:
                    comm_ctx.pre_param_group.apply_fusion_reduced_grad()
                    comm_ctx.pre_param_group = None

                # Process the last module's reduce_scatter and allreduce (comm_fusion=False)
                if TorchHSDPStateV2.pre_all_reduce_groups:
                    for group in TorchHSDPStateV2.pre_all_reduce_groups:
                        # Wait reduce_scatter
                        for hsdp_param in group.hsdp_params:
                            hsdp_param.reduce_scatter_output()
                            hsdp_param.clear_reduce_scatter_output()
                        # Accumulate existing gradients (from previous mini steps) to fused_buffer
                        # This is for gradient accumulation scenario
                        # where previous mini steps used pre_reduce_scatter_params.
                        # The gradients in sharded_param.grad are reduce_scatter results (not allreduced)
                        group.accumulate_existing_grads_to_buffer()
                        # Issue allreduce
                        group.issue_async_allreduce()
                        TorchHSDPStateV2.pending_all_reduce_groups.append(group)
                    TorchHSDPStateV2.pre_all_reduce_groups.clear()

                # Apply gradients for params without all_reduce needs
                self.hsdp_state.reduce_scattered_params()
                # Finally, wait all allreduce and apply gradients
                TorchHSDPStateV2.delay_apply_reduce_grads(self.hsdp_state.device)

                # Handle user config replicated_param
                self.hsdp_state.reduce_params()


    def _backward_hook(self):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        self._hsdp_backward_hook(self.cell, None, None)

    # pylint: disable=W0613
    def _grouped_forward_pre_hook_skip(self, cell, args, kwargs) -> None:
        """Override base ``(args, kwargs)`` return; ``nn.Module`` pre-hook uses ``None`` for no-op."""
        return None

    def _grouped_forward_post_hook_skip(self, outputs) -> None:
        """Override base output pass-through; forward hook uses ``None`` for no-op."""
        return None

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
            self._register_forward_module_hook(mod, self._make_grouped_forward_post_hook(mod))
