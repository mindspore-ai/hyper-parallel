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
from mindspore.common.api import _pynative_executor
from mindspore.utils._pytree import tree_flatten, tree_unflatten
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2, FSDPSchedulerState
from hyper_parallel.platform.mindspore.fully_shard.hook_function import PostBackwardFunction
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, HSDPMeshInfo
from hyper_parallel.platform import get_platform


class MindSporeHSDPSchedulerV2(HSDPSchedulerV2):
    """MindSporeHSDPScheduler is used to implement optimizer level."""
    def zero_grad(self):
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
        if self.mesh.ndim not in (1, 2):
            raise ValueError("fully_shard only support 1D and 2D mesh.")
        if self.mesh.ndim == 1:
            # FSDP2
            self.mesh_info = FSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=0)
        else:
            # HSDP
            self.mesh_info = HSDPMeshInfo(mesh=self.mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        self.hsdp_state = MindSporeHSDPStateV2(self.cell, self.mesh_info, self.config, self.platform, self.device)

    def _register_post_backward_hook(self, args, kwargs):
        """Register backward hook using backward function."""
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        args_kwargs_list = PostBackwardFunction.apply(self, *args_kwargs_list)
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
        self._register_backward_pre_hook(outputs)
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self._hsdp_forward_hook(cell, inputs, outputs)
        return outputs

    # pylint: disable=W0212
    def _backward_pre_hook(self, grad):
        """Execute backward pre hook."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return grad
        self._hsdp_backward_pre_hook(self.cell, None)
        _pynative_executor.queue_backward_final_callback(self._backward_hook)
        return grad

    def _backward_hook(self):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        self._hsdp_backward_hook(self.cell, None, None)

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook on all managed modules."""
        for mod in self.modules:
            mod.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
            mod.register_forward_hook(self._forward_hook)
