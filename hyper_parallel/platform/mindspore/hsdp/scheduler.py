# Copyright 2025 Huawei Technologies Co., Ltd
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
import warnings
from pathlib import Path
from importlib import resources
import mindspore as ms
from mindspore import ops
from mindspore import jit_class, nn
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel
from hyper_parallel.core.hsdp.hsdp_scheduler import HSDPScheduler
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.mindspore.platform_graph import MindSporeGraphPlatform
from hyper_parallel.platform.mindspore.hsdp.state import MindSporeHSDPState
from hyper_parallel.platform.mindspore.hsdp.grad_hook import MindSporeHSDPGradHook
from hyper_parallel.platform.mindspore.hsdp.async_grad_hook import MindSporeHSDPAsyncGradHook


@jit_class
class MindSporeHSDPScheduler(HSDPScheduler):
    """MindSporeHSDPScheduler is used to implement optimizer level."""
    HYPER_PARALLEL_MINDSPORE_SO = "libhyper_parallel_mindspore.so"

    def _init_platform(self):
        """Initialize the platform."""
        if self.config.use_eager_hook:
            self.platform = get_platform()
        else:
            self.platform = MindSporeGraphPlatform()

    def _new_cell_state(self):
        """Create a new cell state."""
        # TODO: why reset use_eager_hook here?
        # self.config.use_eager_hook = ms.get_context("mode") != ms.GRAPH_MODE
        self.hsdp_state = MindSporeHSDPState(self.cell, self.config, self.platform)

    def _new_grad_hook(self):
        """Create a new grad hook."""
        if self.config.use_eager_hook and self.config.comm_async:
            self.grad_hook = MindSporeHSDPAsyncGradHook(self.config, self.platform)
        else:
            self.grad_hook = MindSporeHSDPGradHook(self.config, self.platform)

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook."""
        self.cell.register_forward_pre_hook(self._hsdp_forward_pre_hook)
        self.cell.register_backward_pre_hook(self._hsdp_backward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.cell.register_forward_hook(self._hsdp_forward_hook)
            self.cell.register_backward_hook(self._hsdp_backward_hook)
        elif self.requires_acc_grad:
            self.cell.register_backward_hook(self._hsdp_acc_backward_hook)
        else:
            self.cell.register_backward_hook(self._hsdp_backward_hook)

    def _register_hooks(self):
        """Register hooks."""
        if self.config.use_eager_hook:
            super()._register_hooks()
        else:
            self._register_graph_hook()

    @staticmethod
    def get_pass_library_pass():
        """Safely locate pass library path (compatible with Python 3.8+)"""
        try:
            # Python 3.9+
            if hasattr(resources, "files"):
                return resources.files(
                    "hyper_parallel.platform.mindspore.custom_pass") / \
                    MindSporeHSDPScheduler.HYPER_PARALLEL_MINDSPORE_SO
            # Python 3.8 fallback
            import pkg_resources  # pylint: disable=C0415
            return Path(pkg_resources.resource_filename(
                "hyper_parallel.platform.mindspore.custom_pass",
                MindSporeHSDPScheduler.HYPER_PARALLEL_MINDSPORE_SO
            ))
        except Exception as e:
            warnings.warn(
                f"Failed to locate mindspore custom pass library: {e}")
            return None

    def _register_custom_passes(self):
        """Register custom graph optimization passes to mindspore"""

        so_path = self.get_pass_library_pass()
        if so_path and so_path.exists():
            success = ms.graph.register_custom_pass(
                pass_name="DuplicatePrimOnMultiUsersPass",
                plugin_so_path=str(so_path),
                device="cpu",
                pass_type=ms.graph.CustomPassType.FULL_GRAPH)
            if not success:
                print(f"Failed to register MindSpore custom pass from {so_path}.")
            return success

        print(f"Failed to locate MindSpore custom pass library {so_path}.")
        return False

    def _get_param_forward_hook(self, hsdp_param):
        """Get param forward hook."""
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            # pylint: disable=W0212
            allgather = ops._add_attr(self.platform.all_gather_into_tensor, duplicate_on_multiple_users=True)

            def stateless_param_forward_hook(origin_param):
                output, _ = allgather(origin_param, hsdp_param.sharded_group_info)
                return output

            if not self._register_custom_passes():
                raise RuntimeError(
                    "Mindspore custom pass registration failed but is mandatory for optimizer level "
                    f"{OptimizerLevel.SHARD_OPT_GRAD_PARAM}. "
                    "This optimization level requires graph transformations provided by the custom pass library "
                    f"({self.HYPER_PARALLEL_MINDSPORE_SO}). Ensure MindSpore is installed and the pass library was "
                    "successfully built during package installation."
                )

            return stateless_param_forward_hook

        def stateful_param_forward_hook(origin_param):
            unshared_data, _ = self.platform.all_gather_into_tensor(origin_param, hsdp_param.sharded_group_info)
            return unshared_data
        return stateful_param_forward_hook

    def _get_param_backward_hook(self, hsdp_param):
        """Get hook for param backward process."""
        grad_hook = self.grad_hook.get_hook(hsdp_param)
        def backward_hook(grad):
            return grad_hook(grad)

        def backward_acc_grad_hook(grad):
            return grad_hook(grad)

        if self.requires_acc_grad:
            return backward_acc_grad_hook
        return backward_hook


    def _get_parameter_forward_hook(self, hsdp_forward_hook, hsdp_grad_hook):
        """
        Get parameter forward hook according to the hsdp_forward_hook and hsdp_grad_hook.
        """
        class ForwardHookNet(nn.Cell):
            def __init__(self, hsdp_forward_hook) -> None:
                super().__init__()
                self.hsdp_forward_hook = hsdp_forward_hook
            def construct(self, param):
                return self.hsdp_forward_hook(param)
            def bprop(self, param, out, dout):  # pylint: disable=W0613
                return (dout,)

        fwd_hook_net = ForwardHookNet(hsdp_forward_hook)
        insert_grad_of = ops.InsertGradientOf(hsdp_grad_hook)

        def parameter_forward_hook(param):
            return insert_grad_of(fwd_hook_net(param))
        return parameter_forward_hook


    def _register_graph_hook(self):
        """Register param forward and grad hook."""
        params_hooks = []
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.sharded:
                hsdp_param.param.register_hook(self.grad_hook.get_hook(hsdp_param))
            else:
                param_fwd_hook = self._get_parameter_forward_hook(
                    self._get_param_forward_hook(hsdp_param), self._get_param_backward_hook(hsdp_param))
                params_hooks.append(
                    {"params": [hsdp_param.param], "hook": param_fwd_hook}
                )
        self.cell.register_parameter_forward_hook(params_hooks)

    def _get_grad_buffer_hook(self, hsdp_param):
        """Set grad for hsdp parameter."""
        origin_hook = super()._get_grad_buffer_hook(hsdp_param)
        def set_grad_hook(grad):
            grad = origin_hook(grad)
            hsdp_param.param.grad = grad
            return grad
        return set_grad_hook
