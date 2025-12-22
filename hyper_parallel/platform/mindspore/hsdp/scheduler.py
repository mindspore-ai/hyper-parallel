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
import mindspore as ms
from mindspore import ops
from mindspore.common.tensor import Tensor
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel
from hyper_parallel.core.hsdp.hsdp_scheduler import HSDPScheduler
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.mindspore.platform_graph import MindSporeGraphPlatform
from hyper_parallel.platform.mindspore.hsdp.state import MindSporeHSDPState
from hyper_parallel.platform.mindspore.hsdp.grad_hook import MindSporeHSDPGradHook
from hyper_parallel.platform.mindspore.hsdp.async_grad_hook import MindSporeHSDPAsyncGradHook


class MindSporeHSDPScheduler(HSDPScheduler):
    """MindSporeHSDPScheduler is used to implement optimizer level."""

    def _init_platform(self):
        """init platform"""
        if self.config.use_eager_hook:
            self.platform = get_platform()
        else:
            self.platform = MindSporeGraphPlatform()

    def _new_cell_state(self):
        """new cell state"""
        self.config.use_eager_hook = ms.get_context("mode") != ms.GRAPH_MODE
        self.hsdp_state = MindSporeHSDPState(self.cell, self.config, self.platform)

    def _new_grad_hook(self):
        if self.config.use_eager_hook and self.config.comm_async:
            self.grad_hook = MindSporeHSDPAsyncGradHook(self.config, self.platform)
        else:
            self.grad_hook = MindSporeHSDPGradHook(self.config, self.platform)

    def _register_hooks(self):
        """register hooks"""
        if self.config.use_eager_hook:
            super()._register_hooks()
        else:
            self._register_graph_hook()

    def _get_param_forward_hook(self, hsdp_param):
        """get param forward hook."""

        def stateless_param_forward_hook(origin_param):
            output, _ = self.platform.all_gather_into_tensor(origin_param, hsdp_param.sharded_group_info)
            return output

        def stateful_param_forward_hook(origin_param):
            if hsdp_param.unsharded_param_available:
                return hsdp_param.unsharded_param

            unshared_data, _ = self.platform.all_gather_into_tensor(origin_param, hsdp_param.sharded_group_info)
            ops.assign(hsdp_param.unsharded_param, unshared_data)
            ops.assign(hsdp_param.unsharded_param_available, Tensor(True))
            return hsdp_param.unsharded_param

        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            return stateless_param_forward_hook
        return stateful_param_forward_hook

    def _get_param_backward_hook(self, hsdp_param):
        """get hook for param backward process."""
        grad_hook = self.grad_hook.get_hook(hsdp_param)
        def backward_hook(grad):
            ops.assign(hsdp_param.unsharded_param_available, Tensor(False))
            return grad_hook(grad)

        def backward_acc_grad_hook(grad):
            if self.grad_hook.requires_grad_sync:
                ops.assign(hsdp_param.unsharded_param_available, Tensor(False))
            return grad_hook(grad)

        if self.requires_acc_grad:
            return backward_acc_grad_hook
        return backward_hook

    def _register_graph_hook(self):
        """register param forward and grad hook."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.sharded:
                hsdp_param.param.register_hook(self.grad_hook.get_hook(hsdp_param))
            else:
                hsdp_param.param.register_hsdp_hook(self._get_param_forward_hook(hsdp_param),
                                                    self._get_param_backward_hook(hsdp_param))

    def _get_grad_buffer_hook(self, hsdp_param):
        """set grad for hsdp parameter."""
        origin_hook = super()._get_grad_buffer_hook(hsdp_param)
        def set_grad_hook(grad):
            grad = origin_hook(grad)
            hsdp_param.param.grad = grad
            return grad
        return set_grad_hook
