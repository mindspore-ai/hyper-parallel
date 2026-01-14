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
"""Torch HSDP scheduler"""
from torch.autograd import Variable
from torch.utils._pytree import tree_flatten, tree_unflatten
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel
from hyper_parallel.core.hsdp.hsdp_scheduler import HSDPScheduler, FSDPSchedulerState
from hyper_parallel.platform.torch.hsdp.hook_function import PostBackwardFunction
from hyper_parallel.platform.torch.hsdp.state import TorchHSDPState
from hyper_parallel.platform.torch.hsdp.grad_hook import TorchHSDPGradHook
from hyper_parallel.platform.torch.hsdp.async_grad_hook import TorchHSDPAsyncGradHook
from hyper_parallel.platform import get_platform


class TorchHSDPScheduler(HSDPScheduler):
    """TorchHSDPScheduler is used to implement optimizer level."""

    def _init_platform(self):
        """Initialize the platform."""
        self.platform = get_platform()

    def _new_cell_state(self):
        """Create a new cell state for torch."""
        self.hsdp_state = TorchHSDPState(self.cell, self.config, self.platform)

    def _new_grad_hook(self):
        """
        Create and initialize a new TorchHSDPGradHook instance.
        
        This method instantiates the gradient hook component used for handling
        gradient operations in the HSDP scheduler.
        """
        if self.config.comm_async:
            self.grad_hook = TorchHSDPAsyncGradHook(self.config, self.platform)
        else:
            self.grad_hook = TorchHSDPGradHook(self.config, self.platform)

    def _forward_pre_hook(self, cell, args, kwargs):
        """Execute forward pre hook and set up backward hook."""
        self._hsdp_forward_pre_hook(cell, args)
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        args_kwargs_list = PostBackwardFunction.apply(self, *args_kwargs_list)
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs

    # pylint: disable=W0212
    def _backward_pre_hook(self, cell, grad_outputs):
        """Execute backward pre hook."""
        self._hsdp_backward_pre_hook(cell, grad_outputs)
        Variable._execution_engine.queue_callback(self._backward_hook)

    def _backward_hook(self):
        """Execute backward hook."""
        if self.scheduler_state == FSDPSchedulerState.BACKWARD:
            return
        if self.requires_acc_grad and self.shard_level != OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self._hsdp_acc_backward_hook(self.cell, None, None)
        else:
            self._hsdp_backward_hook(self.cell, None, None)

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook."""
        self.cell.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
        self.cell.register_full_backward_pre_hook(self._backward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.cell.register_forward_hook(self._hsdp_forward_hook)
