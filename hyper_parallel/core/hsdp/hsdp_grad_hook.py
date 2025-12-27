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
"""HSDP gradient hook"""
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel


class HSDPGradHook:
    """HSDP gradient hook"""

    def __init__(self, config, platform):
        """init"""
        self.reduce_dtype = config.reduce_dtype
        self.grad_scale = config.grad_scale
        self.shard_level = config.shard_level
        self.requires_acc_grad = config.requires_acc_grad
        self.use_eager_hook = config.use_eager_hook
        self.requires_grad_sync = False
        self.platform = platform

    def _cast_hook(self, hook, grad):
        """add cast before and after reduce hook"""
        if self.reduce_dtype is None:
            return hook(grad)
        origin_dtype = grad.dtype
        grad_cast = grad.to(self.reduce_dtype)
        output = hook(grad_cast)
        output = output.to(origin_dtype)
        return output

    def _get_final_grad_hook(self, param, grad_hook, no_cast=False):
        """add cast and scale grad"""
        def scale_with_cast_hook(grad):
            output = self._cast_hook(grad_hook, grad)
            if self.grad_scale != 1.0:
                scale_output = output * self.grad_scale
                return scale_output
            return output

        def scale_hook(grad):
            output = grad_hook(grad)
            if self.grad_scale != 1.0:
                scale_output = output * self.grad_scale
                return scale_output
            return output

        if no_cast:
            return scale_hook
        return scale_with_cast_hook

    def _get_hsdp_param_single_node_hook(self, hsdp_param):
        """get hook for unsharded param with single node."""
        def grad_dummy_hook(grad):
            output = grad * self.grad_scale
            return output

        def grad_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            return self._get_final_grad_hook(hsdp_param.param, grad_dummy_hook, no_cast=True)
        return self._get_final_grad_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_unsharded_hook(self, hsdp_param):
        """get hook for unsharded param."""
        def grad_all_reduce_hook(grad):
            output, _ = self.platform.all_reduce(grad, hsdp_param.unsharded_group_info)
            return output

        def grad_acc_all_reduce_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if self.requires_grad_sync:
                output, _ = self.platform.all_reduce(hsdp_param.acc_grad, hsdp_param.unsharded_group_info)
                return output
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_all_reduce_hook
        else:
            grad_hook = grad_acc_all_reduce_hook
        return self._get_final_grad_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_fully_sharded_hook(self, hsdp_param):
        """get hook for fully sharded param."""
        def grad_reduce_scatter_hook(grad):
            output, _ = self.platform.reduce_scatter_tensor(grad, hsdp_param.sharded_group_info)
            return output

        def grad_acc_reduce_scatter_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if self.requires_grad_sync:
                output, _ = self.platform.reduce_scatter_tensor(hsdp_param.acc_grad, hsdp_param.sharded_group_info)
                return output
            return hsdp_param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            output, _ = self.platform.reduce_scatter_tensor(grad, hsdp_param.sharded_group_info)
            hsdp_param.acc_grad.add_(output)
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_reduce_scatter_hook
        elif self.shard_level == OptimizerLevel.SHARD_OPT:
            grad_hook = grad_acc_reduce_scatter_hook
        else:
            grad_hook = grad_reduce_scatter_acc_hook
        return self._get_final_grad_hook(hsdp_param.param, grad_hook)

    def _get_hsdp_param_partial_sharded_hook(self, hsdp_param):
        """get hook for partial sharded param."""
        def grad_reduce_scatter_hook(grad):
            output, _ = self.platform.reduce_scatter_tensor(grad, hsdp_param.sharded_group_info)
            sliced_grad, _ = self.platform.all_reduce(output, hsdp_param.unsharded_group_info)
            return sliced_grad

        def grad_acc_reduce_scatter_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if self.requires_grad_sync:
                output, _ = self.platform.reduce_scatter_tensor(hsdp_param.acc_grad, hsdp_param.sharded_group_info)
                sliced_grad, _ = self.platform.all_reduce(output, hsdp_param.unsharded_group_info)
                return sliced_grad
            return hsdp_param.acc_grad

        def grad_reduce_scatter_acc_hook(grad):
            output, _ = self.platform.reduce_scatter_tensor(grad, hsdp_param.sharded_group_info)
            hsdp_param.acc_grad.add_(output)
            if self.requires_grad_sync:
                output, _ = self.platform.all_reduce(hsdp_param.acc_grad, hsdp_param.unsharded_group_info)
                return output
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            grad_hook = grad_reduce_scatter_hook
        elif self.shard_level == OptimizerLevel.SHARD_OPT:
            grad_hook = grad_acc_reduce_scatter_hook
        else:
            grad_hook = grad_reduce_scatter_acc_hook
        return self._get_final_grad_hook(hsdp_param.param, grad_hook)

    def get_hook(self, hsdp_param):
        """get hook for param gradient process."""
        if not hsdp_param.sharded:
            if hsdp_param.dp_size == 1:
                return self._get_hsdp_param_single_node_hook(hsdp_param)
            return self._get_hsdp_param_unsharded_hook(hsdp_param)

        if hsdp_param.fully_sharded:
            return self._get_hsdp_param_fully_sharded_hook(hsdp_param)

        return self._get_hsdp_param_partial_sharded_hook(hsdp_param)

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.requires_grad_sync = requires_grad_sync
