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
"""HSDP async gradient hook"""
from mindspore import ops
from dist_parallel.spmd.hsdp.hsdp_utils import OptimizerLevel
from dist_parallel.spmd.hsdp.hsdp_context import hsdp_context
from dist_parallel.spmd.hsdp.hsdp_grad_hook import HSDPGradHook
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm


class HSDPAsyncGradHook(HSDPGradHook):
    """HSDP gradient hook with async communication op"""

    def _pre_process(self, grad):
        """process before call async comm op"""
        origin_dtype = None
        if self.reduce_dtype is not None:
            origin_dtype = ops.dtype(grad)
            grad = ops.cast(grad, self.reduce_dtype)
        return origin_dtype, grad

    def _post_process(self, grad, origin_dtype):
        """process after call async comm op"""
        if origin_dtype is not None:
            grad = ops.cast(grad, origin_dtype)

        if self.grad_scale != 1.0:
            return grad * self.grad_scale
        return grad

    def _get_async_hook_handler(self, param, async_hook, post_hook=None):
        """get async process hook"""
        def async_hook_handler(grad):
            param.grad = grad
            origin_dtype, pre_grad = self._pre_process(grad)
            output, handle = async_hook(pre_grad)

            def post_process():
                if post_hook is not None:
                    post_output = post_hook(output)
                else:
                    post_output = output
                post_grad = self._post_process(post_output, origin_dtype)
                grad.data = post_grad

            if handle is None:
                post_process()
            else:
                hsdp_context.set_grad_reduce_handle(handle, post_process)
            return grad
        return async_hook_handler

    def _get_async_param_unsharded_hook(self, hsdp_param):
        """get hook for unsharded param."""
        def grad_all_reduce_hook(grad):
            return comm.all_reduce(grad, group=hsdp_param.unsharded_group_name, async_op=True)

        def grad_acc_all_reduce_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if not self.requires_grad_sync:
                return grad, None
            return comm.all_reduce(hsdp_param.acc_grad, group=hsdp_param.unsharded_group_name, async_op=True)

        if not self.requires_acc_grad:
            grad_hook = grad_all_reduce_hook
        else:
            grad_hook = grad_acc_all_reduce_hook
        return self._get_async_hook_handler(hsdp_param.param, grad_hook)

    def _get_async_param_fully_sharded_hook(self, hsdp_param):
        """get hook for fully sharded param."""
        def grad_reduce_scatter_hook(grad):
            return comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name, async_op=True)

        def grad_acc_reduce_scatter_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if not self.requires_grad_sync:
                return grad, None
            return comm.reduce_scatter_tensor(hsdp_param.acc_grad, group=hsdp_param.sharded_group_name, async_op=True)

        def grad_reduce_scatter_acc_post_hook(output):
            hsdp_param.acc_grad.add_(output)
            return hsdp_param.acc_grad

        if not self.requires_acc_grad:
            return self._get_async_hook_handler(hsdp_param.param, grad_reduce_scatter_hook)

        if self.shard_level == OptimizerLevel.SHARD_OPT:
            return self._get_async_hook_handler(hsdp_param.param, grad_acc_reduce_scatter_hook)
        return self._get_async_hook_handler(hsdp_param.param, grad_reduce_scatter_hook,
                                            grad_reduce_scatter_acc_post_hook)

    def _get_async_param_partial_sharded_hook(self, hsdp_param):
        """get hook for partial sharded param."""
        def grad_reduce_scatter_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name, async_op=False)
            return comm.all_reduce(output, group=hsdp_param.unsharded_group_name, async_op=True)

        def grad_acc_reduce_scatter_hook(grad):
            hsdp_param.acc_grad.add_(grad)
            if not self.requires_grad_sync:
                return grad, None
            output, _ = comm.reduce_scatter_tensor(hsdp_param.acc_grad, group=hsdp_param.sharded_group_name,
                                                   async_op=False)
            return comm.all_reduce(output, group=hsdp_param.unsharded_group_name, async_op=True)

        def grad_reduce_scatter_acc_hook(grad):
            output, _ = comm.reduce_scatter_tensor(grad, group=hsdp_param.sharded_group_name, async_op=False)
            hsdp_param.acc_grad.add_(output)
            if not self.requires_grad_sync:
                return output, None
            return comm.all_reduce(hsdp_param.acc_grad, group=hsdp_param.unsharded_group_name, async_op=True)

        if not self.requires_acc_grad:
            grad_hook = grad_reduce_scatter_hook
        elif self.shard_level == OptimizerLevel.SHARD_OPT:
            grad_hook = grad_acc_reduce_scatter_hook
        else:
            grad_hook = grad_reduce_scatter_acc_hook
        return self._get_async_hook_handler(hsdp_param.param, grad_hook)

    def get_hook(self, hsdp_param):
        """get hook for param gradient process."""
        if not hsdp_param.sharded:
            if hsdp_param.dp_size == 1:
                return self._get_hsdp_param_single_node_hook(hsdp_param)
            return self._get_async_param_unsharded_hook(hsdp_param)

        if hsdp_param.fully_sharded:
            return self._get_async_param_fully_sharded_hook(hsdp_param)

        return self._get_async_param_partial_sharded_hook(hsdp_param)
