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
"""HSDP scheduler"""
import mindspore as ms
from mindspore import ops
from mindspore.common.tensor import Tensor
from dist_parallel.spmd.hsdp.hsdp_utils import OptimizerLevel, HSDPConfig
from dist_parallel.spmd.hsdp.hsdp_state import HSDPState
from dist_parallel.spmd.hsdp.hsdp_grad_hook import HSDPGradHook
from dist_parallel.spmd.hsdp.hsdp_async_grad_hook import HSDPAsyncGradHook
import mindspore.parallel.spmd.hsdp.hsdp_comm as comm


class HSDPScheduler:
    """HSDPScheduler is used to imply optimizer level."""

    def __init__(self, cell, shard_size, threshold, shard_level, requires_acc_grad, grad_scale,
                 reduce_dtype, comm_async, comm_fusion, bucket_size):
        """init hsdp scheduler."""
        self.cell = cell
        self.shard_level = shard_level
        self.no_param_sharded = shard_size == 1
        self.use_pynative_hook = ms.get_context("mode") != ms.GRAPH_MODE
        self.requires_acc_grad = requires_acc_grad
        self.requires_grad_sync = False
        self.reduce_dtype = reduce_dtype
        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []
        self.config = HSDPConfig(
            shard_size,
            threshold,
            requires_acc_grad,
            grad_scale,
            shard_level,
            self.use_pynative_hook,
            reduce_dtype,
            comm_async,
            comm_fusion,
            bucket_size
        )

        self.hsdp_state = HSDPState(cell, self.config)
        if self.use_pynative_hook:
            if comm_async:
                self.grad_hook = HSDPAsyncGradHook(reduce_dtype, grad_scale, shard_level,
                                                   requires_acc_grad, self.use_pynative_hook)
            else:
                self.grad_hook = HSDPGradHook(reduce_dtype, grad_scale, shard_level,
                                              requires_acc_grad, self.use_pynative_hook)
        else:
            self.grad_hook = HSDPGradHook(reduce_dtype, grad_scale, shard_level,
                                          requires_acc_grad, self.use_pynative_hook)

        if self.use_pynative_hook:
            self._register_pynative_hooks()
        else:
            self._register_graph_hook()

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.requires_grad_sync = requires_grad_sync
        self.grad_hook.set_requires_grad_sync(requires_grad_sync)
        self.hsdp_state.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """set gradient to zero."""
        if self.requires_acc_grad:
            self.hsdp_state.zero_grads()

    def _get_param_forward_hook(self, hsdp_param):
        """get param forward hook."""

        def stateless_param_forward_hook(origin_param):
            output, _ = comm.all_gather_into_tensor(origin_param, group=hsdp_param.sharded_group_name)
            return output

        def stateful_param_forward_hook(origin_param):
            if hsdp_param.unsharded_param_available:
                return hsdp_param.unsharded_param

            unshared_data, _ = comm.all_gather_into_tensor(origin_param, group=hsdp_param.sharded_group_name)
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

    def _register_pynative_hooks(self):
        """register cell process hooks."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.param.requires_grad:
                continue
            if self.config.grad_fusion:
                hsdp_param.param.register_hook(self._get_grad_buffer_hook(hsdp_param))
            else:
                hsdp_param.param.register_hook(self.grad_hook.get_hook(hsdp_param))

        if self.no_param_sharded:
            return

        self.cell.register_forward_pre_hook(self._hsdp_forward_pre_hook)
        self.cell.register_backward_pre_hook(self._hsdp_backward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.cell.register_forward_hook(self._hsdp_forward_hook)
            self.cell.register_backward_hook(self._hsdp_backward_hook)
        elif self.requires_acc_grad:
            self.cell.register_backward_hook(self._hsdp_acc_backward_hook)
        else:
            self.cell.register_backward_hook(self._hsdp_backward_hook)

    def _hsdp_forward_pre_hook(self, cell, inputs):
        """forward pre hook to unsharded parameter for forward process."""
        self.hsdp_state.unshard()
        for prefetch_cell in self.forward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.prefetch()

    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """forward hook to shard parameter for saving memory."""
        self.hsdp_state.shard()

    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """backward pre hook to unsharded parameter for backward process."""
        self.hsdp_state.unshard()
        for prefetch_cell in self.backward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.prefetch()

    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for optimizer process or saving memory."""
        self.hsdp_state.shard()

    def _hsdp_acc_backward_hook(self, cell, grad_inputs, grad_outputs):
        """backward hook to shard parameter for grad accumulation when requires_grad_sync is True."""
        if self.requires_grad_sync:
            self.hsdp_state.shard()

    def _get_grad_buffer_hook(self, hsdp_param):
        """set grad for hsdp parameter."""
        def hook(grad):
            hsdp_param.grad = grad
            hsdp_param.param.grad = grad
            self.hsdp_state.set_grad_ready(hsdp_param)
            return grad
        return hook

    def set_forward_prefetch_cells(self, hsdp_cell_list):
        """set forward prefetch cells."""
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list):
        """set backward prefetch cells."""
        self.backward_prefetch_cells = hsdp_cell_list
