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
from hyper_parallel.core.hsdp.hsdp_utils import OptimizerLevel, HSDPConfig
from hyper_parallel.core.hsdp.hsdp_state import HSDPState
from hyper_parallel.core.hsdp.hsdp_grad_hook import HSDPGradHook
from hyper_parallel.core.hsdp.hsdp_async_grad_hook import HSDPAsyncGradHook


class HSDPScheduler:
    """HSDPScheduler is used to implement optimizer level."""

    def __init__(self, cell, shard_size, threshold, shard_level, requires_acc_grad, grad_scale,
                 reduce_dtype, comm_async, comm_fusion, bucket_size):
        """init hsdp scheduler."""
        self.cell = cell
        self.no_param_sharded = shard_size == 1
        self.shard_level = shard_level
        self.requires_acc_grad = requires_acc_grad
        self.requires_grad_sync = False

        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []
        self.config = HSDPConfig(
            shard_size,
            threshold,
            requires_acc_grad,
            grad_scale,
            shard_level,
            True,
            reduce_dtype,
            comm_async,
            comm_fusion,
            bucket_size
        )
        self._init_platform()
        self._new_cell_state()
        self._new_grad_hook()
        self._register_hooks()

    def _init_platform(self):
        """init platform"""
        pass

    def _new_cell_state(self):
        """new cell state"""
        pass
    
    def _new_grad_hook(self):
        """new grad hook"""
        if self.config.comm_async:
            self.grad_hook = HSDPAsyncGradHook(self.config, self.platform)
        else:
            self.grad_hook = HSDPGradHook(self.config, self.platform)

    def _register_hooks(self):
        """register hooks"""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.param.requires_grad:
                continue
            if self.config.grad_fusion:
                hsdp_param.param.register_hook(self._get_grad_buffer_hook(hsdp_param))
            else:
                hsdp_param.param.register_hook(self.grad_hook.get_hook(hsdp_param))

        if self.no_param_sharded:
            return

        self.platform.register_forward_pre_hook(self.cell, self._hsdp_forward_pre_hook)
        self.platform.register_backward_pre_hook(self.cell, self._hsdp_backward_pre_hook)
        if self.shard_level == OptimizerLevel.SHARD_OPT_GRAD_PARAM:
            self.platform.register_forward_hook(self.cell, self._hsdp_forward_hook)
            self.platform.register_backward_hook(self.cell, self._hsdp_backward_hook)
        elif self.requires_acc_grad:
            self.platform.register_backward_hook(self.cell, self._hsdp_acc_backward_hook)
        else:
            self.platform.register_backward_hook(self.cell, self._hsdp_backward_hook)

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        self.requires_grad_sync = requires_grad_sync
        self.grad_hook.set_requires_grad_sync(requires_grad_sync)
        self.hsdp_state.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """set gradient to zero."""
        if self.requires_acc_grad:
            self.hsdp_state.zero_grads()

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
        """set grad ready."""
        def hook(grad):
            hsdp_param.grad = grad
            self.hsdp_state.set_grad_ready(hsdp_param)
            return grad
        return hook

    def set_forward_prefetch_cells(self, hsdp_cell_list):
        """set forward prefetch cells."""
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list):
        """set backward prefetch cells."""
        self.backward_prefetch_cells = hsdp_cell_list
