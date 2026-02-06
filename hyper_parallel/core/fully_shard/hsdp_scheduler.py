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
import functools
from hyper_parallel.core.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.hsdp_utils import HSDPConfigV2, FSDPSchedulerState
from hyper_parallel.core.fully_shard.hsdp_grad_hook import HSDPGradHook
from hyper_parallel.core.fully_shard.hsdp_async_grad_hook import HSDPAsyncGradHook


class HSDPSchedulerContext:
    """This has state shared across FSDP states."""

    def __init__(self) -> None:
        self.all_states = []
        self.iter_forward_root = None
        self.post_backward_final_callback_queued: bool = False
        self.is_last_backward: bool = True
        self.post_optim_event = None


class HSDPSchedulerV2:
    """HSDPScheduler is used to scheduler hsdp"""
    def __init__(self, cell, mesh, reshard_after_forward, shard_placement_fn,
                 mp_policy, offload_policy, ignored_params, device):
        """init hsdp scheduler."""
        self.cell = cell
        self.mesh: DeviceMesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.ignored_params = ignored_params
        self.device = device
        self.scheduler_state = None
        self.scheduler_ctx = HSDPSchedulerContext()
        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []
        self.config = HSDPConfigV2(
            mesh,
            reshard_after_forward,
            shard_placement_fn,
            mp_policy,
            offload_policy,
            ignored_params
        )
        self._init_platform()
        self._new_cell_state()
        self._new_grad_hook()
        self._register_hooks()

    def _init_platform(self):
        """Initialize the platform."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _init_platform")

    def _new_cell_state(self):
        """Create a new cell state."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _new_cell_state")

    def _new_grad_hook(self):
        """Create a new grad hook."""
        if self.config.comm_async:
            self.grad_hook = HSDPAsyncGradHook(self.config, self.platform)
        else:
            self.grad_hook = HSDPGradHook(self.config, self.platform)

    def _register_hooks(self):
        """Register hooks."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _register_hooks.")

    def _register_grad_hook(self):
        """Register parameter grad hook."""
        for hsdp_param in self.hsdp_state.hsdp_params:
            if not hsdp_param.param.requires_grad:
                continue
            if self.config.grad_fusion:
                hsdp_param.param.register_hook(self._get_grad_buffer_hook(hsdp_param))
            else:
                hsdp_param.param.register_hook(self.grad_hook.get_hook(hsdp_param))

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _register_forward_backward_hooks.")

    def set_reshard_after_forward(self, reshard_after_forward: bool):
        """set reshard_after_forward flag"""
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}")
        self.reshard_after_forward = reshard_after_forward
        self.config.reshard_after_forward = reshard_after_forward

    def set_reshard_after_backward(self, reshard_after_backward: bool):
        """set reshard_after_backward flag"""
        if not isinstance(reshard_after_backward, bool):
            raise ValueError(f"reshard_after_backward should be a bool, got {type(reshard_after_backward)}")
        if self.hsdp_state is not None:
            self.hsdp_state.reshard_after_backward = reshard_after_backward

    def set_requires_all_reduce(self, requires_all_reduce: bool):
        """set requires_all_reduce flag"""
        if not isinstance(requires_all_reduce, bool):
            raise ValueError(f"requires_all_reduce should be a bool, got {type(requires_all_reduce)}")
        if self.hsdp_state is not None:
            self.hsdp_state.all_reduce_grads = requires_all_reduce

    def set_requires_grad_sync(self, requires_grad_sync: bool):
        """Set requires grad sync flag to control gradient sync."""
        if not isinstance(requires_grad_sync, bool):
            raise ValueError(f"requires_grad_sync should be a bool, got {type(requires_grad_sync)}")
        self.requires_grad_sync = requires_grad_sync
        self.hsdp_state.set_requires_grad_sync(requires_grad_sync)

    def zero_grads(self):
        """Set gradient to zero."""
        if self.requires_acc_grad:
            self.hsdp_state.zero_grads()

    # pylint: disable=W0613
    def _hsdp_forward_pre_hook(self, cell, inputs):
        """Forward pre hook to unsharded parameter for forward process."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        if len(inputs) > 0:
            self.platform.set_tensor_requires_grad(inputs[0])
        if self.mp_policy.cast_forward_inputs and self.mp_policy.param_dtype:
            cast_fn = functools.partial(self.platform.cast_fp_tensor, self.mp_policy.param_dtype)
            inputs = self.platform.apply_to_tensors(cast_fn, inputs)
        self.hsdp_state.unshard()
        for prefetch_cell in self.forward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.prefetch()
        return inputs

    # pylint: disable=W0613
    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """Forward hook to shard parameter for saving memory."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self.scheduler_state = FSDPSchedulerState.FORWARD
        if self.reshard_after_forward:
            self.hsdp_state.shard()
        if self.mp_policy.output_dtype is not None:
            outputs = self.platform.apply_to_tensors(
                functools.partial(self.platform.cast_fp_tensor, self.mp_policy.output_dtype),
                outputs,
            )
        return outputs

    # pylint: disable=W0613
    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """Backward pre hook to unsharded parameter for backward process."""
        self.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        if self.reshard_after_forward:
            self.hsdp_state.unshard()
        for prefetch_cell in self.backward_prefetch_cells:
            prefetch_cell.hsdp_scheduler.hsdp_state.prefetch()

    # pylint: disable=W0613
    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """Backward hook to shard parameter for optimizer process or saving memory."""
        self.scheduler_state = FSDPSchedulerState.BACKWARD
        self.hsdp_state.post_backward()


    def _get_grad_buffer_hook(self, hsdp_param):
        """Set grad ready."""

        def hook(grad):
            hsdp_param.grad = grad
            self.hsdp_state.set_grad_ready(hsdp_param)
            return grad

        return hook

    def set_forward_prefetch_cells(self, hsdp_cell_list):
        """Set forward prefetch cells."""
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list):
        """Set backward prefetch cells."""
        self.backward_prefetch_cells = hsdp_cell_list

    def set_requires_allreuce(self, requires_all_reduce):
        """set_require_allreuce for HSDP"""
        self.hsdp_state.requires_all_reduce = requires_all_reduce

    def reshard(self,):
        """Reshard parameters after forward or backward."""
        self.hsdp_state.reshard()
