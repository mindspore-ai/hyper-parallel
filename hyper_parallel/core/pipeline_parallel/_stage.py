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
"""pipeline stage"""
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)

import hyper_parallel
from hyper_parallel.core.activation_memory.checkpoint import recompute_handle_collector_ctx
from hyper_parallel.core.activation_memory.reentrant_checkpoint import (
    ReentrantCheckpointHandle,
    reentrant_checkpoint_scope,
)
from hyper_parallel.core.fully_shard.api import HSDPModule


class PipelineStageBase:
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule: Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule: torch.nn.Module, stage_index: int, stage_num: int,
                 group: Optional[dist.ProcessGroup] = None, dyn_shape: bool = False,
                 has_backward: bool = True) -> None:
        """Initialize stage autograd state and validate the communication group."""
        self.submodule = submodule
        self.fwd_cache = {}
        self.bwd_cache = {}
        self._meta_cache = []
        self._dyn_shape = dyn_shape
        self._has_backward = has_backward
        self.group = self._check_pp_group(group)
        self.stage_index = stage_index
        self.stage_num = stage_num
        self.fwd_outputs_cache = {}
        self.last_stage_outputs = None  # Initialized in forward_one_chunk()
        self._trainable_params = None
        self._dw_cache = {}
        self._reentrant_frames = {}
        self._has_sharded_params = any(isinstance(module, HSDPModule) for module in submodule.modules())
        if self._has_sharded_params and not isinstance(submodule, HSDPModule):
            raise ValueError("A pipeline stage containing HSDP units must have an HSDPModule root")
        if self._has_sharded_params and not has_backward:
            raise ValueError("HSDP pipeline stages require backward execution")
        self.fsdp_comm_controller = None
        self._after_forward_chunk = None
        self.retain_backward_graph = False

    def _configure_forward(self) -> None:
        """Apply user residency settings while protecting graphs awaiting dw."""
        if self.fsdp_comm_controller is None:
            return
        for scheduler in self.fsdp_comm_controller.stage_schedulers[self.stage_index]:
            reshard_after_forward = (
                self.fsdp_comm_controller.user_settings[scheduler]["reshard_after_forward"]
                and not self.fsdp_comm_controller.dx_microbatches[self.stage_index]
            )
            if self.fsdp_comm_controller.current_settings[scheduler]["reshard_after_forward"] != reshard_after_forward:
                scheduler.set_reshard_after_forward(reshard_after_forward)
                self.fsdp_comm_controller.current_settings[scheduler]["reshard_after_forward"] = reshard_after_forward

    def _configure_backward(self, micro_index: int) -> None:
        """Configure a gradient-producing backward without consuming old communication."""
        if self.fsdp_comm_controller is None or not self._has_sharded_params:
            return
        if micro_index not in range(self.fsdp_comm_controller.micro_batch_num):
            raise ValueError(f"Invalid backward microbatch {micro_index}")
        if micro_index in self.fsdp_comm_controller.completed_microbatches[self.stage_index]:
            raise RuntimeError(f"Stage {self.stage_index} already completed backward for microbatch {micro_index}")
        is_last_backward = (
            len(self.fsdp_comm_controller.completed_microbatches[self.stage_index])
            == self.fsdp_comm_controller.micro_batch_num - 1
        )
        for scheduler in self.fsdp_comm_controller.stage_schedulers[self.stage_index]:
            requires_gradient_sync = (
                is_last_backward
                or self.fsdp_comm_controller.user_settings[scheduler]["requires_gradient_sync"]
            )
            reshard_after_backward = (
                self.fsdp_comm_controller.user_settings[scheduler]["reshard_after_backward"]
                and not (self.fsdp_comm_controller.dx_microbatches[self.stage_index] - {micro_index})
                and scheduler not in self.fsdp_comm_controller.retained_backward_units
            )
            settings = self.fsdp_comm_controller.current_settings[scheduler]
            if settings["requires_gradient_sync"] != requires_gradient_sync:
                scheduler.set_requires_grad_sync(requires_gradient_sync)
                settings["requires_gradient_sync"] = requires_gradient_sync
            if settings["requires_all_reduce"] != is_last_backward:
                scheduler.set_requires_all_reduce(is_last_backward)
                settings["requires_all_reduce"] = is_last_backward
            if settings["reshard_after_backward"] != reshard_after_backward:
                scheduler.set_reshard_after_backward(reshard_after_backward)
                settings["reshard_after_backward"] = reshard_after_backward

    def _configure_input_backward(self, micro_index: int) -> None:
        """Protect split-backward graphs and disable every gradient communication action."""
        if self.fsdp_comm_controller is None or not self._has_sharded_params:
            return
        if micro_index not in range(self.fsdp_comm_controller.micro_batch_num):
            raise ValueError(f"Invalid input-backward microbatch {micro_index}")
        if (
            micro_index in self.fsdp_comm_controller.dx_microbatches[self.stage_index]
            or micro_index in self.fsdp_comm_controller.completed_microbatches[self.stage_index]
        ):
            raise RuntimeError(f"Stage {self.stage_index} already started backward for microbatch {micro_index}")
        self.fsdp_comm_controller.dx_microbatches[self.stage_index].add(micro_index)
        for scheduler in self.fsdp_comm_controller.stage_schedulers[self.stage_index]:
            settings = self.fsdp_comm_controller.current_settings[scheduler]
            if settings["requires_gradient_sync"]:
                scheduler.set_requires_grad_sync(False)
                settings["requires_gradient_sync"] = False
            if settings["requires_all_reduce"]:
                scheduler.set_requires_all_reduce(False)
                settings["requires_all_reduce"] = False
            if settings["reshard_after_backward"]:
                scheduler.set_reshard_after_backward(False)
                settings["reshard_after_backward"] = False

    def _finish_backward(self, micro_index: int) -> None:
        """Consume a different stage's old tail after this stage's full backward returns."""
        if self.fsdp_comm_controller is None or not self._has_sharded_params:
            return
        if (
            self.fsdp_comm_controller.reduction_tail_stage is not None
            and self.fsdp_comm_controller.reduction_tail_stage != self.stage_index
        ):
            self.fsdp_comm_controller.flush_reduce_scatter(self.fsdp_comm_controller.reduction_tail_stage)
        self.fsdp_comm_controller.reduction_tail_stage = self.stage_index
        self.fsdp_comm_controller.completed_microbatches[self.stage_index].add(micro_index)

    def clear_cache(self) -> None:
        """clear cache."""
        for micro_index in tuple(self._reentrant_frames):
            self._release_reentrant(micro_index)
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()
        self._dw_cache.clear()
        self._meta_cache.clear()

    @staticmethod
    def _clear_recv_buffer(recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
            info.buffer = None

    @staticmethod
    def _check_pp_group(group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, dist.ProcessGroup):
            raise TypeError("Argument 'group' must be type of ProcessGroup, but got type of {type(group)}.")
        return group

    @property
    def is_first_stage(self) -> bool:
        """return if is first stage."""
        return self.stage_index == 0

    @property
    def is_last_stage(self) -> bool:
        """return if is last stage."""
        return self.stage_index == self.stage_num - 1

    def forward_one_chunk(self, micro_index: int, args: Optional[tuple] = None,
                          kwargs: Optional[dict] = None) -> Any:
        """Execution a forward function."""
        self._configure_forward()
        if self.is_first_stage:
            composite_args = args
        else:
            if micro_index in self.args_recv_info:
                composite_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
            else:
                raise RuntimeError(f"The exec order is wrong. The corresponding forward calculation \
                                    is executed before the Receive operation. micro is {micro_index}.")
        composite_kwargs = kwargs or {}
        # The scope also catches direct checkpoint() calls inside forward, not
        # only checkpoint wrappers registered as children of the stage module.
        if self._has_sharded_params:
            scope = recompute_handle_collector_ctx()
        else:
            scope = reentrant_checkpoint_scope(self._get_trainable_params)
        with scope as frames:
            try:
                out = self.submodule(*composite_args, **composite_kwargs)
            except BaseException:
                for frame in frames:
                    if isinstance(frame, ReentrantCheckpointHandle):
                        frame.release()
                raise
        reentrant_frames = tuple(frame for frame in frames if isinstance(frame, ReentrantCheckpointHandle))
        if reentrant_frames:
            self._reentrant_frames[micro_index] = reentrant_frames
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_cache[micro_index] = out_tuple
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        if self._after_forward_chunk is not None:
            self._after_forward_chunk(self.stage_index, micro_index)
        return out

    @staticmethod
    def _filter_grad_outputs(fwd_output):
        """Return outputs with ``requires_grad`` (DTensor → local) for autograd."""
        local_output = []
        for each_out in fwd_output:
            local_tensor = each_out.to_local() if isinstance(each_out, hyper_parallel.DTensor) else each_out
            if local_tensor.requires_grad:
                local_output.append(local_tensor)
        return local_output

    def _build_last_stage_sens(self):
        """Sens tensors for the last stage, aligned 1:1 with rg=True outputs."""
        sens_all = self.get_last_stage_sens(self.last_stage_outputs)
        if not isinstance(sens_all, list):
            return sens_all
        outputs_iter = (self.last_stage_outputs
                        if isinstance(self.last_stage_outputs, (list, tuple))
                        else [self.last_stage_outputs])
        sens = []
        for sensitivity, output in zip(sens_all, outputs_iter):
            local_output = output.to_local() if isinstance(output, hyper_parallel.DTensor) else output
            if local_output.requires_grad:
                sens.append(sensitivity)
        return sens

    def _populate_bwd_cache(self, micro_index):
        """Stash rg=True input grads so they align with peer's grad_recv_info."""
        input_grads = [recv_info.buffer.grad
                       for recv_info in self.args_recv_info[micro_index]
                       if recv_info.requires_grad]
        self.bwd_cache[micro_index] = input_grads

    def _get_trainable_params(self):
        """Return the stable parameter tuple used to partition dx and dw."""
        if self._trainable_params is None:
            full_parameters = {}
            sharded_parameters = set()
            if self._has_sharded_params:
                for scheduler in self.submodule.hsdp_scheduler.scheduler_ctx.all_hsdp_schedulers:
                    for hsdp_param in scheduler.hsdp_state.hsdp_params:
                        sharded_parameters.add(hsdp_param.sharded_param)
                        if hsdp_param.sharded_param.requires_grad:
                            full_parameters[id(hsdp_param.unsharded_param)] = hsdp_param.unsharded_param
            for param in self.submodule.parameters():
                if param.requires_grad and param not in sharded_parameters:
                    full_parameters[id(param)] = param
            self._trainable_params = tuple(full_parameters.values())
        return self._trainable_params

    def _release_reentrant(self, micro_index):
        """Discard per-microbatch replay graphs even after a failed backward."""
        frames = self._reentrant_frames.pop(micro_index, ())
        for frame in frames:
            frame.release()
        self._dw_cache.pop(micro_index, None)

    def _bind_reentrant(self, micro_index, phase):
        """Bind phase on frames, not on caller thread-local state (autograd uses workers)."""
        for frame in self._reentrant_frames[micro_index]:
            frame.set_phase(phase)

    def _reentrant_backward(self, micro_index, phase):
        """Use public autograd targets while keeping the original output graph for dw."""
        outputs, grad_tensors = self._dw_cache[micro_index]
        if self.retain_backward_graph:
            raise ValueError("Reentrant PP does not support retain_backward_graph: frames cannot be replayed "
                             "after the first backward. Disable reentrant checkpointing for this stage.")
        if phase == "full":
            if not self._has_sharded_params:
                self._bind_reentrant(micro_index, "dw")
            torch.autograd.backward(outputs, grad_tensors=grad_tensors)
            return
        inputs = []
        if phase != "dw":
            inputs = [info.buffer for info in self.args_recv_info[micro_index] if info.requires_grad]
        if self._has_sharded_params:
            raise ValueError("Reentrant PP split with FSDP/HSDP is unsupported: parameter identity "
                             "changes across forward and backward.")
        self._bind_reentrant(micro_index, phase)
        if phase == "dw":
            # The zero-element trigger traverses closure-only checkpoint nodes
            # without adding fake edges to real parameters (which fire hooks on None).
            roots = [*self._get_trainable_params(), *(
                frame.backward_trigger for frame in self._reentrant_frames[micro_index]
            )]
            inputs = list({id(root): root for root in roots}.values())
        if inputs and outputs:
            torch.autograd.backward(outputs, grad_tensors=grad_tensors, inputs=inputs,
                                    retain_graph=phase == "dx")

    def backward_one_chunk(self, micro_index: int) -> None:
        """Execution a backward function.

        ``grad_recv_info`` is filtered to rg=True forward outputs (see
        ``exec_fwd_send_ops``), so ``local_output`` and the matching sens list
        must be filtered the same way — torch.autograd.backward rejects tensors
        without ``grad_fn``.  ``bwd_cache`` is then populated with grads for
        rg=True inputs only, aligning 1:1 with the peer's ``grad_recv_info``.
        """
        if not self._has_backward:
            return
        self._configure_backward(micro_index)
        recv_args = []
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]

        fwd_output = self.fwd_cache.pop(micro_index)
        if self.is_last_stage:
            self.fwd_outputs_cache.pop(micro_index, None)
        local_output = self._filter_grad_outputs(fwd_output)

        if not local_output:
            # Nothing to backprop through (e.g. all forward outputs detached).
            self._release_reentrant(micro_index)
            if not self.is_first_stage:
                self._populate_bwd_cache(micro_index)
            self._clear_recv_buffer(self.grad_recv_info, micro_index)
            self._clear_recv_buffer(self.args_recv_info, micro_index)
            self._finish_backward(micro_index)
            return

        grad_tensors = self._build_last_stage_sens() if self.is_last_stage else recv_args
        if micro_index in self._reentrant_frames:
            try:
                self._dw_cache[micro_index] = (local_output, grad_tensors)
                self._reentrant_backward(micro_index, "full")
                self._finish_backward(micro_index)
                if not self.is_first_stage:
                    self._populate_bwd_cache(micro_index)
            finally:
                self._release_reentrant(micro_index)
                self._clear_recv_buffer(self.grad_recv_info, micro_index)
                self._clear_recv_buffer(self.args_recv_info, micro_index)
            return
        # MPipe owner-backward shares the tower's all-gather node across micro
        # graphs, so freeing it on the first backward breaks the later ones.
        torch.autograd.backward(local_output, grad_tensors=grad_tensors,
                                retain_graph=self.retain_backward_graph)
        self._finish_backward(micro_index)

        if not self.is_first_stage:
            self._populate_bwd_cache(micro_index)
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)

    def backward_input_one_chunk(self, micro_index: int) -> None:
        """Compute input gradients and retain only the graph state needed by dw."""
        if not self._has_backward or self.is_first_stage:
            return

        self._configure_input_backward(micro_index)
        input_values = [
            recv_info.buffer for recv_info in self.args_recv_info[micro_index] if recv_info.requires_grad
        ]
        if not input_values:
            # Preserve the original graph; dw will execute one full backward.
            self._dw_cache[micro_index] = None
            self.bwd_cache[micro_index] = []
            return

        recv_args = []
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]
        fwd_output = self.fwd_cache.pop(micro_index)
        local_output = self._filter_grad_outputs(fwd_output)
        grad_tensors = self._build_last_stage_sens() if self.is_last_stage else recv_args
        if not isinstance(grad_tensors, (list, tuple)):
            grad_tensors = [grad_tensors]
        if micro_index in self._reentrant_frames:
            try:
                self._dw_cache[micro_index] = (local_output, grad_tensors)
                if not local_output:
                    self._dw_cache[micro_index] = ([], [])
                    self._populate_bwd_cache(micro_index)
                    return
                self._reentrant_backward(micro_index, "dx")
                self._populate_bwd_cache(micro_index)
            except BaseException:
                self._release_reentrant(micro_index)
                self._clear_recv_buffer(self.grad_recv_info, micro_index)
                self._clear_recv_buffer(self.args_recv_info, micro_index)
                raise
            return
        _, param_groups = stage_backward_input(
            local_output,
            list(grad_tensors),
            input_values,
            iter(self._get_trainable_params()),
        )
        self._dw_cache[micro_index] = param_groups
        self._populate_bwd_cache(micro_index)

    def backward_weight_one_chunk(self, micro_index: int) -> None:
        """Compute parameter gradients from state captured by input backward."""
        if not self._has_backward:
            return
        if self.is_first_stage:
            self.backward_one_chunk(micro_index)
            return
        if micro_index not in self._dw_cache:
            raise RuntimeError(f"stage: {self.stage_index} micro_{micro_index} dw called before dx.")
        if self._dw_cache[micro_index] is None:
            self._dw_cache.pop(micro_index)
            self.backward_one_chunk(micro_index)
            if self.fsdp_comm_controller is not None and self._has_sharded_params:
                self.fsdp_comm_controller.dx_microbatches[self.stage_index].remove(micro_index)
            return

        self._configure_backward(micro_index)

        if micro_index in self._reentrant_frames:
            try:
                self._reentrant_backward(micro_index, "dw")
            finally:
                self._release_reentrant(micro_index)
                self._clear_recv_buffer(self.grad_recv_info, micro_index)
                self._clear_recv_buffer(self.args_recv_info, micro_index)
                if self.is_last_stage:
                    self.fwd_outputs_cache.pop(micro_index, None)
            return

        stage_backward_weight(iter(self._get_trainable_params()), self._dw_cache[micro_index])
        if self.fsdp_comm_controller is not None and self._has_sharded_params:
            self.fsdp_comm_controller.reduce_dw_gradients(self.stage_index)
            self.fsdp_comm_controller.dx_microbatches[self.stage_index].remove(micro_index)
            self.fsdp_comm_controller.completed_microbatches[self.stage_index].add(micro_index)
        self._dw_cache.pop(micro_index)
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)
        if self.is_last_stage:
            self.fwd_outputs_cache.pop(micro_index, None)

    @staticmethod
    def get_last_stage_sens(last_stage_outputs: Any) -> Any:
        """Get last stage sens"""
        p_sens = None
        if isinstance(last_stage_outputs, (list, tuple)):
            p_sens = []
            for out_i in last_stage_outputs:
                if isinstance(out_i, hyper_parallel.DTensor):
                    repeat_num = out_i.layout.repeat_num()
                    sens_i = torch.full_like(out_i.to_local(), 1.0 / repeat_num)
                else:
                    sens_i = torch.full_like(out_i, 1.0)
                p_sens.append(sens_i)
        else:
            if isinstance(last_stage_outputs, hyper_parallel.DTensor):
                repeat_num = last_stage_outputs.layout.repeat_num()
                p_sens = torch.full_like(last_stage_outputs.to_local(), 1.0 / repeat_num)
            else:
                p_sens = torch.full_like(last_stage_outputs, 1.0)

        return p_sens
