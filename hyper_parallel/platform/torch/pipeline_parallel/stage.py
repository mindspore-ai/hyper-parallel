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
"""pipeline stage"""
import hyper_parallel
import torch
import torch.distributed as dist


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
    def __init__(self, submodule, stage_index, stage_num, group=None, dyn_shape=False, has_backward=True):
        self.submodule = submodule
        self.fwd_cache = {}
        self.bwd_cache = {}
        self.meta_cache = []
        self._dyn_shape = dyn_shape
        self._has_backward = has_backward
        self.group = self._check_pp_group(group)
        self.stage_index = stage_index
        self.stage_num = stage_num
        self.fwd_outputs_cache = {}

    def clear_cache(self):
        """clear cache."""
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()
        self._meta_cache.clear()

    def _clear_recv_buffer(self, recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
            info.buffer = None

    def _check_pp_group(self, group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, dist.ProcessGroup):
            raise TypeError("Argument 'group' must be type of ProcessGroup, but got type of {type(group)}.")
        return group

    @property
    def is_first_stage(self):
        """return if is first stage."""
        return self.stage_index == 0

    @property
    def is_last_stage(self):
        """return if is last stage."""
        return self.stage_index == self.stage_num - 1

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function."""
        if self.is_first_stage:
            composite_args = args
        else:
            if micro_index in self.args_recv_info:
                composite_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
            else:
                raise RuntimeError(f"The exec order is wrong. The corresponding forward calculation \
                                    is executed before the Receive operation. micro is {micro_index}.")
        composite_kwargs = kwargs or {}
        out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_cache[micro_index] = out_tuple
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        return out

    def backward_one_chunk(self, micro_index, last_backward=False):
        """Execution a backward function."""
        if not self._has_backward:
            return
        recv_args = []
        if last_backward and isinstance(self.submodule, hyper_parallel.HSDPCell):
            self.submodule.set_requires_grad_sync(True)
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]

        fwd_output = self.fwd_cache.pop(micro_index)
        local_output = []
        # need local_tensor to get gradient.
        for each_out in fwd_output:
            if isinstance(each_out, hyper_parallel.DTensor):
                local_output.append(each_out.to_local())
            else:
                local_output.append(each_out)
        if self.is_first_stage:
            torch.autograd.backward(local_output, grad_tensors=recv_args)
        elif self.is_last_stage:
            sens = self.get_last_stage_sens(self.last_stage_outputs)
            torch.autograd.backward(local_output, grad_tensors=sens)
        else:
            torch.autograd.backward(local_output, grad_tensors=recv_args)
        if not self.is_first_stage:
            input_grads = [recv_info.buffer.grad for recv_info in self.args_recv_info[micro_index]]
            self.bwd_cache[micro_index] = input_grads
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)
