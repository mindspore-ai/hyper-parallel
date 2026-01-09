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
"""mindspore pipeline stage"""
from mindspore import ops


class PipelineStageBase:
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule (Cell): Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule, stage_index, stage_num, group=None, has_backward=True):
        super().__init__()
        self.submodule = submodule
        self.pp_group = self._check_pp_group(group)
        self._backward_func = None
        self.stage_index = stage_index
        self.stage_num = stage_num
        if has_backward:
            self.submodule.set_grad(True)
            self._construct_backward_func()
        self.fwd_inputs_cache = {}
        self.fwd_outputs_cache = {}
        self.bwd_cache = {}

    def clear_cache(self):
        """clear cache."""
        self.fwd_inputs_cache.clear()
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()

    def _check_pp_group(self, group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, str):
            raise TypeError("Argument 'group' must be type of str, but got type of {type(group)}.")
        return group

    def _clear_recv_buffer(self, recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
            info.buffer = None

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
        self.fwd_inputs_cache[micro_index] = (composite_args, composite_kwargs)
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        return out

    def backward_one_chunk(self, micro_index, last_backward=False):
        """Execution a backward function."""
        if not self._has_backward:
            return None
        fwd_args, fwd_kwargs = self.fwd_inputs_cache.pop(micro_index)
        recv_args = []
        if last_backward:
            self.submodule.set_requires_grad_sync(True)
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]
        if self.is_first_stage:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        elif self.is_last_stage:
            sens = self.get_last_stage_sens(self.last_stage_outputs)
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=sens)
        else:
            grad_out = self._backward_func(*fwd_args, **fwd_kwargs, sens=recv_args)
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)
        if not self.is_first_stage:
            self.bwd_cache[micro_index] = grad_out[0][:self._recv_num]
        # return grads for parameters
        if self.is_first_stage:
            return grad_out
        return grad_out[1]

    def _construct_backward_func(self):
        """construct backward func."""
        self._backward_func = None
        if self.is_first_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
        elif self.is_last_stage:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
        else:
            self._backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=True)(
                self.submodule, self.submodule.trainable_params())
