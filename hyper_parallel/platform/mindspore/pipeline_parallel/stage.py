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
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn


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
        self.fwd_outputs_cache = {}
        self.fwd_grad_fn_cache = {}
        self.bwd_cache = {}
        self.last_stage_outputs = None  # Initialized in forward_one_chunk()

    def clear_cache(self):
        """clear cache."""
        self.fwd_outputs_cache.clear()
        self.fwd_grad_fn_cache.clear()
        self.bwd_cache.clear()

    @staticmethod
    def _check_pp_group(group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, str):
            raise TypeError("Argument 'group' must be type of str, but got type of {type(group)}.")
        return group

    @staticmethod
    def _clear_recv_buffer(recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
            info.buffer = None

    @staticmethod
    def _grad_position_from_requires_grad(composite_args):
        """Derive grad_position from composite_args' requires_grad attributes.

        Returns -1 if all tensor args require grad, a tuple of indices if some do,
        and an empty list if none do.
        """
        # pylint: disable=C0415
        from mindspore import Tensor
        tensor_indices = [i for i, a in enumerate(composite_args) if isinstance(a, Tensor)]
        requires_grad_indices = [
            i for i in tensor_indices
            if composite_args[i]._requires_grad  # pylint: disable=protected-access
        ]
        if not requires_grad_indices:
            return []
        if len(requires_grad_indices) == len(tensor_indices):
            return -1
        return tuple(requires_grad_indices)

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
        from hyper_parallel.core.fully_shard.api import HSDPModule  # pylint: disable=C0415
        for _, mod in self.submodule.cells_and_names():
            if not isinstance(mod, HSDPModule):
                continue
            mod.set_reshard_after_forward(False)
        if self.is_first_stage:
            composite_args = args
        else:
            if micro_index in self.args_recv_info:
                composite_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
            else:
                raise RuntimeError(f"The exec order is wrong. The corresponding forward calculation \
                                    is executed before the Receive operation. micro is {micro_index}.")
        composite_kwargs = kwargs or {}
        if self._has_backward:
            grad_position = self._grad_position_from_requires_grad(composite_args)
            weights = tuple(self.submodule.trainable_params())
            out, grad_fn = forward_and_gradfn(
                self.submodule,
                *composite_args,
                weights=weights,
                grad_position=grad_position,
                **composite_kwargs,
            )
            self.fwd_grad_fn_cache[micro_index] = grad_fn
        else:
            out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        return out

    def backward_one_chunk(self, micro_index):
        """Execution a backward function."""
        from hyper_parallel.core.fully_shard.api import HSDPModule  # pylint: disable=C0415
        if not self._has_backward:
            return
        for _, mod in self.submodule.cells_and_names():
            if not isinstance(mod, HSDPModule):
                continue
            mod.set_reshard_after_backward(False)
            mod.set_requires_gradient_sync(False)

        grad_fn = self.fwd_grad_fn_cache.pop(micro_index)
        if self.is_first_stage:
            sens = self._build_padded_sens(micro_index)
            _ = grad_fn(sens=sens)
        else:
            if self.is_last_stage:
                sens = self.get_last_stage_sens(self.last_stage_outputs)
            else:
                sens = self._build_padded_sens(micro_index)
            _ = grad_fn(sens=sens)
        if not self.is_first_stage:
            input_grads = [recv_info.buffer.grad for recv_info in self.args_recv_info[micro_index]
                           if recv_info.requires_grad]
            self.bwd_cache[micro_index] = input_grads
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)

    def _construct_backward_func(self):
        """construct backward func."""
        enable_mindspore_backward_compat()
        self._backward_func = None
