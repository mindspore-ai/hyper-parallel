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
import torch
import torch.distributed as dist

import hyper_parallel


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
        self.last_stage_outputs = None  # Initialized in forward_one_chunk()

    def clear_cache(self):
        """clear cache."""
        self.fwd_outputs_cache.clear()
        self.bwd_cache.clear()
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
    def is_first_stage(self):
        """return if is first stage."""
        return self.stage_index == 0

    @property
    def is_last_stage(self):
        """return if is last stage."""
        return self.stage_index == self.stage_num - 1

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function."""
        # PP drives FSDP unshard/reshard through injected FSDP MetaSteps, so the
        # nested HSDPModules must NOT reshard after their own forward hook.  GPipe
        # runs every micro-batch forward before any backward; if a module resharded
        # after micro-batch 0, the next micro-batch's ``_assert_in_unshard_if_needed``
        # would fail (parameters already freed). Keep them unsharded for the whole
        # stage; the FSDP_RESHARD MetaStep frees them explicitly afterwards.
        for _, mod in self.submodule.named_modules():
            if isinstance(mod, hyper_parallel.HSDPModule):
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
        out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_cache[micro_index] = out_tuple
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
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
        for s, o in zip(sens_all, outputs_iter):
            o_local = o.to_local() if isinstance(o, hyper_parallel.DTensor) else o
            if o_local.requires_grad:
                sens.append(s)
        return sens

    def _populate_bwd_cache(self, micro_index):
        """Stash rg=True input grads so they align with peer's grad_recv_info."""
        input_grads = [recv_info.buffer.grad
                       for recv_info in self.args_recv_info[micro_index]
                       if recv_info.requires_grad]
        self.bwd_cache[micro_index] = input_grads

    def backward_one_chunk(self, micro_index):
        """Execution a backward function.

        ``grad_recv_info`` is filtered to rg=True forward outputs (see
        ``exec_fwd_send_ops``), so ``local_output`` and the matching sens list
        must be filtered the same way — torch.autograd.backward rejects tensors
        without ``grad_fn``.  ``bwd_cache`` is then populated with grads for
        rg=True inputs only, aligning 1:1 with the peer's ``grad_recv_info``.
        """
        if not self._has_backward:
            return
        # Accumulate per-micro-batch gradients locally without resharding or
        # reducing.  The schedule's FSDP_REDUCE_GRAD MetaStep
        # (PipelineStage.execute_reduce_grad) re-enables both flags and performs
        # the single reduce-scatter once after the last micro-batch's backward.
        # Per-micro-batch reduce here would conflict with that explicit reduce and
        # over-count the gradient.
        for _, mod in self.submodule.named_modules():
            if isinstance(mod, hyper_parallel.HSDPModule):
                mod.set_reshard_after_backward(False)
                mod.set_requires_gradient_sync(False)
        recv_args = []
        if micro_index in self.grad_recv_info:
            recv_args = [recv_info.buffer for recv_info in self.grad_recv_info[micro_index]]

        fwd_output = self.fwd_cache.pop(micro_index)
        if self.is_last_stage:
            self.fwd_outputs_cache.pop(micro_index, None)
        local_output = self._filter_grad_outputs(fwd_output)

        if not local_output:
            # Nothing to backprop through (e.g. all forward outputs detached).
            self._clear_recv_buffer(self.grad_recv_info, micro_index)
            self._clear_recv_buffer(self.args_recv_info, micro_index)
            return

        grad_tensors = self._build_last_stage_sens() if self.is_last_stage else recv_args
        torch.autograd.backward(local_output, grad_tensors=grad_tensors)

        if not self.is_first_stage:
            self._populate_bwd_cache(micro_index)
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)
